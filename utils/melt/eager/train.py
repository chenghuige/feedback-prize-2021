#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2018-09-03 15:40:04.947138
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
import torch.distributed as dist

# https://github.com/huggingface/transformers/issues/1801
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

import tensorflow as tf 
from absl import flags
FLAGS = flags.FLAGS

from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

# https://github.com/tensorflow/models/blob/master/official/utils/misc/keras_utils.py
from tensorflow.python.eager import profiler

# tfe = tf.contrib.eager

import sys 
import os
import numpy as np
import inspect
import traceback
import copy
import itertools
import time
import math
import wandb
import multiprocessing as mp
from multiprocessing import Manager
try:
  import pymp
except Exception:
  pass

import gezi
from gezi.summary import SummaryWriter
import melt
logging = melt.logging
# from tqdm.notebook import tqdm
from gezi import tqdm

try:
  from lele.distributed.parallel import DataParallelModel, DataParallelCriterion
except Exception:
  pass

from melt.flow.flow import _try_eval, _on_epoch_end, _async_valid
import transformers
from transformers import create_optimizer
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

#TODO custom loop 需要仿照keras，lighting 重构一下 引入call back简化loop 长度。。。
# projects/ai/qqbrowser/baseline/tensorflow/train.py 参考这个 速度很快 分析一下目前慢的原因
# 支持动态输入 tf.function 输入shape None 不多次构图
# 支持gradient acc
# --loop_only strategy.run(train_step) 就非常慢 比正常loop dataset慢很多 而--gpus=1 就速度正常了
# 但是即使单gpu 如果不是loop_only stategy.run也非常慢 why
# 需要 train_dataset = strategy.experimental_distribute_dataset(train_dataset) ?
# https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy

def torch_(x):
  # if FLAGS.torch_only:
  #   return x
  for dim in x.shape:
    if dim == 0:
      return x

  # if  tf.__version__ < '2':
  x = x.numpy()

  if x.dtype == np.int64 or x.dtype == np.int32 or x.dtype == np.float32 or x.dtype == np.float64:
    x = torch.as_tensor(x)
    x = x.to(device)

  return x

def to_torch(x, y=None):
  if FLAGS.torch_only:
    for key in x:
      if isinstance(x[key], dict):
        x[key] = to_torch(x[key])
      else:
        if type(x[key][0]) not in [np.str_, str]:
          x[key] = x[key].to(device)
    if y is None:
      return x
    else:
      return x, y.to(device)
  
  if y is not None:
    y = torch_(y)

  if not isinstance(x, dict):
    x = torch_(x)
  else:
    for key in x:
      x[key] = to_torch(x[key])
      
  if y is None:
    return x
  else:
    return x, y

def filter_nontorch(x, cpu=False):
  if not isinstance(x, dict):
    if not torch.is_tensor(x):
      return None
    else:
      if cpu:
        return x.cpu()
      else:
        return x
  else:
    keys = list(x.keys())
    for key in keys:
      x[key] = filter_nontorch(x[key])
      if x[key] is None:
        del x[key]
    return x


def eval(eval_fn, labels, predicts, other, step, ofile=None, show=True, do_eval=True, 
         write=True, write_fn=None, num_examples=None, kwargs={}):
  if do_eval:
    try:
      results = eval_fn(labels, predicts, **kwargs)
    except Exception:
      logging.warning('eval fn error')
      logging.warning(traceback.format_exc())
      results = None
  
  if write and ofile:
    kwargs_write = {}
    write_args = inspect.getargspec(write_fn).args 
    #ids = ids[0] if len(keys) == 1 else ids
    logging.info(f'write {len(labels)} valid result for each valid instance to', ofile)
    with open(ofile, 'w', encoding='utf-8') as out:
      if 'ids' in write_args:
        ids = ids[0] if len(keys) == 1 else kwargs['x']
        if 'others' in write_args:
          kwargs_write['others'] = dict([(key, other[key]) for key in other])
        write_fn(ids, labels, predicts, out, **kwargs_write)
      else:
        for i, (id, label, predict) in tqdm(enumerate(zip(zip(*ids), labels, predicts)), total=len(labels), ascii=False):
          id = sep.join(map(str, id))
          if 'other' in write_args:
            kwargs_write['other'] = dict([(key, other[key][i]) for key in other])
          write_fn(id, label, predict, out, **kwargs_write)

  if show:
    if results:
      results_ = type(results)(list(results.items())[:FLAGS.max_metrics_show])
      loss_dict = {'loss': gezi.get('loss'), 'val_loss': gezi.get('valid_loss')}
      gezi.pprint_dict(gezi.dict_rename(results_, 'Metrics/', ''), print_fn=logging.info, 
                      desc=f'[{FLAGS.model_name}] ' + 'epoch:%.2f/%d' % (gezi.epoch(), FLAGS.num_epochs) +  f' {loss_dict}',
                      format='%.4f')
      
      results['epoch'] = gezi.epoch()
      results['insts'] = num_examples
      results['step'] = step
      writer = gezi.DfWriter(FLAGS.log_dir, filename='metrics.csv')
      writer.write(gezi.dict_rename(results, 'Metrics/', ''))
      if FLAGS.write_summary:
        summary = melt.get_summary_writer()
        summary.log(results)
      del results['step']
      results['Metrics/step'] = step
      if FLAGS.wandb:
        try:
          wandb.log(results)
        except Exception:
          pass
        
def evaluate(model, dataset, eval_fn, model_path=None, 
             names=None, write_fn=None, write_streaming=False,
             num_steps=None, num_examples=None, write_valid_only=False, ofile=None,
             suffix='.valid', keys=None, step=None, sep=',', out_hook=None, out_keys=[], str_keys=[], 
             write=True, is_last=False, show=True): 
  
  rank = FLAGS.local_rank
  # if not keys:
  #   keys = ['id']
  if isinstance(keys, str):
    keys = keys.split(',')
  
  if hasattr(model, 'eval'):
    model.eval()

  if not write_fn:
    write_streaming = True

  predicts_list = []
  labels_list = []
  ids_list = [[] for _ in range(len(keys))]
  others_list = None

  if not ofile:
    ofile = model_path + suffix if model_path and write else None
  if write_streaming:
    out = open(ofile, 'w', encoding='utf-8') if ofile else None
    if out:
      if names is not None:
        print(*names, sep=sep, file=out)
  else:
    out = None

  # TODO pass model is useless...
  if write_fn:
    kwargs_write = {} 
    write_args = inspect.getargspec(write_fn).args 

  timer_ = gezi.Timer()
  # desc = 'eval' if not FLAGS.valid_hour else f'{FLAGS.valid_hour}-{FLAGS.eval_round}-eval'
  desc = 'Eval Predicting'
  with torch.no_grad():
    for step, (x, y) in tqdm(enumerate(dataset), total=num_steps, desc=desc, ascii=False, leave=FLAGS.eval_leave):
      gezi.set_global_step(step)
      keys = [key for key in keys if key in x]
      
      if num_steps and step == num_steps:
        break 

      if FLAGS.torch:
        x, y = to_torch(x, y)

      if not FLAGS.torch and  'training' in inspect.getfullargspec(model.call).args:
        predicts = model(x, training=False)
      else:
        predicts = model(x)
      other = out_hook(model) if out_hook else melt.out_hook(model, out_keys)
      if isinstance(predicts, dict):
        for key in predicts:
          if key != 'pred' and not isinstance(predicts[key], (list, tuple, dict)):
            other[key] = predicts[key]
        predicts = predicts['pred']

      if not others_list:
        others_list = [[] for _ in range(len(other))]

      if FLAGS.torch:
        predicts = predicts.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
      else:
        predicts = predicts.numpy() 
        y = y.numpy()
        for key in keys:
          x[key] = x[key].numpy() 
          if key in str_keys:
            x[key] = gezi.decode(x[key])
        for key in other:
          other[key] = other[key].numpy() 

      for key in str_keys:
        if key in x:
          x[key] = gezi.decode(x[key])
        
      predicts_list.append(gezi.squeeze(predicts))
      labels_list.append(gezi.squeeze(y))
      ids = [gezi.squeeze(x[key]) for key in keys]
      
      if FLAGS.torch:
        for i in range(len(keys)):
          try:
            ids[i] = ids[i].detach().cpu().numpy()
          except Exception:
            pass

        for key in other:
          other[key] = other[key].detach().cpu().numpy()

      for i in range(len(keys)):
        ids_list[i].append(ids[i])

      for i, key in enumerate(other): 
        others_list[i].append(gezi.squeeze(other[key]))
    
  FLAGS.valid_time = timer_.elapsed_minutes()

  if out:
    out.close()

  if hasattr(model, 'train'):
    model.train()

  try:
    ids = [np.concatenate(ids_list[i])[:num_examples] for i in range(len(keys))]
    predicts = np.concatenate(predicts_list)[:num_examples]
    labels = np.concatenate(labels_list)[:num_examples]
    others = [np.concatenate(others_list[i])[:num_examples] for i in range(len(other))]
  except Exception:
    ids = [list(itertools.chain(*ids_list[i]))[:num_examples] for i in range(len(keys))]
    predicts = list(itertools.chain(*predicts_list))[:num_examples]
    labels = list(itertools.chain(*labels_list))[:num_examples]
    others =[list(itertools.chain(*others_list[i]))[:num_examples] for i in range(len(other))]

  if FLAGS.work_mode != 'train':
    if FLAGS.torch:
      # TODO seems not free, still 1851M 
      torch.cuda.synchronize()
      torch.cuda.empty_cache()
      import gc
      gc.collect()
    else:
      pass
 
  for i, key in enumerate(other): 
    other[key] = others[i]
  
  results = None
  if rank == 0:
    kwargs = {}
    args = inspect.getargspec(eval_fn).args    
    if 'model_path' in args:
      kwargs['model_path'] = model_path
    if 'ids' in args:
      kwargs['ids'] = ids
    if 'info' in args:
      kwargs['info'] = dict(zip(keys, ids))
    if 'x' in args:
      kwargs['x'] = dict(zip(keys, ids))
    if 'model' in args:
      kwargs['model'] = model
    if 'other' in args:
      kwargs['other'] = other
    if 'is_last' in args:
      kwargs['is_last'] = is_last

    # TODO check here, even with comm.barrier might still hang
    do_eval = not write_valid_only
    write = not write_streaming and ofile
    step = melt.get_eval_step()
    melt.inc_eval_step()
    if ((not FLAGS.async_eval) or is_last) and (not FLAGS.async_eval_last):
      results = eval(eval_fn, labels, predicts, other, step, ofile, show=True, 
                    do_eval=do_eval, write=write, write_fn=write_fn, 
                    num_examples=num_examples, kwargs=kwargs)
    else:
      p = mp.Process(target=eval, args=(eval_fn, labels, predicts, other, step, ofile, True, do_eval, write, write_fn, num_examples, kwargs))
      p.start()
  return results

def inference(model, dataset, model_path, 
              names=None, debug_names=None, 
              write_fn=None, write_streaming=False,
              num_steps=None, num_examples=None,
              suffix='.infer', sep=','):

  if hasattr(model, 'eval'):
    model.eval()
  if not write_fn:
    write_streaming = True
  
  assert model_path
  ofile = model_path + suffix
  ofile2 = ofile + '.debug'
  if write_streaming:
    if write_fn and len(inspect.getargspec(write_fn).args) == 4:
      out_debug = open(ofile2, 'w', encoding='utf-8')
    else:
      out_debug = None
    out = open(ofile, 'w', encoding='utf-8') 
  else:
    out = None
    out_debug = None
  
  if write_streaming:
    if names is not None:
      print(*names, sep=sep, file=out)
    if debug_names and out_debug:
      print(*debug_names, sep=sep, file=out_debug)

  predicts_list = []
  ids_list = []
  with torch.no_grad():
    for (x, _) in tqdm(dataset, total=num_steps, desc='test', ascii=False):
      if FLAGS.torch:
        x = to_torch(x)
      if not FLAGS.torch and  'training' in inspect.getfullargspec(model.call).args:
        predicts = model(x, training=False)
      else:
        predicts = model(x)
      if FLAGS.torch:
        predicts = predicts.detach().cpu()
      else:
        predicts = predicts.numpy()
      # here id is str in py3 will be bytes
      if not FLAGS.torch:
        x['id'] = x['id'].numpy()

      # ids = gezi.decode(x['id'])

      ids = gezi.squeeze(ids)
      predicts = gezi.squeeze(predicts)

      if not write_streaming:
        predicts_list.append(predicts)
        ids_list.append(ids)
      else:
        for id, predict in zip(ids, predicts.numpy()):
          if write_fn is None:
            if not gezi.iterable(predict):
              predict = [predict]
            print(id, *predict, sep=sep, file=out)
          else:
            if out_debug:
              write_fn(id, predict, out, out_debug)
            else:
              write_fn(id, predict, out)
    
  if out:
    out.close()
  if out_debug:
    out_debug.close()

  if not write_streaming:
    try:
      # concat list so like [[512,], [512,]...] -> [512 * num_batchs]
      # ore [[512, 3], [512,3] ..] -> [512 * num_batchs, 3]
      ids = np.concatenate(ids_list)[:num_examples]
    except Exception:
      ids = ['0'] * num_examples
    predicts = np.concatenate(predicts_list)[:num_examples]

    if len(inspect.getargspec(write_fn).args) == 4:
      write_fn(ids, predicts, ofile, ofile2)
    else:
      write_fn(ids, predicts, ofile)

def load_torch_model(path, model=None, optimizer=None, map_location=None, includes=None, excludes=None):
  checkpoint = torch.load(path, map_location=map_location)

  if optimizer:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])

  state = checkpoint['state_dict']   

  # TODO just use load_state_dict with strict=False ?
  new_state = {}
  model_ = model.module if hasattr(model, 'module') else model

  def is_ok(key):
    if includes:
      for incl_key in includes:
        if incl_key in key:
          return True
      return False 
    if excludes:
      for excl_key in excludes:
        if excl_key in key:
          return False
    return True

  for key, val in state.items():
    if key in model_.state_dict() and is_ok(key):
      logging.debug(f'update key: {key}')
      new_state[key] = val
  checkpoint['epoch'] = checkpoint.get('epoch', 0)
  checkpoint['step'] = checkpoint.get('step', 0)
  logging.debug('num updated keys from checkpoint', len(new_state), 'epoch:', checkpoint['epoch'], 'step:', checkpoint['step'])

  # this is for model state has more params then loaded so just partial update mode state with key,vals from loaded     
  new_params = model_.state_dict()
  new_params.update(new_state)
  model_.load_state_dict(new_params)

  model.eval()

  return checkpoint

# TODO support FLAGS.optimizers
def get_torch_optimizer(optimizer, model, num_steps_per_epoch=None, params=None):

  if FLAGS.work_mode != 'train':
    return None, None

  if hasattr(model, 'module'):
    model = model.module

  def _get_bert_infos():
    # ic(num_steps_per_epoch, FLAGS.num_decay_epochs, FLAGS.num_epochs)
    num_train_steps = int(num_steps_per_epoch * (FLAGS.num_decay_epochs or FLAGS.num_epochs))
    num_warmup_steps = FLAGS.warmup_steps or (FLAGS.warmup_epochs and int(num_steps_per_epoch * FLAGS.warmup_epochs)) \
                          or (FLAGS.warmup_proportion and int(num_train_steps * FLAGS.warmup_proportion))
    # assert num_warmup_steps
    return num_train_steps, num_warmup_steps

  def _update_scheduler(optimizer, scheduler):
    if scheduler == 'bert':
        num_train_steps, num_warmup_steps = _get_bert_infos()
        optimizer.update(num_train_steps + 1, num_warmup_steps)

  def _update_schedulers(optimizer, schedulers):
    if schedulers:
      if not hasattr(optimizer, 'optimizers'):
        assert len(schedulers) == 1, len(schedulers)
        _update_scheduler(optimizer, schedulers[0])
      else:
        assert len(schedulers) == len(optimizer.optimizers), f'{len(schedulers)} {len(optimizer.optimizers)}'
        for op, scheduler in zip(optimizer.optimizers, schedulers):
          _update_scheduler(op, scheduler)
  scheduler = None    
  if FLAGS.round == 0:
    if optimizer is None or isinstance(optimizer, str):
      params = params or model.parameters()
      import lele
      optimizer_ = optimizer if isinstance(optimizer, str) else FLAGS.optimizer
      names = optimizer_.split('-')
      scheduler = None if len(names) == 1 else names[0]
      gezi.add_global('schedulers', scheduler)
      optimizer_name = names[-1] 
      if optimizer_name.lower() == 'adabelief':
        from adabelief_pytorch import AdaBelief as Optimizer
      elif optimizer_name.lower() == 'adamw':
        from transformers import AdamW
        Optimizer = AdamW
      else:
        Optimizer = getattr(torch.optim, optimizer_name)
      
      kwargs = {'lr': FLAGS.learning_rate} 
      
      args = inspect.getfullargspec(Optimizer).args
      if 'eps' in args:
        kwargs['eps'] = FLAGS.opt_epsilon
      if 'weight_decay' in args:
        kwargs['weight_decay'] = FLAGS.opt_weight_decay
      if 'amsgrad' in args:
        kwargs['amsgrad'] = FLAGS.opt_amsgrad
      if 'momentum' in args:
        kwargs['momentum'] = FLAGS.opt_momentum 
      if 'correct_bias' in args:
        kwargs['correct_bias'] = False
      # if 'correct_bias' in args:
      #   kwargs['correct_bias'] = True

      logging.debug(f'Optimizer {Optimizer} with args {kwargs} lr:{FLAGS.learning_rate}, lrs:{FLAGS.learning_rates}')
      optimizer = Optimizer(params, **kwargs)
      # ic(params)
      ic(optimizer)

      if scheduler == 'noam':
        optimizer = lele.training.optimizers.NoamOpt(128, 2, 4000, optimizer)
      elif scheduler == 'bert':
        num_train_steps, num_warmup_steps = _get_bert_infos()
        logging.debug('num_train_steps', num_train_steps, 'num_warmup_steps', num_warmup_steps, 'warmup_proportion', FLAGS.warmup_proportion)
        #optimizer = Optimizer(model.parameters(), lr=0)
        # optimizer = lele.training.optimizers.BertOpt(
        #                     FLAGS.learning_rate, 
        #                     FLAGS.min_learning_rate,
        #                     num_train_steps + 1,
        #                     num_warmup_steps,
        #                     optimizer,
        #                     power=FLAGS.learning_rate_decay_power
        #                     )
        scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, 
                                                                   num_warmup_steps=num_warmup_steps, 
                                                                   num_training_steps=num_train_steps + 1,
                                                                   power=FLAGS.learning_rate_decay_power)
      elif scheduler == 'cosine':
        num_train_steps, num_warmup_steps = _get_bert_infos()
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=1,
            last_epoch=-1,
        )
    elif callable(optimizer):
      kwargs = {}
      if 'num_steps_per_epoch' in inspect.getargspec(optimizer).args:
        kwargs['num_steps_per_epoch'] = num_steps_per_epoch
      optimizer = optimizer(model, **kwargs) 

    gezi.set_global('optimizer', optimizer)
  else:
    optimizer = gezi.get_global('optimizer')
    _update_schedulers(optimizer, gezi.get_global('schedulers', None))

  return optimizer, scheduler

def get_l2_sum(model):
  if not FLAGS.torch:
    l2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables]).numpy() 
  else:
    l2_fn = torch.nn.MSELoss(reduction='sum')
    l2 = sum(l2_fn(p, torch.zeros_like(p)).item() for p in model.parameters() if p.requires_grad) / 2.
  return l2

def get_total_params(model):
  if not FLAGS.torch:
    total_params = model.count_params()
  else:
    model = model if not hasattr(model, 'module') else model.module
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return total_params

# 理论上torch和keras可以复用，因为目前torch的 bert三角学习率是在optimzier外层又封装了一个class loop流程保持lr透明
# 这里目前只用于keras的学习率调整
# TODO move to specific files..
def triangle_learning_rate(init_lr, global_step, decay_steps, warmup_steps, min_learning_rate=0.):
  # https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/training/learning_rate_decay.py#L157-L255
  power = 1.0 
  end_learning_rate = min_learning_rate
  global_step = min(global_step, decay_steps)
  learning_rate = (init_lr - end_learning_rate) * \
                   pow((1 - global_step / decay_steps), power) + \
                   end_learning_rate
  if warmup_steps:
    warmup_percent_done = global_step / warmup_steps
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = global_step < warmup_steps
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
  learning_rate = max(learning_rate, min_learning_rate)
  return learning_rate

def is_triangle_learning_rate(name=None):
  if not name:
    name = FLAGS.optimizer

  if name.startswith('triangle') or name.startswith('bert'):
    return True
  return False

# TODO support more decay
def adjust_learning_rate(init_lr, global_step, num_steps_per_epoch, min_learning_rate=0.):
  if is_triangle_learning_rate():
    decay_steps = int(
        num_steps_per_epoch * (FLAGS.num_decay_epochs or FLAGS.num_epochs))
    if FLAGS.warmup_steps:
      warmup_steps = FLAGS.warmup_steps 
    elif FLAGS.warmup_epochs:
      warmup_steps = num_steps_per_epoch * FLAGS.warmup_epochs
    elif FLAGS.warmup_proportion:
      warmup_steps = int(decay_steps * FLAGS.warmup_proportion) 
    else:
      warmup_steps = 0
    lr = triangle_learning_rate(init_lr, global_step, decay_steps, warmup_steps, min_learning_rate)
    return lr

  return init_lr

def prepare_decay_learning_rate(num_steps_per_epoch):
  if FLAGS.learning_rate_decay_factor > 0:
    #assert FLAGS.learning_rate_values is None, 'use exponential_decay or piecewise_constant?'
    #NOTICE if you do finetune or other things which might change batch_size then you'd better direclty set num_steps_per_decay
    #since global step / decay_steps will not be correct epoch as num_steps per epoch changed
    #so if if you change batch set you have to reset global step as fixed step
    assert FLAGS.num_steps_per_decay or (FLAGS.num_epochs_per_decay and num_steps_per_epoch), 'must set num_steps_per_epoch or num_epochs_per_decay and num_steps_per_epoch'
    decay_steps = FLAGS.num_steps_per_decay or int(num_steps_per_epoch * FLAGS.num_epochs_per_decay)    
    decay_start_step = FLAGS.decay_start_step or int(num_steps_per_epoch * FLAGS.decay_start_epoch)
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    logging.debug('learning_rate_decay_factor:{} decay_epochs:{} decay_steps:{} decay_start_epoch:{} decay_start_step:{}'.format(
        FLAGS.learning_rate_decay_factor, FLAGS.num_epochs_per_decay, decay_steps, FLAGS.decay_start_epoch, decay_start_step))
  else:
    decay_steps = 0
    decay_start_step = 0

  return decay_steps, decay_start_step

def count_examples(Dataset):
  logging.debug('-------count examples')
  batch_size = 10000
  if (FLAGS.valid_input or FLAGS.valid_files):
    dir = gezi.dirname(FLAGS.valid_input)
    file = os.path.join(dir, 'num_records.txt') 

    if os.path.exists(file):
      logging.info('exists', file)
    else:
      dataset = Dataset('valid')
      dataset = dataset.make_batch(batch_size, repeat=False, shuffle=False)
      
      num_examples = 0
      for item in tqdm(dataset, total=int(3000000 / batch_size)):
        num_examples += item[1].numpy().shape[0]
      
      logging.info(f'wrinting {num_examples} to {file}')

      gezi.write_to_txt(num_examples, file)

  if FLAGS.test_input:
    dir = gezi.dirname(FLAGS.test_input)
    file = os.path.join(dir, 'num_records.txt') 

    if os.path.exists(file):
      logging.info('exists', file)
    else:
      dataset = Dataset('test')
      dataset = dataset.make_batch(batch_size, repeat=False, shuffle=False)
      
      num_examples = 0
      for item in tqdm(dataset, total=int(3000000 / batch_size)):
        num_examples += item[1].numpy().shape[0]

      logging.debug(f'wrinting {num_examples} to {file}')

      gezi.write_to_txt(num_examples, file)
  
  dir = gezi.dirname(FLAGS.train_input)
  file = os.path.join(dir, 'num_records.txt') 
  if os.path.exists(file):
    logging.info('exists', file)
  else:
    dataset = Dataset('train')
    dataset = dataset.make_batch(batch_size, repeat=False, shuffle=False)
    
    num_examples = 0
    for item in tqdm(dataset, total=int(3000000 / batch_size)):
      num_examples += item[1].numpy().shape[0]

    logging.info(f'wrinting {num_examples} to {file}')

    gezi.write_to_txt(num_examples, file)

class PytObj(object):
  def __init__(self, x):
    self.x = x

  def numpy(self):
    return self.x

class PytMean(object):
  def __init__(self):
    self._val = 0. 
    self.count = 0

    self.is_call = True

  def clear(self):
    self._val = 0
    self.count = 0

  def __call__(self, val):
    if not self.is_call:
      self.clear()
      self.is_call = True
    self._val += val.item()
    self.count += 1

  def result(self, write_summary=False):
    if self.is_call:
      self.is_call = False
    if not self.count:
      val = 0
    else:
      val = self._val / self.count
    # TODO just for compact with tf ..
    return PytObj(val)

def train(model, 
          loss_fn=None,
          Dataset=None,  
          dataset=None,
          valid_dataset=None,
          valid_dataset2=None,
          test_dataset=None,
          eval_info_dataset=None,
          test_info_dataset=None,
          gen_dataset_fn=None,
          evaluate_fn=None, 
          inference_fn=None,
          eval_fn=None,
          eval_keys=[],
          write_valid=None,
          valid_names=None,
          infer_names=None,
          infer_debug_names=None,
          valid_write_fn=None,
          infer_write_fn=None,
          valid_suffix='.valid',
          infer_suffix='.infer',
          write_streaming=False,
          optimizer=None,
          variables_list_fn=None,
          lr_params=None,
          init_fn=None,
          weights=1.0, 
          sep=',',
          out_hook=None,
          out_keys=[],
          str_keys=[],
          callbacks=[],
          metrics=[],
          initial_epoch=0,
          return_info=False, 
          dry_run=False):
          
  wandb_run = gezi.get('wandb_run')
  lr_params = lr_params or gezi.get('lr_params')
  global device
  if weights is None:
    weights = 1.

  rank = FLAGS.local_rank
  is_eager = tf.executing_eagerly()
  
  write_valid = write_valid if write_valid is not None else FLAGS.write_valid
  
  if not loss_fn and hasattr(model, 'get_loss_fn'):
    loss_fn = model.get_loss_fn()

  if not hasattr(model, 'eval_keys'):
    model.eval_keys = eval_keys
  else:
    model.eval_keys = model.eval_keys or eval_keys
  eval_keys = model.eval_keys

  if not hasattr(model, 'out_keys'):
    model.out_keys = out_keys
  else:
    model.out_keys = model.out_keys or out_keys
  out_keys = model.out_keys   
  
  if not hasattr(model, 'str_keys'):
    model.str_keys = str_keys
  else:
    model.str_keys = model.str_keys or str_keys
  str_keys = model.str_keys   
  logging.debug('eval_keys:', model.eval_keys, 'out_keys:', model.out_keys, 'str_keys:', model.str_keys)

  if FLAGS.keras:
    try:
      eval_keys = model.eval_keys
    except Exception:
      logging.error(traceback.format_exc())
      logging.info('For keras use melt.Model instead of keras.Model')

  if Dataset is None:
    assert dataset is not None
  elif FLAGS.work_mode == 'count':
    count_examples(Dataset)
    exit(0)

  if rank == 0:
    logging.debug('Dataset', Dataset, 'dataset', dataset, 'valid_dataset', valid_dataset, 'test_dataset', test_dataset, loss_fn)

  if FLAGS.torch:
    global device, use_gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_gpu = torch.cuda.is_available()
    if FLAGS.plot_graph:
      device = torch.device("cpu")
      use_gpu = False
    if FLAGS.round == 0:
      if rank == 0:
        # print_fn = logging.info if FLAGS.round == 0 and FLAGS.work_mode == 'train' else logging.debug
        print_fn = logging.debug
        print_fn(model) # keras will show model after first training step
      if 'SHOW' in os.environ or FLAGS.work_mode == 'show':
        exit(0)

      dev_count = torch.cuda.device_count()    
      torch.manual_seed(FLAGS.seed or 0)
      if dev_count > 1:
        torch.cuda.manual_seed_all(FLAGS.seed or 0)
    
      logging.debug('torch.cuda.device_count', dev_count, 'FLAGS.num_gpus', FLAGS.num_gpus)
      if use_gpu:
        if not FLAGS.distributed:
          # https://github.com/pytorch/pytorch/issues/2230
          gpus = gezi.get('gpus')
          gpus = sorted(gpus)
          torch.cuda.set_device(gpus[0]) 
          device = torch.device('cuda', gpus[0])
          ic(gpus, device)
        else:
          if dev_count > 1:
            torch.cuda.set_device(rank)
          else:
            torch.cuda.set_device(0)
      
      if FLAGS.distributed:        
        rank = dist.get_rank()
        if dev_count > 1:
          device = torch.device('cuda', rank)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[rank],
                                                          output_device=rank)
        dist.barrier()
      elif FLAGS.num_gpus > 1:
        # device ids is needed otherwise only 0 work..
        # model = torch.nn.DataParallel(model, device_ids=list(range(FLAGS.num_gpus)))
        model = torch.nn.DataParallel(model, device_ids=gpus)
        # model = DataParallelModel(model)
        model = model.to(device)  
      else:
        model = model.to(device)

      if hasattr(model, 'build'):
        model.build()

      gezi.set('model', model)
      ## https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
      ## not work yet
      # if dev_count > 1:
      #   loss_fn = DataParallelCriterion(loss_fn)
      gezi.set('loss_fn', loss_fn)
      gezi.set('device', device)
    else:
      model = gezi.get('model')
      loss_fn = gezi.get('loss_fn')
      device = gezi.get('device')

  input_ =  FLAGS.train_input 
  inputs = FLAGS.train_files or gezi.list_files(input_) 

  all_inputs = inputs

  batch_size = melt.batch_size()
  batch_size_ = batch_size if not FLAGS.distributed else FLAGS.batch_size
  eval_batch_size = melt.eval_batch_size()
  eval_batch_size_ = eval_batch_size if not FLAGS.distributed else FLAGS.eval_batch_size
  num_gpus = int(melt.batch_size() / FLAGS.batch_size) if not gezi.is_cpu_only() else 0

  num_epochs = FLAGS.num_epochs if FLAGS.num_epochs != 0 else 1024

  valid_inputs = None
  num_valid_examples = None
 
  inputs = FLAGS.train_files or inputs
  inputs += gezi.list_files(FLAGS.train_input2)
  # ic(FLAGS.train_input, FLAGS.shuffle, FLAGS.shuffle_files, FLAGS.input_files)
  if FLAGS.train_input and not FLAGS.shuffle and not FLAGS.shuffle_files and not FLAGS.input_files:
    try:
      inputs = sorted(inputs, key=lambda x: int(os.path.basename(x).split('.')[0]))
    except Exception as e:
      logging.debug(e)

  if valid_dataset is None:
    if (FLAGS.valid_input or FLAGS.valid_files) and not valid_inputs:
      valid_inputs = FLAGS.valid_files or gezi.list_files(FLAGS.valid_input)
      try:
        valid_inputs = sorted(valid_inputs, key=lambda x: int(os.path.basename(x).split('.')[0]))
      except Exception as e:
        logging.debug(e)

  if FLAGS.valid_exclude and valid_inputs:
    valid_inputs = [x for x in valid_inputs if not FLAGS.valid_exclude in x]

  if FLAGS.train_exclude and inputs:
    inputs_exclude = gezi.list_files(FLAGS.train_exclude)
    inputs = [x for x in inputs if x not in inputs_exclude]
    
  if valid_inputs and inputs and not FLAGS.allow_valid_train:
    inputs = [x for x in inputs if not x in valid_inputs]

  logging.debug('inputs:', len(inputs), inputs[:10])
  ic(len(inputs), inputs[:2])
  if valid_inputs:
    ic(len(valid_inputs), valid_inputs[:2])
  
  model.mode = 'test'
  keras_inited = False
  if not FLAGS.torch:
    if model is not None and hasattr(model, 'init_predict'):
      if not hasattr(model, 'inited_predict') or not model.inited_predict:
        model.init_predict()
        model.inited_predict = True
    try:
      # example = melt.first_input(inputs[0])
      ## 这里很重要! 因为subclass 需要这里输入input先build()一下 考虑使用class而不是function接口 对应动态输入 同时需要相同scope 和后面model.complile (husky/train.py)
      strategy = melt.distributed.get_strategy()
      # @tf.function
      def train_first(example):
        # example = example.copy()
        model(example)
      example = next(iter(dataset or Dataset('train').make_batch(batch_size, [inputs[0]])))[0]
      # if not gezi.get('tpu'):
      #   # 多gpu模式需要这里 model(example)不行 ValueError: Variable was not created in the distribution strategy scope of (<tensorflow.python.distribute.mirrored_strategy.MirroredStrategy object at 0x7fc6a56ce438>).
      #   # 需要强制在strategy下面 当然也可以最外层 with strategy.scope():
      #   strategy.run(train_first, args=(example,))
      # else:
      #   # tpu eager strategy.run不行 好像不支持eager模式 tf.function没有尝试 但是model(exmaple)是可以的
      #   model(example)

      with strategy.scope():
        # change to use train mode as train might has more weights then valid
        try:
          model(example, training=True)
          model(example, training=False)
        except Exception:
          model(example)
        
      try:
        model.init_gradient_acc()
      except Exception:
        pass

      if FLAGS.model2tb:
        melt.model2tb(model)

      # model.predict_on_batch(example)
      # print(example)
      # print(model.get_weights())
      print_fn = logging.info if FLAGS.round == 0 and FLAGS.work_mode in ('train', 'show') else logging.debug
      # TODO still can not detect output_shape... show multiple
      if FLAGS.round == 0:
        melt.print_model(model, print_fn=print_fn, depth=FLAGS.print_depth)
        if FLAGS.print_depth == 1 or FLAGS.print_depth < 0:
          melt.print_model(model, print_fn=logging.debug, depth=0)
      if not FLAGS.keras: # for keras will later use callback
        total_params = model.count_params()
        l2 = melt.get_l2_sum(model) / total_params
        logging.info('Model total training parameters is:', total_params, 'with initial l2:', l2)
      keras_inited = True
    except Exception:
      logging.warning(traceback.format_exc())
      keras_inited = False

  repeat_fn = None
  if dataset is None:
    dataset = Dataset('train', num_instances=FLAGS.num_train)
    assert len(inputs) > 0, input_

    kwargs = {}
    if FLAGS.torch and FLAGS.distributed:
      kwargs['world_size'] = dist.get_world_size()
      kwargs['rank'] = dist.get_rank()
    if FLAGS.parts and FLAGS.use_shard:
      kwargs['world_size'] = FLAGS.parts
      kwargs['rank'] = FLAGS.part

    train_dataset = dataset.make_batch(batch_size_, inputs, repeat=True, 
                                       simple_parse=FLAGS.simple_parse, 
                                       cache=FLAGS.cache or FLAGS.cache_train, 
                                       cache_file=FLAGS.cache_file,
                                       **kwargs)
    # num_examples = melt.get_num_records(inputs)
    num_examples = len(dataset)
    # if FLAGS.fold is not None:
    #   num_valid_examples = melt.get_num_records(valid_inputs)
      # num_examples = num_examples - num_valid_examples
    if FLAGS.num_train:
      num_examples = min(num_examples, FLAGS.num_train)
  else:
    # assert FLAGS.torch_only, 'only torch only currently support input dataset not Dataset class type, because we do not have len function there'
    train_dataset = dataset
    if FLAGS.torch_only:
      num_examples = FLAGS.num_train or len(train_dataset.dataset)
    else:
      num_examples = FLAGS.num_train or len(train_dataset)

  if valid_dataset is None:
    if (FLAGS.valid_input or FLAGS.valid_files) and not valid_inputs:
      valid_inputs = FLAGS.valid_files or gezi.list_files(FLAGS.valid_input)

  if FLAGS.valid_exclude and valid_inputs:
    valid_inputs = [x for x in valid_inputs if not FLAGS.valid_exclude in x]

  if valid_inputs:
    logging.debug('valid_inputs:', len(valid_inputs), valid_inputs[:10])
    ic(len(valid_inputs), valid_inputs[:2])

  # num_valid_examples = None
  if valid_dataset is not None:
    # mainly for torch now, actutally len(valid_dataset) == -(-num_examples // batch_size) == num_steps_per_epoch
    if FLAGS.torch_only:
      num_full_valid_examples = len(valid_dataset.dataset) if not hasattr(valid_dataset.dataset, 'dataset') else len(valid_dataset.dataset.dataset)
      num_valid_examples = FLAGS.num_valid or num_full_valid_examples
    else:
      try:
        num_full_valid_examples = len(valid_dataset)
      except Exception:
        num_full_valid_examples = FLAGS.num_full_valid or FLAGS.num_valid
      num_valid_examples = FLAGS.num_valid or num_full_valid_examples
    # or num_valid_steps_per_epoch = len(valid_dataset)
    try:
      gezi.set('num_full_valid_examples', num_full_valid_examples)
      gezi.set('num_full_valid_steps_per_epoch', -(-num_full_valid_examples // eval_batch_size))
    except Exception:
      pass
    num_valid_steps_per_epoch = -(-num_valid_examples // eval_batch_size) if num_valid_examples else None   
    valid_dataset2_iter = itertools.cycle(valid_dataset2)
  else:
    if valid_inputs and Dataset is not None:
      kwargs = {}
      if FLAGS.torch and FLAGS.distributed:
        kwargs['world_size'] = dist.get_world_size()
        kwargs['rank'] = dist.get_rank()
      if FLAGS.parts and FLAGS.use_shard: # 如果use_shard=False 不走dataset.shard 手动切分
        kwargs['world_size'] = FLAGS.parts
        kwargs['rank'] = FLAGS.part

      valid_dataset_ = Dataset('valid', num_instances=FLAGS.num_valid)
      cache_valid = not FLAGS.num_valid and FLAGS.cache_valid # 如果设定num_valid数据是非repeat的状态 第一次没有遍历到结束 无法cache
      
      # 在tpu环境下 可能需要设置drop remainder为True保持固定batch_size shape 这样设置repeat=True不丢失最后一个batch数据(特别是infer是不能丢的)，但是会重复最后一个batch
      # 少量数据 可以自己再后处理 [:num_valid] 或者不处理误差也很小 evaluate默认是[:num_valid] 自己手写的loop需要注意一下
      repeat = True if gezi.get('tpu') or FLAGS.drop_remainder else False  

      # this is actually eval_dataset
      valid_dataset = valid_dataset_.make_batch(eval_batch_size_, valid_inputs, subset='valid', repeat=repeat, hvd_shard=True, 
                                                cache=cache_valid, **kwargs)
      # valid_dataset = Dataset('valid').make_batch(eval_batch_size_, valid_inputs, repeat=False)
      # this is valid dataset valid nbatch in parallel with traning process
      valid_dataset2 = valid_dataset_.make_batch(eval_batch_size_ if not FLAGS.keras else batch_size_, 
                                                 valid_inputs, subset='valid', repeat=True, initializable=False, hvd_shard=False,
                                                 cache=FLAGS.cache_valid, **kwargs)

      eval_info_dataset_ = Dataset('valid', is_info=True, eval_keys=eval_keys)
      eval_info_dataset = eval_info_dataset_.make_batch(eval_batch_size_, valid_inputs, subset='valid', repeat=repeat, hvd_shard=True, 
                                                        cache=cache_valid, **kwargs)     
      try:
        valid_dataset2_iter = iter(valid_dataset2)
      except Exception:
        pass
    else:
      valid_dataset = None
      valid_dataset2 = None

  if num_examples:
    num_steps_per_epoch = -(-num_examples // batch_size)
  else:
    num_steps_per_epoch = None
  logging.info('num_train_examples:', num_examples, 'num_steps_per_epoch:', num_steps_per_epoch)

  if num_valid_examples is None:
    if (FLAGS.valid_input or FLAGS.valid_files):
      try:
        num_valid_examples = len(valid_dataset_)
        gezi.set('num_full_valid_examples', num_valid_examples)
        gezi.set('num_full_valid_steps_per_epoch', -(-num_valid_examples // eval_batch_size))
      except Exception:
        pass

  if FLAGS.num_valid and num_valid_examples is not None:
    num_valid_examples = min(num_valid_examples, FLAGS.num_valid)

  num_valid_steps_per_epoch = -(-num_valid_examples // eval_batch_size) if num_valid_examples else None   
  if num_valid_examples:
    logging.info('num_valid_examples:', num_valid_examples, 'num_valid_steps_per_epoch:', num_valid_steps_per_epoch)

  if test_dataset is None:
    if FLAGS.test_input or FLAGS.test_files:
      test_inputs = FLAGS.test_files or gezi.list_files(FLAGS.test_input)
      try:
        test_inputs = sorted(test_inputs, key=lambda x: int(os.path.basename(x).split('.')[0]))
      except Exception as e:
        logging.debug(e)
      logging.debug('test_inputs:', len(test_inputs), test_inputs[:10])
      ic(len(test_inputs), test_inputs[:2])
    else:
      test_inputs = None
  
  num_test_examples = None
  if test_dataset is not None:
    test_inputs = None
    if FLAGS.torch_only:
      num_test_examples = len(test_dataset.dataset)
    else:
      num_test_examples = FLAGS.num_test or len(test_dataset)
  else:
    if test_inputs and Dataset is not None:
      if FLAGS.torch and FLAGS.distributed:
        kwargs['world_size'] = dist.get_world_size()
        kwargs['rank'] = dist.get_rank()
      if FLAGS.parts and FLAGS.use_shard:
        kwargs['world_size'] = FLAGS.parts
        kwargs['rank'] = FLAGS.part
      test_dataset_ = Dataset('test', num_instances=FLAGS.num_test)
      repeat = True if FLAGS.drop_remainder else False
      test_dataset = test_dataset_.make_batch(eval_batch_size_, test_inputs, subset='test', repeat=repeat,
                                        cache=FLAGS.cache_test, cache_file=FLAGS.cache_file, **kwargs)
      test_info_dataset_ = Dataset('test', is_info=True)                                        
      test_info_dataset = test_info_dataset_.make_batch(eval_batch_size_, test_inputs, subset='test', repeat=repeat,
                                                   cache=FLAGS.cache_test, cache_file=FLAGS.cache_file, **kwargs)      
      num_test_examples = FLAGS.num_test or len(test_dataset_)
    else:
      test_dataset = None
  num_test_steps_per_epoch = -(-num_test_examples // eval_batch_size) if num_test_examples else None
  if num_test_examples:
    logging.info('num_test_examples:', num_test_examples, 'num_test_steps_per_epoch:', num_test_steps_per_epoch)
  
  if rank == 0:
    writer = gezi.DfWriter(FLAGS.log_dir, filename='metrics.csv')
    summary = melt.get_summary_writer()
    if FLAGS.round == 0:
      if FLAGS.train_valid_summary:
        train_logger = SummaryWriter(os.path.join(FLAGS.log_dir, 'train'))
        valid_logger = SummaryWriter(os.path.join(FLAGS.log_dir, 'valid'))
      else:
        train_logger, valid_logger = None, None
      gezi.set_global('summary_related', [train_logger, valid_logger])
    else:
      train_logger, valid_logger = gezi.get_global('summary_related')

  # if is_eager:
  #   global_step = tf.compat.v1.train.get_or_create_global_step()
  # else:
  global_step = melt.GlobalStep(0)
  
  if not FLAGS.torch:
    learning_rate = tf.Variable(FLAGS.learning_rate, name="learning_rate")
  else:
    learning_rate = melt.LearningRate(FLAGS.learning_rate)
  
  melt.set_global('learning_rate', learning_rate)

  learning_rate_weight = melt.get_global('learning_rate_weight')
  try:
    learning_rate_weights = melt.get_global('learning_rate_weights')
  except Exception:
    learning_rate_weights = None

  info = {
    'Dataset': Dataset,
    'dataset': train_dataset, 
    'eval_dataset': valid_dataset,
    'valid_dataset': valid_dataset2,
    'test_dataset': test_dataset,
    'inputs': inputs,
    'valid_inputs': valid_inputs,
    'test_inputs': test_inputs,
    'eval_info_dataset': eval_info_dataset,
    'test_info_dataset': test_info_dataset,
    'num_examples': num_examples,
    'num_valid_examples': num_valid_examples,
    'num_test_examples': num_test_examples,
    'num_steps_per_epoch': num_steps_per_epoch,
    'num_valid_steps_per_epoch': num_valid_steps_per_epoch,
    'num_test_steps_per_epoch': num_test_steps_per_epoch
  }
  gezi.set('info', info)
  if return_info:
    return info 
   
  # ckpt dir save models one per epoch
  ckpt_dir = os.path.join(FLAGS.model_dir, 'ckpt')
  os.system('mkdir -p %s' % ckpt_dir)
  checkpoint_prefix = os.path.join(FLAGS.model_dir, 'ckpt')
  checkpoint_prefix_epoch = os.path.join(ckpt_dir, 'ckpt')

  latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
  if latest_checkpoint:
    logging.info('Latest checkpoint:', latest_checkpoint)
  
  if os.path.exists(FLAGS.model_dir + '.index'):
    latest_checkpoint = FLAGS.model_dir  

  if FLAGS.model_path:
    latest_checkpoint = FLAGS.model_path
  if FLAGS.work_mode == 'train':
    FLAGS.del_model_path = latest_checkpoint

  if not FLAGS.torch:
    assert FLAGS.round == 0, 'TODO eager tf(tf2) not support FLAGS.rounds !=  1 yet'
    if not optimizer and FLAGS.work_mode in ('train', 'show'):
      if FLAGS.optimizers:
        # TODO not support multiple optimziers right now
        logging.warning('Not support multiple optimizers for eager right now, just use the first optimizer')
        FLAGS.optimizer = FLAGS.optimizers.split(',')[0]
      optimizer_name = FLAGS.optimizer.split('-')[-1]
      logging.info('Optimizer name:', optimizer_name)
      # TODO
      # optimizer = melt.get_optimizer_byname(optimizer_name, learning_rate)
      optimizer, _ = create_optimizer(init_lr=FLAGS.lr,
                                   num_train_steps=int(num_steps_per_epoch * num_epochs),
                                   num_warmup_steps=int(num_steps_per_epoch * num_epochs * 0.1))
      ic(optimizer)

    # TODO...
    kwargs = {
      # 'learning_rate': learning_rate,
      # 'learning_rate_weight': learning_rate_weight,
      'model': model,
      # 'global_step': global_step
    }
    
    if optimizer is not None:
      kwargs['optimizer'] = optimizer
    # if learning_rate_weights is not None:
    #   kwargs['learning_rate_weights'] = learning_rate_weights
   
    checkpoint = tf.train.Checkpoint(**kwargs)
    status = checkpoint.restore(latest_checkpoint)
    logging.info(f'Loading tf eager model [{latest_checkpoint}], status {status}')
    manager = tf.train.CheckpointManager(checkpoint, directory=FLAGS.model_dir, 
                                                      max_to_keep=FLAGS.max_models_keep,
                                                      checkpoint_name='ckpt')

    epoch_checkpoint = tf.train.Checkpoint(**kwargs)
    epoch_manager = tf.train.CheckpointManager(epoch_checkpoint, directory=ckpt_dir, 
                                              max_to_keep=FLAGS.max_models_keep, checkpoint_name='ckpt')

    start_epoch = int(global_step.numpy() / num_steps_per_epoch)
  else:
    optimizer, scheduler = get_torch_optimizer(optimizer, model, num_steps_per_epoch, lr_params)

    start_epoch = 0  

    # for loop_train
    if FLAGS.train_loop:
      start_hour = None
      total_rounds = 0
    
    pretrained = None 
    if FLAGS.pretrained:
      if os.path.isdir(FLAGS.pretrained):
        pretrained = FLAGS.pretrained
      else:
        pretrained = f'{os.path.dirname(FLAGS.model_dir)}/{FLAGS.pretrained}'
    latest_path = melt.latest_checkpoint(pretrained or FLAGS.model_dir)
    if latest_path and os.path.exists(latest_path):
      latest_checkpoint = latest_path
    else:
      latest_checkpoint = None

    if FLAGS.model_path:
      latest_checkpoint = FLAGS.model_path
    if FLAGS.work_mode == 'train':
      FLAGS.del_model_path = latest_checkpoint

    if FLAGS.round == 0:
      #--------------------try restore checkpoint
      if latest_checkpoint:
        # NOTICE must only load 0! if all load now will OOM...  TODO FIXME
        if rank == 0:
          logging.info(f'Loading torch model from: [{latest_path}]')
          checkpoint = load_torch_model(latest_path, model=model, optimizer=optimizer if FLAGS.restore_optimizer else None,
                                        map_location=device, includes=FLAGS.restore_include.split(','),
                                        excludes=FLAGS.restore_exclude.split(','))

        if not FLAGS.torch_restart:
          if rank == 0:
            start_epoch = checkpoint['epoch']
            step = checkpoint['step']
            if FLAGS.train_loop:
              start_hour = checkpoint.get('start_hour', None)
              total_rounds = checkpoint.get('round', 0)
          else:
            start_epoch = 0
            step = 0
        
          global_step.assign(step)

  if FLAGS.train_loop:
    global_step.assign(0)
    start_epoch = 0

  if FLAGS.torch:
    if optimizer is not None:
      if hasattr(optimizer, 'set_step'):
        optimizer.set_step(global_step.numpy())
      else:
        optimizer._step = global_step.numpy()
      logging.debug('optimizer:', optimizer)

      learning_rate.assign(optimizer.param_groups[0]['lr'])
      logging.debug('learning rate got from optimizer as', learning_rate.numpy())

  # not used much for we do this in optimzier wrapper 
  learning_rate.assign(learning_rate * FLAGS.learning_rate_start_factor)
  if learning_rate_weights is not None:
    learning_rate_weights.assign(learning_rate_weights * FLAGS.learning_rate_start_factor)

  will_valid = valid_dataset is not None and not FLAGS.work_mode == 'test' and not 'SHOW' in os.environ and not 'QUICK' in os.environ
  
  if global_step.numpy() == 0:
    will_valid = False

  if gezi.get_env('EVFIRST') == '1':
    will_valid = True
  
  if gezi.get_env('EVFIRST') == '0':
    will_valid = False

  will_valid = False

  if FLAGS.work_mode == 'valid' or gezi.get_env('METRIC') == '1':
    will_valid = True

  model.mode = 'valid'
  if hasattr(model, 'eval'):
    model.eval()
  names = None 
  if not valid_names and infer_names:
    valid_names = [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:]

  model_path = latest_checkpoint
  names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None

  ofile = None
  metric_eval_fn = lambda is_last=False, write=write_valid: \
                                evaluate(model, valid_dataset, eval_fn, model_path,
                                         names, valid_write_fn, write_streaming,
                                         num_steps=num_valid_steps_per_epoch, 
                                         num_examples=num_valid_examples,
                                         suffix=valid_suffix, sep=sep, keys=eval_keys, 
                                         out_hook=out_hook, out_keys=out_keys, str_keys=str_keys,
                                         write=write_valid, write_valid_only=FLAGS.write_valid_only, ofile=ofile,
                                         is_last=is_last, show=True)

  if not FLAGS.metric_eval:
    metric_eval_fn = None

  # TODO eager 特别是pytorch的流程需要重新梳理写一下 直接用lighting ?
  if will_valid:
    args = inspect.getfullargspec(eval_fn).args
    if eval_fn and not 'dataset' in args:
      _try_eval(FLAGS.model_dir, FLAGS.log_dir, metric_eval_fn)
    else:
      eval_fn = eval_fn or evaluate_fn
      if eval_fn is not None:
        kwargs = {}   
        if 'show' in args:
          kwargs['show'] = True
        if 'info' in args:
          kwargs['info'] = x
        if 'x' in args:
          kwargs['x'] = x
        if 'model' in args:
          kwargs['model'] = model
        if 'other' in args:
          kwargs['other'] = x
        if 'eval_step' in args:
          kwargs['eval_step'] = melt.get_eval_step()
        if 'step' in args:
          kwargs['step'] = global_step.numpy()
        if 'is_last' in args:
          kwargs['is_last'] = global_step.numpy() == int(num_steps_per_epoch * FLAGS.num_epochs)
        if 'steps' in args:
          kwargs['steps'] = num_valid_steps_per_epoch
        if 'loss_fn' in args:
          kwargs['loss_fn'] = loss_fn
        if 'ofile' in args:
          kwargs['ofile'] = None
        if 'outdir' in args:
          kwargs['outdir'] = FLAGS.model_dir
        if 'num_examples' in args:
          kwargs['num_examples'] = num_valid_examples
        if 'return_dict' in args:
          kwargs['return_dict'] = True
        if 'desc' in args:
          kwargs['desc'] = 'eval'
        if 'is_last' in args:
          kwargs['is_last'] = True
          
        if FLAGS.parts and not FLAGS.use_shard:
          from husky.callbacks.evaluate import _prepare_eval_part
          valid_dataset, steps, num_valid_examples = _prepare_eval_part(FLAGS.part, FLAGS.parts)
          if 'steps' in kwargs:
            kwargs['steps'] = steps
          if 'num_examples' in kwargs:
            kwargs['num_examples'] = num_valid_examples
          if FLAGS.parts:
            if 'desc' in kwargs:
              kwargs['desc'] = f'eval: {FLAGS.part}/{FLAGS.parts}'
        
        results = eval_fn(valid_dataset, **kwargs)

    if FLAGS.work_mode == 'valid':
      return
      # exit(0)
    
  if 'test' in FLAGS.work_mode or gezi.get_env('TEST') == '1' or gezi.get_env('INFER') == '1':
    logging.info('--------test/inference')
    if test_dataset:
      model.mode = 'test'
      if hasattr(model, 'eval'):
        model.eval()
      if inference_fn is None:
        # model_path = FLAGS.model_dir + '.pyt' if not latest_checkpoint else latest_checkpoint
        # logging.info('model_path', model_path)
        #assert latest_checkpoint
        inference(model, test_dataset, latest_checkpoint, 
                  infer_names, infer_debug_names, infer_write_fn, write_streaming,
                  num_test_steps_per_epoch, num_test_examples, suffix=infer_suffix)
      else:
        inference_fn(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), num_test_steps_per_epoch)
    exit(0)
  
  if 'SHOW' in os.environ:
    num_epochs = start_epoch + 1
  
  ## tfe.metrics.Mean will auto add scalar to tensorboard, but for consistency, not add to tb (result write_summary=False)
  Mean =  tf.compat.v2.metrics.Mean if not FLAGS.torch else PytMean
  num_insts = 0

  #-------------------------start training
  model.mode = 'train'
  if hasattr(model, 'train'):
    model.train()
  
  decay_steps, decay_start_step = prepare_decay_learning_rate(num_steps_per_epoch)

  # for eager right now is repeat mode, so not need for epoch, but we can stop at inter loop last check global step and exit
  for callback in callbacks:
    if hasattr(callback, 'set_model'):
      callback.set_model(model)

  # ----------------------------------------main loop here
  # eager mode for safe one more epoch loop but actually stop due to global step 
  # profile_context = lambda: gezi.DummyContextManager() if not FLAGS.torch else lambda: torch.autograd.profiler.profile(FLAGS.enable_profiling, use_gpu)
  # # with profile_context() as prof:
  # print(profile_context)
  ## TODO why one more epoch ? maybe for 1.2 epoch like this..
  if not FLAGS.train_loop and (num_epochs > int(num_epochs)):
    end_epoch = 0 + int(num_epochs) + 1
  else:
    end_epoch = 0 + int(num_epochs)
  
  batch_size = melt.batch_size()
  num_gpus = int(melt.batch_size() / FLAGS.batch_size)

  total_step = melt.get_total_step()
  logging.sprint(total_step)

  timer = gezi.Timer()
  loss_avg = Mean()
    
  timer_, FLAGS.total_time, FLAGS.train_time, FLAGS.valid_time = gezi.Timer(reset=False), None, None, None
  strategy = melt.distributed.get_strategy()
  # with torch.autograd.profiler.profile(enabled=FLAGS.enable_profiling, use_cuda=use_gpu) as prof:
  # worker = gezi.AsyncWorker(train_dataset, num_steps_per_epoch * num_epochs, prefetch=10)
  # worker.start()
  total_steps = int(num_epochs * num_steps_per_epoch)
  if FLAGS.torch and FLAGS.swa_ratio > 0:
    swa_model = AveragedModel(model)
    gezi.set('swa_model', swa_model)
    swa_start = total_steps * (1 - FLAGS.swa_ratio)
    # swa_scheduler = SWALR(optimizer, swa_lr=FLAGS.learning_rate * FLAGS.swa_lr_ratio)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
  else:
    swa_model = None
    swa_start = None
  if FLAGS.fold is None:  
    pbar = tqdm(total=total_steps, desc=f'[{FLAGS.mn}] Epochs:{num_epochs}')
  else:
    pbar = tqdm(total=total_steps, desc=f'[{FLAGS.mn}_{FLAGS.fold}] Epochs:{num_epochs}')
  # for epoch in tqdm(range(0, end_epoch), desc='Training', ascii=False):
  for callback in callbacks:
    if hasattr(callback, 'on_train_start'):
      callback.on_train_start()
  for epoch in range(0, end_epoch):
    logging.debug('------------------------epoch:', epoch)
    if epoch < start_epoch:
      pbar.update(num_steps_per_epoch)
      continue
    
    if gen_dataset_fn is not None:
      train_dataset = gen_dataset_fn(int(epoch))
    
    for callback in callbacks:
      if hasattr(callback, 'on_epoch_start'):
        kwargs = {}
        if 'lr' in inspect.getargspec(callback.on_epoch_start).args:
          kwargs['lr'] = learning_rate
        callback.on_epoch_start(epoch, **kwargs)

    # FLAGS.torch only will not use eager, FLAGS.torch still use eager tf reading
    if FLAGS.torch_only:
      if train_dataset.sampler is not None and hasattr(train_dataset.sampler, 'set_epoch'):
        # if not set each epoch shuffle same seed..
        train_dataset.sampler.set_epoch(epoch)
      train_dataset.dataset.epoch = gezi.epoch()
        
    train_hour = FLAGS.train_hour if FLAGS.loop_train else None
    desc = 'Epoch:%2d/%d' % (epoch + 1, int(num_epochs)) if not train_hour else '%s-%d/%d Epoch:%2d/%d' % (train_hour, FLAGS.round + 1, FLAGS.num_rounds, epoch + 1, int(num_epochs))
    # TODO https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy
    # train_dataset = strategy.experimental_distribute_dataset(train_dataset) 
    # t = tqdm(enumerate(train_dataset), total=num_steps_per_epoch, desc=desc, ascii=False, leave=False)
    update_per_second = 10
    update_interval = 1 / update_per_second
    last_update_time = time.time()
    # for i, (x, y) in t:
    for i, (x, y) in enumerate(train_dataset):
    # t = tqdm(range(num_steps_per_epoch), total=num_steps_per_epoch, desc=desc, ascii=False)
    # for i in t:
    #   batch = worker.get()
    #   x, y = batch
      # if i == 0:
      #   print(x['id'][0])
      ## TODO use update time check and update only exceed update interval
      # if i % 10 == 0:
      now = time.time()
      time_diff = now - last_update_time
      postfix = {}
      if gezi.get('loss'):
        postfix['loss'] = gezi.get('loss')
      if gezi.get('valid_loss'):
        postfix['val_loss'] = gezi.get('valid_loss')
      # postfix.update({k: f'{v:.4f}' for k,v in gezi.get('scalars', {}).items() if 'loss' in k and not 'val_' in k})
      postfix.update({k: f'{v:.4f}' for k,v in gezi.get('scalars', {}).items() if k.startswith('loss') and not 'val_' in k})
      if postfix and time_diff >= update_interval:
        # t.set_postfix(postfix)  # TODO check if this will cause slower...
        pbar.set_postfix(postfix)
      
      # t.set_postfix(loss=gezi.get('loss', 'none'), val_loss=gezi.get('valid_loss', 'none'))
      # TODO eager profiler
      if FLAGS.profile_interval_steps and i % FLAGS.profile_interval_steps == 0:
        profiler.start()
      # continue  # to test if only loop
      if FLAGS.torch:
        x, y = to_torch(x, y) # to test if only loop + totorch
        # continue
        if FLAGS.torch_lr:
          try:
            if hasattr(optimizer, 'rate'):
              learning_rate.assign(optimizer.rate())
            else:
              learning_rate.assign(scheduler.get_last_lr()[0])
          except Exception:
            learning_rate = 0

      if loss_fn is not None:
        def loss_fn_(x, y):
          y_ = model(x)
          kwargs = {}
          loss_args = inspect.getargspec(loss_fn).args
          if 'x' in loss_args:
            kwargs['x'] = x 
          if 'model' in loss_args:
            # kwargs['model'] = model
            # TODO need model.module for data parallel ? model.logit not found ..
            kwargs['model'] = model if not hasattr(model, 'module') else model.module
          if 'weights' in loss_args:
            weights_ = x[weights] if isinstance(weights, str) else weights
            kwargs['weights'] = weights_
          if 'weight' in loss_args:
            weights_ = x[weights] if isinstance(weights, str) else weights
            kwargs['weight'] = weights_
          if 'training' in loss_args:
            kwargs['training'] = model.training
          if 'step' in loss_args:
            kwargs['step'] = global_step.numpy()
          if 'epoch' in loss_args:
            kwargs['epoch'] = melt.epoch()
          
          if FLAGS.rdrop_rate > 0:
            y2_ = model(x)

          if FLAGS.torch:
            x['step'] = global_step.numpy()
            x['epoch'] = melt.epoch()
            # ic(y_.shape, y.shape)
            if FLAGS.rdrop_rate > 0:
              loss = 0.5 * (loss_fn(y_, y, **kwargs) + loss_fn(y2_, y, **kwargs))
              rdrop_loss_fn = gezi.get('rdrop_loss_fn')
              rloss = rdrop_loss_fn(y_, y2_, x)
              loss += rloss * FLAGS.rdrop_rate
              return loss
            return loss_fn(y_, y, **kwargs)
          else:
            return loss_fn(y, y_, **kwargs)
      else:
        loss_fn_ = None

      for callback in callbacks:
        if hasattr(callback, 'on_batch_begin'):
          kwargs = {}
          if 'lr' in inspect.getargspec(callback.on_batch_begin).args:
            kwargs['lr'] = learning_rate
          callback.on_batch_begin(global_step.numpy(), **kwargs)
      
      K.set_learning_phase(1)
      model.mode = 'train'
      if hasattr(model, 'train'):
        model.train()

      # add 1 before train, mainly for usage gezi.is_valid_step() in model.py or loss.py during training
      gezi.set_global_step(global_step.numpy() + 1)
      # train steps loop inteval summaries
      if not FLAGS.torch:
        hvd_ = None 
        ## tf function 慢的原因是因为有dynamic输入每次输入type或者shape变化都会重新构图 当然可以按照FLAGS.static_input来区分是否加tf.fucntion 但是这里保留eager这个流程主要为了方便debug直接print 
        ## 而不需要使用tf.print
        @tf.function 
        def _train_step(x, y):
          # ic(x)
          if not FLAGS.loop_only:
            loss, grads = melt.eager.grad(model, x, y, loss_fn_, hvd=hvd_)
            # if FLAGS.clip_gradients:
            #   grads, _ = tf.clip_by_global_norm(grads, FLAGS.clip_gradients)
            grads_and_vars = zip(grads, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars)
          else:
            loss = 0.
          return loss

        if FLAGS.static_input:
          @tf.function
          def train_step(x, y):
            return _train_step(x, y)
        else:
          # TODO 解析设置None的情况,支持dyanmic长度 并且不重构图
          if i == 0:
            logging.warning('not using @tf.function currently for FLAGS.static_input=False, custom loop might be very slow, try to use FLAGS.static_input=True if no varlen exists')
          def train_step(x, y):
            return _train_step(x, y)            
            
        ## TODO 
        if melt.distributed.is_dummy_strategy(strategy): #single gpu
          loss = train_step(x, y)
        else:
          # slow...
          loss = strategy.run(train_step, args=(x, y))
          loss = strategy.reduce("MEAN", loss, axis=None)
          
        gezi.set('loss', '%.4f' % loss.numpy())
      else:
        loss = loss_fn_(x, y)
        # worker.prefetch()
        loss = loss / FLAGS.acc_steps
        # ic(i)
        loss.backward()
        
        # TODO 这里的optimizer step对应 acc_steps 会ok吗
        if ((global_step.numpy() + 1) % FLAGS.acc_steps == 0) or (global_step.numpy() + 1 == total_steps):
          if FLAGS.clip_gradients > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            FLAGS.clip_gradients)
          optimizer.step()
          optimizer.zero_grad()
          
        if scheduler is not None:
          if swa_start and global_step.numpy() > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
          else:
            scheduler.step()
            
        gezi.set('loss', '%.4f' % loss.item())
      K.set_learning_phase(0)
      model.mode = 'valid'
      if hasattr(model, 'eval'):
        model.eval()

      global_step.assign_add(1)
      gezi.set('epoch', global_step.numpy() / num_steps_per_epoch)
      
      if FLAGS.profile_interval_steps and i % FLAGS.profile_interval_steps == 0:
        profiler_result = profiler.stop()
        profiler.save(os.path.join(FLAGS.model_dir, 'profiler_result'), profiler_result)

      for callback in callbacks:
        if hasattr(callback, 'on_batch_end'):
          kwargs = {}
          if 'lr' in inspect.getargspec(callback.on_batch_end).args:
            kwargs['lr'] = learning_rate
          callback.on_batch_end(global_step.numpy(), **kwargs)

      if i == 0 and epoch == start_epoch:
        if FLAGS.round == 0:
          if not FLAGS.torch:
            # logging.info('-----------------------------keras_inited:', keras_inited)
            if not keras_inited:
              #------------------eager show info keras info
              if model is not None and hasattr(model, 'init_predict'):
                if not hasattr(model, 'inited_predict') or not model.inited_predict:
                  model.init_predict()
                  model.inited_predict = True
              print_fn = logging.info if FLAGS.round == 0 and FLAGS.work_mode in ('train', 'show') else logging.debug
              # # TODO still can not detect output_shape... show multiple
              melt.print_model(model, print_fn=print_fn, depth=FLAGS.print_depth)
              try:
                total_params = model.count_params()
                l2 = melt.get_l2_sum(model) / total_params
                logging.info('Model total training parameters is:', total_params, 'with initial l2:', l2)
              except Exception as e:
                ic(e)
              if FLAGS.plot_graph:
                logging.info(f'---------ploting to {FLAGS.model_dir}/graph.png')
                plot_model(model, to_file=f'{FLAGS.model_dir}/graph.png', show_shapes=True, show_layer_names=True)
                exit(0)
          
            if 'SHOW' in os.environ or FLAGS.work_mode == 'show' or FLAGS.plot_graph:
              exit(0)

          else:
            if FLAGS.plot_graph:
              from torchviz import make_dot
              dot_file = f'{FLAGS.model_dir}/graph'
              with gezi.Timer(f'Dot write graph to {dot_file}', print_fn=logging.info) as t:
                dot = make_dot(loss, params=dict(model.named_parameters()))
                dot.render(f'{FLAGS.model_dir}/graph') # write .pdf file
              with gezi.Timer(f'Write summary of graph to tensorboard', print_fn=logging.info) as t:
                summary.graph(model.cpu(), (filter_nontorch(x, cpu=True),))
              exit(0)
        
        if FLAGS.work_mode == 'train' and FLAGS.metric_eval and FLAGS.monitor_l2:
          total_params = get_total_params(model)
          l2 = get_l2_sum(model) / total_params
          logging.debug('Model total training parameters is:', total_params, 'with initial l2:', l2)
          # gezi.set_global('total_params', total_params)
          # gezi.set_global('l2', l2)
          FLAGS.l2_ = l2
          FLAGS.params_ = total_params
          
      loss_avg(loss)

      # torch 在optimzier的外围封装类处理了
      if not FLAGS.torch:
        learning_rate.assign(adjust_learning_rate(FLAGS.learning_rate, global_step.numpy(), num_steps_per_epoch, FLAGS.min_learning_rate))
    
      # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
      if FLAGS.torch:
        del loss

      # eval_batch_size = list(x.values())[0].shape[FLAGS.batch_size_dim] if type(x) == type({}) else x.shape[FLAGS.batch_size_dim]
      eval_batch_size = FLAGS.eval_batch_size
      num_insts += batch_size
      # Notice here step == 1 is equal to tf graph train_once.py step == 0
      if global_step.numpy() % FLAGS.interval_steps == 0 \
        or global_step.numpy() == 1 or global_step.numpy() == 100 or global_step.numpy() == 200 \
          or global_step.numpy() == num_steps_per_epoch:
        #checkpoint.save(checkpoint_prefix)
        elapsed = timer.elapsed()
        elapsed_steps = global_step.numpy() - timer.step
        timer.step = global_step.numpy()
        steps_per_second = elapsed_steps / elapsed 
        instances_per_second = num_insts / elapsed 
        # instances_per_second = interval_steps * batch_size / elapsed
        num_insts = 0
        
        if num_steps_per_epoch is None:
          hours_per_epoch = None
          epoch_time_info = ''
        else:
          hours_per_epoch = num_steps_per_epoch / elapsed_steps * elapsed / 3600
          mintues_per_epoch = hours_per_epoch * 60
          epoch_time_info = '1epoch:[{:.1f}h]'.format(hours_per_epoch) if hours_per_epoch > 1 else  '1epoch:[{:.1f}m]'.format(mintues_per_epoch)

        args = ['epoch:%.2f/%d' % ((global_step.numpy() / num_steps_per_epoch), num_epochs), 
                'step:%5d' % global_step.numpy(), 
                'elap:[%.2f]' % elapsed,
                'batch:[%d]' % batch_size,
                'gpus:[%d]' % num_gpus, 
                'steps/s:[%.1f]' % steps_per_second,
                'insts/s:[%s]' % np.format_float_scientific(instances_per_second, precision=1, trim='0'),
                '%s' % epoch_time_info,
                'lr:[%s]' % np.format_float_scientific(learning_rate.numpy() if hasattr(learning_rate, 'numpy') else learning_rate, precision=1, trim='0'),
                'train:[%.4f]' % loss_avg.result().numpy()]
        gezi.set('loss', '%.4f' % loss_avg.result().numpy())
        if FLAGS.train_hour:
          args = [f'train_hour:{FLAGS.train_hour}', *args]
        summaries_train , summaries_valid = None, None
        summaries_train = gezi.get_global('summaries/scalar', None)
        if summaries_train is not None:
          summaries_train = summaries_train.copy()
        if valid_dataset2:
          ## NOTICE will always the first batch ... as below
          #x, y = next(iter(valid_dataset2))
          x, y = next(valid_dataset2_iter)
          if FLAGS.torch:
            x, y = to_torch(x, y)
          model.mode = 'valid'
          if hasattr(model, 'eval'):  
            model.eval()
          if loss_fn_ is not None:
            with torch.no_grad():
              valid_loss = loss_fn_(x, y)
          else:
            valid_loss = 0.
            if model.losses:
              _ = model(x)
              valid_loss += sum(model.losses)
          valid_loss = valid_loss.numpy() if not FLAGS.torch else valid_loss.item()
          model.mode = 'train'
          if hasattr(model, 'train'):
            model.train()
          summaries_valid = gezi.get_global('summaries/scalar', None)
          if summaries_valid is not None:
            summaries_valid = summaries_valid.copy()
          args = [*args, 'valid:[%.4f]' % valid_loss]
          gezi.set('valid_loss', '%.4f' % valid_loss)
          logging.info2(*args)
          if rank == 0:
            if global_step.numpy() % FLAGS.valid_interval_steps == 0:
              step = global_step.numpy() + total_step # total_step is always 0 if not using FLAGS.train_loop else records total step(of all train rounds now)
              if FLAGS.write_summary:
                if valid_logger:
                  valid_logger.scalar('loss', valid_loss, step)
              if wandb_run:
                wandb.log({'History/val_loss': valid_loss, 'step': step})
              if FLAGS.write_summary:
                summary.log({'History/val_loss': valid_loss, 'step': step})
        else:
          logging.info2(*args)    

        if rank == 0:
          if global_step.numpy() % FLAGS.valid_interval_steps == 0:
            step = global_step.numpy() + total_step
            if FLAGS.write_summary:
 
              if FLAGS.train_hour:
                try:
                  summary.scalar('other/train_time', int(FLAGS.train_hour), step, 0)
                except Exception:
                  pass

            m = {
              'History/train_loss': loss_avg.result().numpy(),
              'History/learning_rate': learning_rate.numpy() if hasattr(learning_rate, 'numpy') else learning_rate,
              'Perf/hours_per_epoch': hours_per_epoch,
              'Perf/steps_per_second': steps_per_second,
              'Perf/insts_per_second': instances_per_second,
              'History/step': step,
              'Perf/step': step,
              'step': step
            }
            m.update(
              gezi.dict_prefix(gezi.get('scalars', {}), 'History/')
            )
            if wandb_run:
              wandb.log(m)
            if FLAGS.write_summary:
              summary.log(m)
            def _add(summaries, tag):
              if summaries:
                for key, val in summaries.items():
                  if '{tag}' in key:
                    key = key.replace('{tag}', 'train')
                  elif tag not in key: # loss -> train/loss loss/click - > loss/train/click
                    l = key.split('/', 1)
                    l.insert(-1, tag)
                    key = '/'.join(l)
                  summary.scalar(key, val, step)
            _add(summaries_train, 'train')
            _add(summaries_valid, 'valid')

            if train_logger:
              train_logger.scalar('loss', loss_avg.result().numpy(), step)
        loss_avg = Mean()

      # metric_eval_interval_steps not used much just use nvs or vie
      if valid_dataset and FLAGS.metric_eval_interval_steps and global_step.numpy() and global_step.numpy() % FLAGS.metric_eval_interval_steps == 0:
        model.mode = 'valid'
        eval_step = int(global_step.numpy() / FLAGS.metric_eval_interval_steps)
        if hasattr(model, 'eval'):
          model.eval()
        with torch.no_grad():
          results = None
          if eval_fn and not 'dataset' in inspect.getfullargspec(eval_fn).args:
            names = valid_names if valid_names is not None else [infer_names[0]] + [x + '_y' for x in infer_names[1:]] + infer_names[1:] if infer_names else None
            is_last = False
            if FLAGS.nvs and int(global_step.numpy() / FLAGS.metric_eval_interval_steps) == FLAGS.nvs:
              is_last = True
              if swa_model is not None:
                torch.optim.swa_utils.update_bn(dataset, swa_model)
            ic(is_last, global_step.numpy(), int(global_step.numpy() / FLAGS.metric_eval_interval_steps), global_step.numpy() == int(num_steps_per_epoch * FLAGS.num_epochs))
            resutls = evaluate(model, valid_dataset, eval_fn, eval_keys, None, 
                              names, valid_write_fn, write_streaming,
                              num_valid_steps_per_epoch, num_valid_examples, step=eval_step, sep=sep, out_hook=out_hook, 
                              wirte=write_valid, is_last=is_last)
          else:
            eval_fn = eval_fn or evaluate_fn
            ic(is_last, global_step.numpy(), int(global_step.numpy() / FLAGS.metric_eval_interval_steps), global_step.numpy() == int(num_steps_per_epoch * FLAGS.num_epochs))
            if eval_fn is not None:
              kwargs = {}   
              if 'show' in args:
                kwargs['show'] = True
              if 'info' in args:
                kwargs['info'] = x
              if 'x' in args:
                kwargs['x'] = x
              if 'model' in args:
                kwargs['model'] = model
              if 'eval_step' in args:
                kwargs['eval_step'] = eval_step
              if 'step' in args:
                kwargs['step'] = global_step.numpy()
              if 'is_last' in args:
                kwargs['is_last'] = global_step.numpy() == int(num_steps_per_epoch * FLAGS.num_epochs)
              if 'steps' in args:
                kwargs['steps'] = num_valid_steps_per_epoch
              if 'loss_fn' in args:
                kwargs['loss_fn'] = loss_fn
              if 'ofile' in args:
                kwargs['ofile'] = None
              if 'outdir' in args:
                kwargs['outdir'] = FLAGS.model_dir
              if 'num_examples' in args:
                kwargs['num_examples'] = num_valid_examples
              if 'return_dict' in args:
                kwargs['return_dict'] = True
              if 'desc' in args:
                kwargs['desc'] = 'eval'
              # if 'is_last' in args:
              #   if FLAGS.nvs and int(global_step.numpy() / FLAGS.metric_eval_interval_steps) == FLAGS.nvs:
              #     kwargs['is_last'] = True
              #   else:
              #     kwargs['is_last'] = False
                  
              if FLAGS.parts and not FLAGS.use_shard:
                from husky.callbacks.evaluate import _prepare_eval_part
                valid_dataset, steps, num_valid_examples = _prepare_eval_part(FLAGS.part, FLAGS.parts)
                if 'steps' in kwargs:
                  kwargs['steps'] = steps
                if 'num_examples' in kwargs:
                  kwargs['num_examples'] = num_valid_examples
                if FLAGS.parts:
                  if 'desc' in kwargs:
                    kwargs['desc'] = f'eval: {FLAGS.part}/{FLAGS.parts}'
              
              if is_last:
                if swa_model is not None:
                  torch.optim.swa_utils.update_bn(dataset, swa_model)
              
              results = eval_fn(valid_dataset, show=True, **kwargs)
              
        if FLAGS.torch:
          if not FLAGS.torch_lr:
            # control learning rate by tensorflow learning rate
            for param_group in optimizer.params:
              # important learning rate decay
              param_group['lr'] = learning_rate.numpy()
              
        model.mode = 'train'
        if hasattr(model, 'train'):  
          model.train()

      epoch_save = False
      if rank == 0:
        # TODO save ok ?
        def _torch_save(save_epoch=False):
          epoch = int(global_step.numpy() / num_steps_per_epoch)
          state = {
                  'epoch': epoch,
                  'step': global_step.numpy(),
                  'state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                }
          if FLAGS.save_optimizer:
            state['optimizer'] = optimizer.state_dict()
          if FLAGS.train_hour:
            state['train_hour'] = FLAGS.train_hour
          if FLAGS.valid_hour:
            state['valid_hour'] = FLAGS.valid_hour
          if FLAGS.train_loop:
            state['round'] = total_rounds + FLAGS.round + 1
            state['start_hour'] = FLAGS.train_hour if not start_hour else start_hour
          
          model_path = f'{FLAGS.model_dir}/model.pt'
          logging.debug(f'save checkpint to {model_path}')
          torch.save(state, model_path) 
          gezi.write_to_txt(os.path.basename(model_path), f'{FLAGS.model_dir}/checkpoint.txt')

        if global_step.numpy() % FLAGS.save_interval_steps == 0:
          if FLAGS.torch:
            _torch_save()
          else:
            manager.save(global_step.numpy())

        epoch_save = False
        if FLAGS.save_interval_epochs and global_step.numpy() % int(num_steps_per_epoch * FLAGS.save_interval_epochs) == 0:
          epoch_ = global_step.numpy() / num_steps_per_epoch if num_steps_per_epoch else None
          if FLAGS.torch:
            # latest_model_path = melt.latest_checkpoint(FLAGS.model_dir)
            _torch_save(save_epoch=True)
            # if latest_model_path:
            #   os.remove(latest_model_path)
          else:
            # logging.info(f'save epoch {int(global_step.numpy() / num_steps_per_epoch)} checkpint to {checkpoint_prefix_epoch}')
            epoch_manager.save(int(global_step.numpy() / num_steps_per_epoch))
          epoch_save = True
          
      if FLAGS.learning_rate_decay_factor > 0:
        if global_step.numpy() >= decay_start_step and global_step.numpy() % decay_steps == 0:
          lr = max(learning_rate.numpy() * FLAGS.learning_rate_decay_factor, FLAGS.min_learning_rate)
          if lr < learning_rate.numpy():
            learning_rate.assign(lr)
            if FLAGS.torch:
              for param_group in optimizer.params:
                param_group['lr'] = learning_rate.numpy()
      
      epoch_valid = False
      is_last = global_step.numpy() == int(num_steps_per_epoch * FLAGS.num_epochs)
      if valid_dataset and FLAGS.metric_eval and  FLAGS.valid_interval_epochs > 0 and \
        (global_step.numpy() % int(num_steps_per_epoch * FLAGS.valid_interval_epochs + 0.5) == 0 or is_last) \
            or FLAGS.first_interval_epoch > 0 and \
              global_step.numpy() == int(num_steps_per_epoch * FLAGS.first_interval_epoch):
        epoch_valid = True
        FLAGS.train_time = timer_.elapsed_minutes()
        if FLAGS.write_valid_final and global_step.numpy() % num_steps_per_epoch == FLAGS.num_epochs: 
          write_valid = True
        model.mode = 'valid'
        if hasattr(model, 'eval'):
          model.eval()

        if rank == 0:
          step = int(global_step.numpy() / int(num_steps_per_epoch * FLAGS.valid_interval_epochs))
          melt.inc_train_step()
          step = melt.inc_eval_step(save_file=(not FLAGS.async_valid))

        if not FLAGS.async_valid:
          results = None
          args = inspect.getfullargspec(eval_fn).args
          # ic(is_last, global_step.numpy(), global_step.numpy() == int(num_steps_per_epoch * FLAGS.num_epochs))
          if eval_fn and not 'dataset' in args:
            results = metric_eval_fn(is_last=is_last)
          else:
            eval_fn = eval_fn or evaluate_fn
            if eval_fn is not None:
              kwargs = {}   
              if 'info' in args:
                kwargs['info'] = x
              if 'x' in args:
                kwargs['x'] = x
              if 'model' in args:
                kwargs['model'] = model
              if 'other' in args:
                kwargs['other'] = x
              if 'eval_step' in args:
                kwargs['eval_step'] = melt.get_eval_step()
              if 'step' in args:
                kwargs['step'] = global_step.numpy()
              if 'is_last' in args:
                kwargs['is_last'] = is_last
              if 'steps' in args:
                kwargs['steps'] = num_valid_steps_per_epoch
              if 'loss_fn' in args:
                kwargs['loss_fn'] = loss_fn
              if 'ofile' in args:
                kwargs['ofile'] = None
              if 'outdir' in args:
                kwargs['outdir'] = FLAGS.model_dir
              if 'num_examples' in args:
                kwargs['num_examples'] = num_valid_examples
              if 'return_dict' in args:
                kwargs['return_dict'] = True
              if 'desc' in args:
                kwargs['desc'] = 'eval'
              
              if FLAGS.parts and not FLAGS.use_shard:
                from husky.callbacks.evaluate import _prepare_eval_part
                valid_dataset, steps, num_valid_examples = _prepare_eval_part(FLAGS.part, FLAGS.parts)
                if 'steps' in kwargs:
                  kwargs['steps'] = steps
                if 'num_examples' in kwargs:
                  kwargs['num_examples'] = num_valid_examples
                if FLAGS.parts:
                  if 'desc' in kwargs:
                    kwargs['desc'] = f'eval: {FLAGS.part}/{FLAGS.parts}'
              
              results = eval_fn(valid_dataset, **kwargs)

      if epoch_save and epoch_valid:
        logging.debug(f'Round:{FLAGS.round}', 'Epoch:%.1f' % epoch_ , f'Train:[{FLAGS.train_hour}] Valid:[{FLAGS.valid_hour}]', 'TrainTime:{:.1f}m'.format(FLAGS.train_time))
        if FLAGS.async_valid and valid_dataset:
          FLAGS.total_time = (time.time() - gezi.get_global('start_time')) / 60
          _async_valid()

      if test_dataset and FLAGS.do_test and global_step.numpy() % int(num_steps_per_epoch * FLAGS.inference_interval_epochs) == 0:
        model.mode = 'test'
        if hasattr(model, 'eval'):
          model.eval()
        if inference_fn is None:
          logging.info('write infer result with model_path', model_path)
          inference(model, test_dataset, model_path, 
                    infer_names, infer_debug_names, infer_write_fn, write_streaming,
                    num_test_steps_per_epoch, num_test_examples, suffix=infer_suffix, sep=sep)
        else:
          inference_fn(model, test_dataset, tf.train.latest_checkpoint(ckpt_dir), num_test_steps_per_epoch)

      # might be used for continue training
      if num_epochs and num_steps_per_epoch and int(global_step.numpy() / num_steps_per_epoch) >= epoch + 1:
        break
      if num_epochs and num_steps_per_epoch and int(global_step.numpy() / num_steps_per_epoch) >= num_epochs:
        logging.info(f'Finshed training of {num_epochs} epochs')
        break
      
      pbar.update(1)
    
    FLAGS.total_time = (time.time() - gezi.get_global('start_time')) / 60
    logging.debug(f'Round:{FLAGS.round} Train:{FLAGS.train_hour} Valid:{FLAGS.valid_hour}', 'TotalTime:{:.1f}m'.format(FLAGS.total_time))
    _on_epoch_end(FLAGS.model_dir, FLAGS.log_dir, save_model=True, del_model_path=latest_checkpoint)

    for callback in callbacks:
      if hasattr(callback, 'on_epoch_end'):
        kwargs = {}
        if 'lr' in inspect.getargspec(callback.on_epoch_end).args:
          kwargs['lr'] = learning_rate
        callback.on_epoch_end(epoch, **kwargs)
  
  for callback in callbacks:
    if hasattr(callback, 'on_train_end'):
      callback.on_train_end()
  
  if rank == 0:
    melt.inc_total_step(int(num_steps_per_epoch * num_epochs))

  if FLAGS.enable_profiling:
    with open(f"{FLAGS.log_dir}/prof.txt", "w") as prof_f:
      logging.info(f'write profiling to {FLAGS.log_dir}/prof.txt')
      prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
    logging.info(f'write profiling chrome trace to {FLAGS.log_dir}/prof.json')
    prof.export_chrome_trace(f"{FLAGS.log_dir}/prof.json")
  
  if FLAGS.train_loop:
    logging.info(f'Done for {FLAGS.train_input}')

  # worker.stop()
  pbar.close()

  return 0

