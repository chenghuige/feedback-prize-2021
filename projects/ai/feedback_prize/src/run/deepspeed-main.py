#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   torch-train.py
#        \author   chenghuige  
#          \date   2019-08-02 01:05:59.741965
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

import torch
from torch import nn
from torch.nn import functional as F

from src import config
from src.config import *
from src.torch.model import Model
from src.torch.dataset import Dataset
from src.torch.loss import calc_loss
from src import util

import melt
import gezi
import lele

import argparse
import deepspeed
from gezi import tqdm
import inspect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=1,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args

import threading
import queue
import time

class AsyncWorker(threading.Thread):
    def __init__(self, dataloader, total):
        threading.Thread.__init__(self)
        self.req_queue = queue.Queue()
        self.ret_queue = queue.Queue()
        self.dataloader = iter(dataloader)
        self.total = total
        self.prefetch_idx = 3
        for i in range(self.prefetch_idx):
            self.req_queue.put(1)

    def run(self):
        while True:
            dataset_type = self.req_queue.get(block=True)
            if dataset_type is None:
                break
            batch = next(self.dataloader)
            self.req_queue.task_done()
            self.ret_queue.put(batch)

    def get(self):
        batch = self.ret_queue.get()
        self.ret_queue.task_done()
        return batch

    def __iter__(self):
      return self.get()

    def prefetch(self):
        if self.prefetch_idx < self.total:
            self.req_queue.put(1)
            self.prefetch_idx += 1

    def stop(self):
        self.req_queue.put(None)

def main(_):
  config.init()
  mt.init()
  FLAGS.torch_only = True
  os.system(f'cp ./main.py {FLAGS.model_dir}')
  os.system(f'cp ./config.py {FLAGS.model_dir}')
  os.system(f'cp ./torch/model.py {FLAGS.model_dir}')
  os.system(f'cp ./torch/loss.py {FLAGS.model_dir}')
  dataset_meta_root = '..'
  os.system(f'cp {dataset_meta_root}/dataset-metadata.json {FLAGS.model_dir}')   

  train_dl, eval_dl, valid_dl = util.get_dataloaders()

  model_name = FLAGS.model
  model = Model()

  loss_fn = Criterion()

  dist = gezi.get('dist')
  rank = dist.get_rank()
  dev_count = torch.cuda.device_count()  
  
  global device
  if dev_count > 1:
    device = torch.device('cuda', rank)
  
  gezi.set('device', device)

  parameters = filter(lambda p: p.requires_grad, model.parameters())
  args = add_argument()

  model, optimizer, _, __ = deepspeed.initialize(
      args=args, model=model, model_parameters=parameters)

  batch_size_ = model.train_micro_batch_size_per_gpu()

  batch_size = model.train_batch_size()

  grad_steps = model.gradient_accumulation_steps()
  
  kwargs = {}
  kwargs['world_size'] = dist.get_world_size()
  kwargs['rank'] = dist.get_rank()

  worker = AsyncWorker(train_dl, len(train_dl), 10)
  worker.start()

  # l = []
  # t = tqdm(enumerate(train_dataset), total=num_steps_per_epoch, ascii=True)
  # for i, (x, y) in t:
  #   x, y = lele.to_torch(x, y, cuda=False)
  #   l.append((x, y))
  # train_dataset = l

  avg_loss = lele.PytMean()
  num_epochs = args.epochs
  step = 0
  for epoch in range(args.epochs):
    desc = 'train:%d' % epoch
    # t = tqdm(enumerate(train_dataset), total=num_steps_per_epoch, desc=desc, ascii=True)
    # for i, (x, y) in t:
    t = tqdm(range(len(train_dl)), desc=desc, ascii=True)
    for i in t:
      batch = worker.get()
      x, y = batch

      postfix = {}
      if gezi.get('loss'):
        postfix['loss'] = gezi.get('loss')
      if gezi.get('valid_loss'):
        postfix['val_loss'] = gezi.get('valid_loss')
      t.set_postfix(postfix)

      x, y = lele.to_torch(x, y)
      y_ = model(x)
      loss = calc_loss(y_, y, x, i, step=step, epoch=epoch, training=True)

      worker.prefetch()
      model.backward(loss)
      model.step()

      avg_loss(loss)
      gezi.set('loss', '%.4f' % loss.item())
          
    step += 1
  
  worker.stop()

  # eval_fn, eval_keys = util.get_eval_fn_and_keys()
  # valid_write_fn = ev.valid_write
  # out_hook = ev.out_hook

  # weights = None if not FLAGS.use_weight else 'weight'

  # fit(model,  
  #     loss_fn,
  #     Dataset,
  #     eval_fn=eval_fn,
  #     eval_keys=eval_keys,
  #     valid_write_fn=valid_write_fn,
  #     out_hook=out_hook,
  #     weights=weights)

  print('-----------DONE')
   
if __name__ == '__main__':
  main(None)
  
