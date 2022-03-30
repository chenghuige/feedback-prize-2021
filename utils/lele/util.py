#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2018-10-17 06:52:08.997327
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import flags
FLAGS = flags.FLAGS

from typing import Callable
import pandas as pd

import tensorflow as tf
import torch
from torch import full_like, nn
import torch.utils.data
#from torch.utils.data import Dataset, ConcatDataset

import copy
import random
import traceback
import numpy as np
import itertools
from datasets import Dataset

import gezi 
from gezi import tqdm
logging = gezi.logging

def adjust_lrs(x, ratio=None, name='learning_rate_weights'):
  import tensorflow as tf
  if ratio is None:
    ratios = tf.compat.v1.get_collection(name)[-1].numpy()
    # TODO will this hurt performance ? change to use learning rate weights without tf dependence?
    ratios = torch.as_tensor(ratios).cuda()
    x = x * ratios + x.detach() * (1 - ratios)
  else:
    x = x * ratio + x.detach() * (1 - ratio)
  return x 

def get_device():
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_weights(model, path, map_location=None, return_checkpoint=False, return_updated=False, renames={}, to_device=True, eval=True): 
  try:
    checkpoint = torch.load(path, map_location=map_location)
  except Exception:    
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
  state = checkpoint['state_dict']   
  
  # ic(gezi.get_mem_gb())
  model_ = model.module if hasattr(model, 'module') else model
  full_update = True
  model_state_dict = model_.state_dict()
  mismatch_ignores = set()
  for key in model_state_dict:
    if key not in state:
      full_update = False
    elif state[key].shape != model_state_dict[key].shape:
      mismatch_ignores.add(key)
      full_update = False
  if full_update:
    model_.load_state_dict(state)
  else:
    new_params = model_state_dict
    # ic(new_params.keys())
    if not renames:
      new_params.update({k:v for k, v in state.items() if k in new_params and k not in mismatch_ignores})
    else:
      ori = list(renames.keys())[0]
      dest = list(renames.values())[0]
      new_params.update({k.replace(ori, dest): v for k, v in state.items() if k.replace(ori, dest) in new_params and k.replace(ori, dest) not in mismatch_ignores})
    
    # ic(new_params.keys())
    model_.load_state_dict(new_params)
  
  if not return_checkpoint:
    del state
    del checkpoint
    
  if to_device:
    device = get_device()
    model.to(device)
  if eval:
    model.eval()
  
  if not return_checkpoint:
    return
  
  if not return_updated:
    return checkpoint

  updated_params = []
  for name, param in model_.named_parameters():
    if name in state:
      updated_params.append(param)

  return checkpoint, updated_params 

# def load_weights(model, path, map_location=None, return_checkpoint=False, return_updated=False, renames={}, to_device=True, eval=True): 
#   try:
#     checkpoint = torch.load(path, map_location=map_location)
#   except Exception:    
#     checkpoint = torch.load(path, map_location=torch.device('cpu'))
#   state = checkpoint['state_dict']   
  
#   # ic(gezi.get_mem_gb())
#   model_ = model.module if hasattr(model, 'module') else model
#   full_update = True
#   for key in  model_.state_dict():
#     if key not in state:
#       full_update = False
#   if full_update:
#     model_.load_state_dict(state)
#   else:
#     new_params = model_.state_dict()
#     # ic(new_params.keys())
#     if not renames:
#       new_params.update({k:v for k, v in state.items() if k in new_params})
#     else:
#       ori = list(renames.keys())[0]
#       dest = list(renames.values())[0]
#       new_params.update({k.replace(ori, dest): v for k, v in state.items() if k.replace(ori, dest) in new_params})
    
#     # ic(new_params.keys())
#     model_.load_state_dict(new_params)
  
#   if not return_checkpoint:
#     del state
#     del checkpoint
    
#   if to_device:
#     device = get_device()
#     model.to(device)
#   if eval:
#     model.eval()
  
#   if not return_checkpoint:
#     return
  
#   if not return_updated:
#     return checkpoint

#   updated_params = []
#   for name, param in model_.named_parameters():
#     if name in state:
#       updated_params.append(param)

#   return checkpoint, updated_params 

load = load_weights

def save_model(model, model_dir, model_name='model.pt', fp16=False):
  if fp16:
    model.half()
  state = {
            'state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
          }
  if os.path.isdir(model_dir):
    torch.save(state, f'{model_dir}/{model_name}')
  else:
    torch.save(state, model_dir)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

try:
  import torch 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
  pass
import numpy as np 


def torch_(x, cuda=True):
  global device
  if FLAGS.torch_only:
    return x
  for dim in x.shape:
    if dim == 0:
      return x

  # if tf.__version__ < '2':
  x = x.numpy()

  device = gezi.get('device') or device

  if x.dtype == np.int64 or x.dtype == np.int32 or x.dtype == np.float32 or x.dtype == np.float64:
    x = torch.as_tensor(x)
    if cuda:
      x = x.to(device)

  return x

def to_torch(x, y=None, cuda=True, torch_only=False):
  global device
  if torch_only or FLAGS.torch_only:
    if cuda:
      device = gezi.get('device') or device
      for key in x:
        if type(x[key]) != np.ndarray and not isinstance(x[key], (list, tuple)):
          x[key] = x[key].to(device)
      return x, y.to(device)
    else:
      return x, y

  if y is not None:
    y = torch_(y, cuda)

  if not isinstance(x, dict):
    x = torch_(x, cuda)
  else:
    for key in x:
      x[key] = to_torch(x[key], cuda=cuda)
      
  if y is None:
    return x
  else:
    return x, y

#---------------padding input data

#https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/12

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=vec.dtype)], dim=dim)

class PadCollate2:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        
    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max([torch.Tensor(x[0]).shape[self.dim] for x in batch])
        #print('----------', max_len)
        # pad according to max_len
        batch = [(pad_tensor(torch.Tensor(x[0]), pad=max_len, dim=self.dim), x[1]) for x in batch]
        # stack all
        xs = torch.stack([torch.Tensor(x[0]) for x in batch], dim=0)
        ys = torch.Tensor([x[1] for x in batch])
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
      
class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        
    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max([x[0].size(self.dim) for x in batch])
        #print('----------', max_len)
        # pad according to max_len
        batch = [(pad_tensor(x[0], pad=max_len, dim=self.dim), x[1]) for x in batch]
        # stack all
        xs = torch.stack([x[0] for x in batch], dim=0)
        ys = torch.Tensor([x[1] for x in batch])
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

class NpDictPadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        
    def pad_collate(self, batch):
      ys = [None] * len(batch)
      input = {}
      ys[0] = batch[0][1]
      max_len = {}
      
      for key, val in batch[0][0].items():
        if isinstance(val, np.ndarray):
          val = torch.as_tensor(val)
          max_len[key] = len(val)
        else:
          if isinstance(val, list):
            if type(val[0]) == int:
              val = torch.as_tensor(np.asarray(val))
            else:
              val = torch.as_tensor(np.asarray(val)).float()
            max_len[key] = len(val)
        input[key] = [val] * len(batch)
       
      for i in range(1, len(batch)):
        ys[i] = batch[i][1]
        for key, val in batch[i][0].items():
          if isinstance(val, np.ndarray):
            val = torch.as_tensor(val)
            if len(val) > max_len[key]:
              max_len[key] = len(val)
          else:
            if isinstance(val, list):
              if type(val[0]) == int:
                val = torch.as_tensor(np.asarray(val))
              else:
                val = torch.as_tensor(np.asarray(val)).float()
              if len(val) > max_len[key]:
                max_len[key] = len(val)
          input[key][i] = val
          
      for key, val_list in input.items():
        if key in max_len:
          for i in range(len(val_list)):
            val_list[i] = pad_tensor(val_list[i], pad=max_len[key], dim=self.dim)
            #print(i, val_list[i].shape, max_len[key])
    
          input[key] = torch.stack(val_list, dim=0)
        else:
          #... TODO why np.arry.dtype not dp.str_ but <U3 <U4 ?
          input[key] = np.asarray(input[key])
          if type(input[key][0]) != np.str_:
            input[key] = torch.as_tensor(input[key])
            
      ys = torch.as_tensor(np.asarray(ys))
      return input, ys
        
    def __call__(self, batch):
        return self.pad_collate(batch)
      
class DictPadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        
    def pad_collate(self, batch):
      ys = [None] * len(batch)
      input = {}
      ys[0] = batch[0][1]
      max_len = {}
      
      for key, val in batch[0][0].items():
        #if not isinstance(val, str):
        if isinstance(val, torch.Tensor):
          if not len(val.size()):
            val = val.expand(1)
          max_len[key] = val.size(self.dim)
        input[key] = [val] * len(batch)
       
      for i in range(1, len(batch)):
        ys[i] = batch[i][1]
        for key, val in batch[i][0].items():
          #if not isinstance(val, str):
          if isinstance(val, torch.Tensor):
            if not len(val.size()):
              val = val.expand(1)
            if len(val) > max_len[key]:
              max_len[key] = val.size(self.dim)
          input[key][i] = val
          
      for key, val_list in input.items():
        if key in max_len:
          for i in range(len(val_list)):
            val_list[i] = pad_tensor(val_list[i], pad=max_len[key], dim=self.dim)  
          input[key] = torch.stack(val_list, dim=0)
        else:
          input[key] = np.array(input[key])

      #list of tensor ->
      ys = torch.stack(ys, dim=0)
      return input, ys
        
    def __call__(self, batch):
      return self.pad_collate(batch)

# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def keras_init(model, emb=True, linear=True):
  for m in model.modules():
    if emb:
      if isinstance(m, (nn.Embedding, nn.EmbeddingBag)):
        if m.weight.requires_grad:
          nn.init.uniform_(m.weight, -0.05, 0.05)
    if linear:
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        
def keras_init_children(model, emb=True, linear=False):
  for m in model.children():
    if emb:
      if isinstance(m, (nn.Embedding, nn.EmbeddingBag)):
        if m.weight.requires_grad:
          nn.init.uniform_(m.weight, -0.05, 0.05)
    if linear:
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        
def keras_weights(x):
  if isinstance(x, (nn.Embedding, nn.EmbeddingBag)):
    if x.weight.requires_grad:
      nn.init.uniform_(x.weight, -0.05, 0.05)
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(x.weight)
    nn.init.zeros_(x.bias)
    
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

  def __call__(self, val=None):
    if val is None:
      return self.result()
    if not self.is_call:
      self.clear()
      self.is_call = True
    self._val += val.item()
    self.count += 1

  def result(self):
    if self.is_call:
      self.is_call = False
    if not self.count:
      val = 0
    else:
      val = self._val / self.count
    # TODO just for compact with tf ..
    return PytObj(val)

  def numpy(self):
    return self.result().numpy()
  
def predicts(model, inputs, batch_size=None, desc='Predicting', dynamic_keys=[], mask_key=None):
  with torch.no_grad():
    assert isinstance(inputs, dict)
    assert 0 in inputs
    dataloaders = []
    other_inputs = {}
    for i, inputs_ in inputs.items():
      if isinstance(i, int):
        # ic(i, inputs_.keys())
        inputs__ = {}
        for key in inputs_:
          try:
            if not type(inputs_[key][0]) in [np.str_, str]:
              inputs__[key] = inputs_[key] 
          except Exception:
            ic(key)
        inputs_ = inputs__
        dataset = Dataset.from_dict(inputs_)
        device = get_device()
        dataset.set_format(type='torch', device=device)
        assert batch_size, 'need batch size if your inputs is not dataloader but dict'
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        dataloaders.append(dataloader)
      else:
        if not type(inputs_[0]) in [np.str_, str]:
          other_inputs[i] = inputs_
        
    if other_inputs:
      dataset = Dataset.from_dict(other_inputs)
      device = get_device()
      dataset.set_format(type='torch', device=device)
      assert batch_size, 'need batch size if your inputs is not dataloader but dict'
      dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
      dataloader = iter(dataloader)
    
    res = None
    total = len(dataloaders[0])
    dataloaders = [iter(x) for x in dataloaders]
    for i in tqdm(range(total), desc=desc):
      inputs_list = [next(dataloaders[j]) for j in range(len(dataloaders))]
      inputs = {}
      for j, inputs_ in enumerate(inputs_list):
        if mask_key is not None:
          max_len = inputs_[mask_key].sum(1).max()
          for key in dynamic_keys + [mask_key]:
            if key in inputs_:
              inputs_[key] = inputs_[key][:,:max_len]
        inputs[j] = inputs_
      if other_inputs:
        inputs.update(next(dataloader))
      preds = model(inputs)
      if isinstance(preds, dict):
        if not res:
          res = {key: [] for key in preds}
        for key in preds:
          res[key].append(gezi.squeeze(preds[key].detach().cpu().numpy()))
      else:
        if not res:
          res = []
        res.append(gezi.squeeze(preds.detach().cpu().numpy()))
      
    if isinstance(res, dict):
      for key in res:
        try:
          res[key] = np.concatenate(res[key])
        except Exception:
          # l = []
          # for l_ in res[key]:
          #   l.extend(l_)
          # res[key] = l
          res[key] = list(itertools.chain(*res[key]))
    else:
      try:
        res = np.contanate(res)
      except Exception:
        res = list(itertools.chain(*res))
    
    return res
  
      
def predict(model, inputs, batch_size=None, desc='Predicting', dynamic_keys=[], mask_key=None):
  with torch.no_grad():
    if isinstance(inputs, dict):
      if 0 in inputs:
        return predicts(model, inputs, batch_size, desc, dynamic_keys, mask_key)
      inputs_ = {}
      for key in inputs:
        if (not type(inputs[key][0]) in [np.str_, str]):
          inputs_[key] = inputs[key]  
      inputs = inputs_ 
      # TODO support dynamic length with data collactor padding to max lenght in a batch
      dataset = Dataset.from_dict(inputs)
      device = get_device()
      dataset.set_format(type='torch', device=device)
      assert batch_size, 'need batch size if your inputs is not dataloader but dict'
      dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    else:
      dataloader = inputs
    res = None
    for inputs in tqdm(dataloader, desc=desc):
      if mask_key is not None:
        max_len = inputs[mask_key].sum(1).max()
        for key in dynamic_keys + [mask_key]:
          if key in inputs:
            inputs[key] = inputs[key][:,:max_len]
      preds = model(inputs)
      if isinstance(preds, dict):
        if not res:
          res = {key: [] for key in preds}
        for key in preds:
          res[key].append(gezi.squeeze(preds[key].detach().cpu().numpy()))
      else:
        if not res:
          res = []
        res.append(gezi.squeeze(preds.detach().cpu().numpy()))
    
    if isinstance(res, dict):
      for key in res:
        try:
          res[key] = np.concatenate(res[key])
        except Exception as e:
          # ic(key, e)
          l = []
          for l_ in res[key]:
            l.extend(l_)
          res[key] = l
    else:
      try:
        res = np.contanate(res[key])
      except Exception:
        l = []
        for l_ in res:
          l.extend(l_)
        res = l
    return res
  
  
def get_tfrecord_inputs(TFRecordDataset, files, bs=512):
  ds = TFRecordDataset()
  dl = ds.make_batch(bs, filenames=files, return_numpy=True)
  inputs = None
  for x, y in tqdm(dl, total=ds.num_steps, desc=files[0], leave=False):
    if not inputs:
      inputs = {k: list(v) for k, v in x.items()}
      inputs['y'] = list(y)
    else:
      for key in x:
        inputs[key].extend(list(x[key]))
      inputs['y'].extend(list(y)) 
  for k in inputs:
    inputs[k] = np.asarray(inputs[k])
    try:
      inputs[k] = torch.as_tensor(inputs[k])
    except Exception:
      pass
  return inputs

# https://github.com/ufoym/imbalanced-dataset-sampler/blob/c2ef9d9529f2eb25306aab5e199a99eff455b2cd/torchsampler/imbalanced.py#L40
# from torchsampler import ImbalancedDatasetSampler

# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     sampler=ImbalancedDatasetSampler(train_dataset),
#     batch_size=args.batch_size,
#     **kwargs
# )

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        # elif isinstance(dataset, torchvision.datasets.MNIST):
        #     return dataset.train_labels.tolist()
        # elif isinstance(dataset, torchvision.datasets.ImageFolder):
        #     return [x[1] for x in dataset.imgs]
        # elif isinstance(dataset, torchvision.datasets.DatasetFolder):
        #     return dataset.samples[:][1]
        # elif isinstance(dataset, torch.utils.data.Subset):
        #     return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
      
def seed_everything(seed: int):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
    
def freeze(model):
  for param in model.parameters():
    param.requires_grad = False
    
def unfreeze(model):
  for param in model.parameters():
    param.requires_grad = True
        
def get_word_embeddings(backbone):
  if hasattr(backbone, 'word_embedding'):
    # xlnet
    return backbone.word_embedding
  if hasattr(backbone, 'embeddings'):
    # most bert models
    if hasattr(backbone.embeddings, 'word_embedding'):
      return backbone.embeddings.word_embedding
    else:
      # deberta-v2
      return backbone.embeddings.word_embeddings
  if hasattr(backbone, 'shared'):
    return backbone.shared
  if hasattr(backbone, 'wte'):
    # gpt2
    return backbone.wte

def get_optimizer_params(model, backbone_lr=None, base_lr=None, weight_decay=False):
  ## 去掉了weight decay 似乎影响不大 不过目前线上的short模型仍然是之前带有weight decay模式训练出来的
  optimizer_parameters = []
  if weight_decay:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if backbone_lr is None:
      # 之前配置 不能完全确定 似乎weight decay降低了集成效果?
      optimizer_parameters = [
          {
              "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
              "weight_decay": 0.01,
          },
          {
              "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
              "weight_decay": 0.0,
          },
      ]
    else:
      backbone_params = model.backbone.parameters()
      backbone_params = list(map(id, backbone_params))
      
      optimizer_parameters = [
          {
              "params": [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and (id(p) in backbone_params)],
              "weight_decay": 0.01,
              'lr': backbone_lr,
          },
          {
              "params": [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and (not id(p) in backbone_params)],
              "weight_decay": 0.01,
              'lr': base_lr, 
          },
          {
              "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and (id(p) in backbone_params)],
              "weight_decay": 0.0,
              'lr': backbone_lr,
          },
          {
              "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and (not id(p) in backbone_params)],
              "weight_decay": 0.0,
              'lr': base_lr,
          },
      ]
  else:
    if backbone_lr is not None:
      backbone_params = model.backbone.parameters()
      backbone_params = list(map(id, backbone_params))
      # ic([p for p in model.parameters() if (id(p) in backbone_params)])
      # ic([p for p in model.parameters() if (not id(p) in backbone_params)])
      optimizer_parameters = [
          {
              "params": [p for p in model.parameters() if (id(p) in backbone_params)],
              'lr': backbone_lr,
          },
          {
              "params": [p for p in model.parameters() if (not id(p) in backbone_params)],
              'lr': base_lr, 
          }
      ]
  return optimizer_parameters
  
class FreezeCallback(object):
  def __init__(self, model, freeze_epochs=1):
    self.model = model
    self.freeze_epochs = freeze_epochs
  
  def on_train_start(self):
    if self.freeze_epochs > 0:
      ic('freeze model', self.freeze_epochs)
      freeze(self.model)
    
  def on_epoch_end(self, epoch):
    if self.freeze_epochs > 0 and (epoch + 1) == self.freeze_epochs:
      ic('unfreeze model', epoch)
      unfreeze(self.model)
      