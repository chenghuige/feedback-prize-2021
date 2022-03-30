#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   test-dataset.py
#        \author   chenghuige  
#          \date   2022-01-14 23:57:13.373154
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from curses import def_shell_mode
from operator import index

import sys 
sys.path.append('..')
import os
import random
import numpy as np
import gezi
from gezi import tqdm
import melt as mt
import lele
from src.dataset import Dataset as TFRecordDataset
from torch.utils.data import Dataset as TorchDataset
import torch
from src import util
from absl import app, flags
FLAGS = flags.FLAGS

class Dataset(TorchDataset):
  def __init__(self, files, subset='valid', files_list=[], indexes=[]):
    self.inputs_list = []
    ic(subset, files[:2])
    inputs = lele.get_tfrecord_inputs(TFRecordDataset, files)
    ids = gezi.decode(gezi.squeeze(inputs['id']))
    # ic(subset, ids, len(ids))
    if subset == 'valid':
      gezi.set('id', ids)
      gezi.set('valid_ids', set(ids))
      # ic(subset, len(gezi.get('valid_ids')))
      # ic(subset, len((gezi.get('valid_ids') & gezi.get('train_ids'))))
      if not FLAGS.online:
        assert len((gezi.get('valid_ids') & gezi.get('train_ids'))) == 0
    else:
      gezi.set('train_ids', set(ids))
      # ic(len(gezi.get('train_ids')))
      
    self.inputs_list.append(inputs)

    self.subset = subset
    for files in files_list:
      inputs = lele.get_tfrecord_inputs(TFRecordDataset, files)
      # ic(subset, ids, len(ids))
      if subset == 'train':
        ids2 = gezi.get('train_ids') | set(ids)
        ids = gezi.decode(gezi.squeeze(inputs['id']))
        # ic(subset, len(ids), len(ids2))
        assert len(ids) == len(ids2)
      self.inputs_list.append(inputs)
        
    num_datasets = len(self.inputs_list)
    if not indexes:
      indexes = [0] * (num_datasets - 2) + list(range(num_datasets))
      # indexes = list(range(num_datasets))
    else:
      assert len(indexes) >= num_datasets
      indexes = [int(x) for x in indexes if int(x) < num_datasets]
    self.indexes = indexes
    self.aug_index = len([x for x in indexes if x == 0])
    if subset == 'train':
      ic(num_datasets, indexes, FLAGS.aug_start_epoch)
    
    self.inputs = self.inputs_list[0]
    # ic(len(self.inputs['y']))
    self.ignores = set(['id'])
    
    classes = self.inputs['classes'].numpy().astype(int)
    num_classes = classes.shape[1]
    classes *= np.asarray([2 ** (i - num_classes + 1) for i in range(num_classes)]).astype(int)
    self.labels = classes.sum(-1)
    
    self.epoch = 0
    aug_rates = [float(x) for x in FLAGS.aug_rates] if FLAGS.aug_rates else [None] 
    aug_rates = aug_rates + [aug_rates[-1]] * 10000
    self.aug_rates = aug_rates
    # ic(self.aug_rates[:5])

  def getitem(self, inputs, idx):
    x = {k: v[idx] for k, v in inputs.items() if k not in self.ignores}
    y = inputs['y'][idx]
        
    return x, y

        
  def __getitem__(self, idx):
    if not FLAGS.multi_inputs:
      index = 0
      if self.subset == 'train' and len(self.inputs_list) >= 1:
        aug_rate = self.aug_rates[int(self.epoch)]
        if aug_rate is None:
          index = random.choice(self.indexes)
        else:
          p = random.random()
          if p < (1. - aug_rate):
            index = 0
          else:
            index = random.choice(self.indexes[self.aug_index:])

        if self.epoch < FLAGS.aug_start_epoch:
          index = 0
      
      # if idx % 100 == 0:
      #   ic(idx, index, self.epoch)
      
      inputs = self.inputs_list[index]
      x, y = self.getitem(inputs, idx)
    else:
      x = {}
      for i, inputs in enumerate(self.inputs_list):
        x[i], _ = self.getitem(inputs, idx)
    
      masks = []
      start_masks = []
      labels = []
      starts = []
      re_positions = []
      for i, x_ in x.items():
        if isinstance(i, int):
          masks.append(x_['mask'])
          start_masks.append(x_['start_mask'])
          labels.append(x_['label'])
          starts.append(x_['start'])
          # re_positions.append(x_['word_relative_positions'])
      x['word_ids'] = np.asarray(list(range(x_['num_words'])) + [util.null_wordid()] * (FLAGS.max_words - x_['num_words']))
      visited = np.stack(masks, 0).sum(0)
      x['mask'] = np.clip(visited, None, 1)
      x['start_mask'] = np.clip(np.stack(start_masks, 0).sum(0), None, 1)
      x['start'] = np.clip(np.stack(starts, 0).sum(0), None, 1)
      x['sep'] = x['start']
      counts = np.clip(visited, 1, None)
      x['word_counts'] = counts
      label = np.stack(labels, 0).sum(0)
      x['label'] = (label / counts).astype(np.long)
      # relative_positions = np.stack(re_positions, 0).sum(0)
      # x['word_relative_positions'] =(relative_positions / counts).astype(np.float32)
      y = x['label']
      
      for key in x_:
        if key not in x:
          x[key] = x_[key]
   
    return x, y
    
  def __len__(self):
    return len(self.inputs['y'])
  
  def get_labels(self):
    return self.labels

