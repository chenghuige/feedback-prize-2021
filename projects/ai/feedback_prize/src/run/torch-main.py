#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:02.802049
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from genericpath import exists

import sys
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'WARNING'

import glob
import pandas as pd
from tensorflow import keras
import torch
from torch.utils.data import DataLoader

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

import gezi
from gezi import tqdm
logging = gezi.logging
import melt as mt

import src
from src.torch.dataset import Dataset
import src.eval as ev
from src import config
from src.config import *
from src import util
from src.torch.model import Model

def main(_):
  timer = gezi.Timer()
  
  FLAGS.torch_only = True
  fit = mt.fit  
     
  config.init()
  mt.init()

  model = Model()
  model.eval_keys = ['id', 'label', 'mask', 'start', 'end', 'sep', 'dis_start', 'dis_end', 
                      'para_count', 'para_type', 'para_mask', 'para_index', 
                      'input_ids', 'attention_mask', 'word_ids', 'num_words']
  model.str_keys = ['id']
  model.out_keys = ['start_logits', 'parts', 'para_logits', 'end_logits', 'cls_logits'] + [f'{cls_}_logits' for cls_ in classes]
  
  # if FLAGS.torch:
  #   if FLAGS.backbone_lr:
  #     backbone_params = model.backbone.parameters()
  #     ignored_params = list(map(id, backbone_params))
  #     # 注意这时候backbone_params就是空的了
  #     base_params = filter(lambda p: id(p) not in ignored_params,
  #                         model.parameters())
  #     param_groups = [
  #           {'params': base_params, 'lr': FLAGS.base_lr},
  #           {'params': model.backbone.parameters(), 'lr': FLAGS.backbone_lr}
  #         ]
  #     gezi.set('lr_params', param_groups)
      
  train_ds = Dataset(FLAGS.train_files, 'train')
  valid_ds = Dataset(FLAGS.valid_files, 'valid')
  valid_ds2 = Dataset(FLAGS.valid_files, 'valid')
  num_workers = 4
  kwargs = {'num_workers': num_workers, 'pin_memory': True}  
  train_dl = DataLoader(train_ds, mt.batch_size(), shuffle=True, **kwargs)

  valid_dl = DataLoader(valid_ds, mt.eval_batch_size(), shuffle=False, **kwargs)
  valid_dl2 = DataLoader(valid_ds2, mt.batch_size(), shuffle=False, **kwargs)
      
  fit(model,  
      dataset=train_dl,
      eval_dataset=valid_dl,
      valid_dataset=valid_dl2,
      eval_fn=ev.evaluate
      ) 


  dataset_meta_root = '..'
  os.system(f'cp {dataset_meta_root}/dataset-metadata.json {FLAGS.model_dir}')    
 
if __name__ == '__main__':
  app.run(main)  
