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

import gezi
from gezi import tqdm
logging = gezi.logging
import melt as mt
import lele

import src
import src.eval as ev
from src import config
from src.config import *
from src import util
from l2.dataset import Dataset
from l2.model import Model

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

def eval(y_true, y_pred, x):
  res = {'score': 0}
  # return res
  y_pred = gezi.sigmoid(y_pred)
  # auc = gezi.metrics.group_auc(y_true, y_pred, x['id'], weighted=False)
  # res['Metrics/auc'] = auc
  # base_auc = gezi.metrics.group_auc(y_true, (x['model'] == 1).astype(float), x['id'], weighted=False)
  # res['Metrics/base_auc'] = base_auc
  # base_auc2 = gezi.metrics.group_auc(y_true, (x['model'] == 17).astype(float), x['id'], weighted=False)
  # res['Metrics/base_auc2'] = base_auc2
  m = {
    'id': x['id'],
    # 'model': x['model'],
    # 'fold': x['fold'],
    # 'num_gts': x['num_gts'],
    # 'num_words': x['num_words'],
    'gt': y_true,
    'pred': y_pred
  }
  # df = pd.DataFrame(m)
  # df.to_csv(f'{FLAGS.model_dir}/valid.csv')
  # ic(df)
  gezi.save(m, f'{FLAGS.model_dir}/start.pkl')
  return res

def main(_):
  timer = gezi.Timer()
  fit = mt.fit  
  
  FLAGS.encode_cls = True
  FLAGS.torch = True
  bs = 16
  FLAGS.bs = FLAGS.bs or bs
  lr = 1e-4
  FLAGS.lr = FLAGS.lr or lr
  FLAGS.opt = 'adamw'
  ep = 1
  FLAGS.ep = FLAGS.ep or ep
  # FLAGS.nvs = FLAGS.ep
  # FLAGS.vie = 0.1
  FLAGS.vie = 1
  FLAGS.num_classes = len(ALL_CLASSES)
  FLAGS.run_version = FLAGS.run_version or RUN_VERSION
  FLAGS.folds = 5
  FLAGS.fold = FLAGS.fold or 0
  FLAGS.run_version += f'/{FLAGS.fold}'
  FLAGS.wandb = False
  FLAGS.write_summary = False
  gpus = 1
  FLAGS.gpus = FLAGS.gpus or gpus
  
  pres = ['offline', 'online']
  pre = pres[int(FLAGS.online)]
  model_dir = f'../working/{pre}/{FLAGS.run_version}/model'  
  FLAGS.model_dir = FLAGS.model_dir or model_dir
  if FLAGS.mn == 'model':
    FLAGS.mn = 'l2'
  ic(FLAGS.model_dir, FLAGS.mn)
  records_pattern = f'{FLAGS.idir}/tfrecords.l2/train/*.tfrec'
  files = gezi.list_files(records_pattern) 
  ic(records_pattern)
  FLAGS.train_files = files
  if FLAGS.online:
    FLAGS.allnew = True

  FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]
  if not FLAGS.online:
    FLAGS.train_files = [x for x in FLAGS.train_files if x not in FLAGS.valid_files]
  
  FLAGS.start_loss = True
  FLAGS.start_loss_rate = 10.
  
  mt.init()
  
  os.system(f'cp ./*.py {FLAGS.model_dir}')
  dataset_meta_root = '..'
  os.system(f'cp {dataset_meta_root}/dataset-metadata.json {FLAGS.model_dir}')      
      
  strategy = mt.distributed.get_strategy()
  with strategy.scope():    
    model = Model()
    model.eval_keys = ['id', 'model', 'fold', 'num_gts', 'num_words']
    model.str_keys = ['id']
    model.out_keys = ['start_logits', 'parts', 'para_logits', 'end_logits', 'cls_logits'] + [f'{cls_}_logits' for cls_ in classes]
    
    fit(model,  
        Dataset=Dataset,
        eval_fn=eval
        ) 
    
  if FLAGS.save_final:
    gezi.save_model(model, FLAGS.model_dir, fp16=FLAGS.save_fp16)
    
 
if __name__ == '__main__':
  app.run(main)  
