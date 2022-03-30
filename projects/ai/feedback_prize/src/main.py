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

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

def main(_):
  timer = gezi.Timer()
  fit = mt.fit  
     
  config.init()
  mt.init()
  
  os.system(f'cp ./*.py {FLAGS.model_dir}')
  os.system(f'cp -rf ./torch {FLAGS.model_dir}')
  dataset_meta_root = '..'
  # dataset_name = f'{FLAGS.hug}-{FLAGS.sm}-{int(FLAGS.mui)}'
  # content = open(f'{dataset_meta_root}/dataset-metadata.json').readline().strip()
  # content = content.replace('feedback-model', dataset_name)
  # with open(f'{FLAGS.model_dir}/dataset-metadata.json', 'w') as f:
  #   print(content, file=f)
  os.system(f'cp {dataset_meta_root}/dataset-metadata.json {FLAGS.model_dir}')      
  
  if not FLAGS.torch:
    from src.tf.model import Model
  else:
    from src.torch.model import Model
    
  strategy = mt.distributed.get_strategy()
  with strategy.scope():    
    model = Model()
    model.eval_keys = ['id', 'label', 'mask', 'start', 'end', 'sep', 'dis_start', 'dis_end', 
                      'para_count', 'para_type', 'para_mask', 'para_index', 
                      'input_ids', 'attention_mask', 'word_ids', 'num_words']
    model.str_keys = ['id']
    model.out_keys = ['start_logits', 'parts', 'para_logits', 'end_logits', 'cls_logits'] + [f'{cls_}_logits' for cls_ in classes]
    
    if FLAGS.torch:     
      optimizer_params = lele.get_optimizer_params(model, FLAGS.backbone_lr, FLAGS.base_lr, FLAGS.weight_decay)
      gezi.set('lr_params', optimizer_params)
      gezi.set('rdrop_loss_fn', model.calc_rdrop_loss)
    
    if not FLAGS.torch_only:
      from src.dataset import Dataset
      fit(model,  
          Dataset=Dataset,
          eval_fn=ev.evaluate
          ) 
    else:    
      train_dl, eval_dl, valid_dl = util.get_dataloaders()
          
      fit(model,  
          dataset=train_dl,
          eval_dataset=eval_dl,
          valid_dataset=valid_dl,
          gen_dataset_fn=util.get_dataset if FLAGS.dataset_per_epoch else None,
          eval_fn=ev.evaluate
          ) 
  
  # if FLAGS.online or (FLAGS.fold == 0 and gezi.get('eval_metric', 0) > FLAGS.min_save_score):
  if FLAGS.online:
    if not FLAGS.torch:
      model.build_model().save_weights(f'{FLAGS.model_dir}/model2.h5')  

    #   swa_model = gezi.get('swa_model')
    #   if swa_model is not None:
    #     # loader = gezi.get('info')['dataset']
    #     # torch.optim.swa_utils.update_bn(loader, swa_model)
    #     model = swa_model 
    #   state = {
    #             'state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
    #           }
    #   torch.save(state, f'{FLAGS.model_dir}/model2.pt')

  if FLAGS.save_final:
    gezi.save_model(model, FLAGS.model_dir, fp16=FLAGS.save_fp16)
  
  gezi.folds_metrics_summary(FLAGS.model_dir, FLAGS.folds)
    
 
if __name__ == '__main__':
  app.run(main)  
