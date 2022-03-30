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

import gezi
from gezi import tqdm
logging = gezi.logging
import melt as mt

import src
from src.dataset import Dataset
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
  
  if not FLAGS.torch:
    from src.model import Model
  else:
    from src.torch_model import Model
  
  strategy = mt.distributed.get_strategy()
  with strategy.scope():    
    model = Model()
    model.eval_keys = ['id', 'label', 'mask', 'start', 'end', 'sep', 'dis_start', 'dis_end', 
                       'para_count', 'para_type', 'para_mask', 'para_index', 
                       'input_ids', 'attention_mask', 'word_ids', 'num_words']
    model.str_keys = ['id']
    model.out_keys = ['start_logits', 'parts', 'para_logits', 'end_logits', 'cls_logits'] + [f'{cls_}_logits' for cls_ in classes]
    
    if FLAGS.torch:
      if FLAGS.backbone_lr:
        backbone_params = model.backbone.parameters()
        ignored_params = list(map(id, backbone_params))
        # 注意这时候backbone_params就是空的了
        base_params = filter(lambda p: id(p) not in ignored_params,
                            model.parameters())
        param_groups = [
              {'params': base_params, 'lr': FLAGS.base_lr},
              {'params': model.backbone.parameters(), 'lr': FLAGS.backbone_lr}
            ]
        gezi.set('lr_params', param_groups)
    
    fit(model,  
        Dataset=Dataset,
        eval_fn=ev.evaluate
        ) 
  
  # if FLAGS.online or (FLAGS.fold == 0 and gezi.get('eval_metric', 0) > FLAGS.min_save_score):
  if FLAGS.online:
    if not FLAGS.torch:
      model.build_model().save_weights(f'{FLAGS.model_dir}/model2.h5')  
    # else:
    #   swa_model = gezi.get('swa_model')
    #   if swa_model is not None:
    #     # loader = gezi.get('info')['dataset']
    #     # torch.optim.swa_utils.update_bn(loader, swa_model)
    #     model = swa_model 
    #   state = {
    #             'state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
    #           }
    #   torch.save(state, f'{FLAGS.model_dir}/model2.pt')
    
  dataset_meta_root = '..'
  os.system(f'cp {dataset_meta_root}/dataset-metadata.json {FLAGS.model_dir}')    
 
if __name__ == '__main__':
  app.run(main)  
