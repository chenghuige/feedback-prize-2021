#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
#        \author   chenghuige  
#          \date   2021-07-31 08:49:47.444181
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
import numpy as np
from icecream import ic

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

from tensorflow import keras

import gezi
logging = gezi.logging
import melt as mt
from pretrain import config
from pretrain.dataset import Dataset

def main(_):
  timer = gezi.Timer()
  fit = mt.fit 
  config.init()
  mt.init()

  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    model = mt.pretrain.bert.Model(FLAGS.backbone)

    ic(model.bert.layers)
    fit(model,  
        loss_fn=model.get_loss(),
        Dataset=Dataset,
        metrics=['accuracy']
       ) 

  model.bert.save_weights(f'{FLAGS.model_dir}/bert.h5')
  model.bert.save_pretrained(f'{FLAGS.model_dir}/bert')

if __name__ == '__main__':
  app.run(main)   
