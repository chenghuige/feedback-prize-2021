#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2021-01-09 17:51:11.308942
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app, flags
FLAGS = flags.FLAGS

import tensorflow as tf
from tensorflow.keras import backend as K
import melt as mt
from src import util
from src.config import *
from src.util import *

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    keys = []
    excl_keys = ['word_ids_str'] 
    # dynamic_keys = ['token_logits', 'start_logits'] 
    dynamic_keys = []
    self.auto_parse(keys=keys, exclude_keys=excl_keys + dynamic_keys)
    self.adds(dynamic_keys)
    fe = self.parse_(serialized=example)
          
    mt.try_append_dim(fe)
    x = fe
    # y = tf.minimum(fe['score'], 1.)
    y = fe['start']
     
    return x, y
  
