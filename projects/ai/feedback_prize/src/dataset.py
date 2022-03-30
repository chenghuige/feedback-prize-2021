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
    dynamic_keys = [] 
    self.auto_parse(keys=keys, exclude_keys=excl_keys + dynamic_keys)
    self.adds(dynamic_keys)
    fe = self.parse_(serialized=example)
          
    mt.try_append_dim(fe)
    
    if FLAGS.merge_tokens:
      fe['start'] = fe['word_start']
      fe['start2'] = fe['word_start']
      fe['end'] = fe['word_end']
      fe['end2'] = fe['word_end']
      fe['mask'] = fe['word_mask']
      fe['start_mask'] = fe['word_start_mask']
      fe['label'] = fe['word_label']
      
    # start2 会确保整体数目和para数目完全一致，而start则可能因为一个word如果选择都标注start 而多于para数目
    fe['sep'] = fe['start2']
    if FLAGS.pred_method == 'end':
      fe['sep'] = fe['end2']
          
    fe['parts_count'] = tf.reduce_sum(fe['sep'], -1)
    # Nothing_B, Nothing_I, Claim_B, Claim_I ... 8 -> 16 分类
    if FLAGS.method == 1:
      # 0 -> 0(I), 1(B)  1 -> 1(I), 2(B) 
      fe['label'] = fe['label'] * 2 + fe['start']
    else:
      if FLAGS.method == 3:
        fe['start'] = fe['dis_start']
        fe['end'] = fe['dis_end']

    # if not FLAGS.soft_start:
    #   fe['start'] = tf.cast(fe['start'] == 1, tf.int32)
    # else:
    #   fe['start'] = tf.stack([1. - fe['start'], fe['start']], -1)
      
    x = fe
    y = fe['label']
     
    return x, y
  
