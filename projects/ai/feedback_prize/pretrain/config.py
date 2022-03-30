#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2021-07-31 09:00:06.027197
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

import gezi
from gezi import logging
import melt as mt
from src import config
 
def init():
  config.init()
  FLAGS.model_dir = f'{FLAGS.idir}/pretrain'
  FLAGS.mn = 'pretrain'
  FLAGS.ep = 10
  if 'large' in FLAGS.backbone:
    FLAGS.bs = int(FLAGS.bs / 2)
