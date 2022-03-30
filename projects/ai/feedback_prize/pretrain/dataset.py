#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2021-07-31 08:49:52.078016
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from pandas.io import feather_format
from icecream import ic
import tensorflow as tf

from gezi import tqdm
from .config import *

class Dataset(mt.Dataset):
  def __init__(self, subset='train', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    keys, excl_keys = [], []
    self.auto_parse(keys=keys, exclude_keys=excl_keys)
    fe = self.parse_(serialized=example)

    mt.try_append_dim(fe)
    x = fe
    y = x['input_ids']
    return x, y
