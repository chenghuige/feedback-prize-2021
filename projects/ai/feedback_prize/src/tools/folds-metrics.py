#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   folds-metrics.py
#        \author   chenghuige  
#          \date   2022-03-06 11:29:33.169801
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import glob
import gezi

v = sys.argv[1]

files = glob.glob(f'../working/offline/{v}/0/*/command.txt')
commands = [open(x).readline().strip() for x in files]
commands = [x + ' --folds_metrics' for x in commands]
for command in commands:
  gezi.system(command)

