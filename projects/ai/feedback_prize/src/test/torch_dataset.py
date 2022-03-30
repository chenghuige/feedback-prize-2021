#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   test-dataset.py
#        \author   chenghuige  
#          \date   2022-01-14 23:57:13.373154
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from curses import def_shell_mode

import sys 
sys.path.append('..')
import os
import gezi
from gezi import tqdm
import melt as mt
import lele
from src.dataset import Dataset as MyDataset
from torch.utils.data import Dataset as TorchDataset
import torch

class Dataset(TorchDataset):
  def __init__(self, files):
    inputs = {}
    
    bs = 512
    ds = MyDataset()
    dl = ds.make_batch(bs, filenames=files)
    ic(len(ds), ds.num_steps)

    for x, y in tqdm(dl, total=ds.num_steps, desc=files[0], leave=False):
      if not inputs:
        inputs = {k: list(v) for k, v in x.items()}
        inputs['y'] = list(y)
      else:
        for key in x:
          inputs[key].extend(list(x[key]))
        inputs['y'].extend(list(y))    
    
    self.inputs = inputs

  def __getitem__(self, idx):
    # ignores = ['id']
    x = {k: torch.as_tensor(v[idx]) for k, v in self.inputs.items()}
    y = torch.as_tensor(self.inputs['y'][idx])
    return x, y
    
  def __len__(self):
    return len(self.inputs['y'])

