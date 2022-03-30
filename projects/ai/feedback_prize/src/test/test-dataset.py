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
from datasets import Dataset
import torch

gezi.init_flags()
files = gezi.list_files('../input/feedback-prize-2021/tfrecords.electra.len512/train/*.tfrec')
ic(files[:2])

bs = 512
ds = MyDataset()
# dl = ds.make_batch(bs, filenames=files, return_numpy=True)
dl = ds.make_batch(bs, filenames=files)
ic(len(ds), ds.num_steps)

inputs = {}
for x, y in tqdm(dl, total=ds.num_steps):
  if not inputs:
    inputs = {k: list(v) for k, v in x.items()}
    inputs['y'] = list(y)
  else:
    for key in x:
      inputs[key].extend(list(x[key]))
    inputs['y'].extend(list(y))
      
# # TODO dict 如果有string似乎不行
# dataset = Dataset.from_dict(inputs)
# device = lele.get_device()
# ic(device)
# dataset.set_format(type='torch', device=device)
# ic(dataset[0])
# ic(dataset[-1])
# dl = torch.utils.data.DataLoader(dataset, batch_size=bs)
# for x in tqdm(dl):
#   ic(x)
#   break


# class MyTorchDataset(TorchDataset):
#   def __init__(self, inputs):
#     self.inputs = inputs

#   def __getitem__(self, idx):
#     x = {k: v[idx] for k, v in self.inputs.items()}
#     y = self.inputs['y'][idx]
#     return x, y
    
#   def __len__(self):
#     return len(self.inputs)

# dataset = MyTorchDataset(inputs)
# dl = torch.utils.data.DataLoader(dataset, batch_size=bs)
# for x in tqdm(dl):
#   ic(x)
#   break
