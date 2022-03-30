#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rl.py
#        \author   chenghuige  
#          \date   2022-01-20 13:50:19.909762
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import torch
from src.get_preds import P

def greedy_decode(start_logits, token_logits):
  seps = (torch.softmax(start_logits)[:,:,1] > P['sep_prob'] ).long()
  # x = torch.cumsum(seps, 1)
  # ic(seps, x)
  return seps

def sample_decode(start_logits, token_logits):
  h, w = start_logits.shape[0], start_logits.shape[1]
  start_probs = torch.softmax(start_logits.view(-1, 2),dim=1)
  seps = torch.multinomial(start_probs, 1).view(h, w).long()
  # x = torch.cumsum(seps, 1)
  # ic(seps, x)
  return seps
