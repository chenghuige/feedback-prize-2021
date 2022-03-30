#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   metrics.py
#        \author   chenghuige  
#          \date   2022-02-03 04:30:03.195910
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
import pandas as pd
from src.config import *

def get_start_end(span):
  if isinstance(span, str):
    l = span.split()
    s, e = int(l[0]), int(l[-1])
    e += 1
  else:
    s, e = span
  return s, e

def to_dict(types, spans):
  m = {}
  for cls in CLASSES:
    m[cls] = []
  for type, span in zip(types, spans):
    if type == 'Nothing':
      continue
    s, e = get_start_end(span)
    m[type].append([s, e])
  return m

def calc_f1(m, return_dict=True):
  f1_scores = []
  res = {}
  for c in CLASSES:
    TP = m[c]['match']
    FP = m[c]['pred'] - TP
    FN = m[c]['gt'] - TP
    if m[c]['gt'] == 0 and m[c]['pred'] == 0:
      continue
    else:
      f1_score = TP / (TP + 0.5 * (FP + FN))
      if return_dict:
        res[c] = f1_score
    f1_scores.append(f1_score)
  f1_score = np.mean(f1_scores)
  if not return_dict:
    return f1_score
  else:
    res['Overall'] = f1_score
    return res

def is_match(gt, pred):
  s = min(gt[1], pred[1]) - max(gt[0], pred[0]) 
  intersect = max(0, s)
  return intersect / (gt[1] - gt[0]) >= 0.5 and intersect / (pred[1] - pred[0]) >= 0.5

def calc_match(gts, preds):
  matches = 0
  matched = set()
  for gt in gts:
    score = 0
    best_pred = None
    for pred in preds:
      if is_match(gt, pred):
        if tuple(pred) in matched:
          continue
        # assert not tuple(pred) in matched
        # if tuple(pred) in matched:
        #   ic(gts, preds)
        #   exit(0)
        score_ = match_score(gt, pred)
        if score_ > score:
          score = score_
          best_pred = pred
        if score == 1:
          break
    if score:
      matched.add(tuple(best_pred))
      matches += 1
    
  return matches

def prepare_f1(gt, pred):
  m = {}
  for c in CLASSES:
    m[c] = {
      'match': calc_match(gt[c], pred[c]),
      'gt': len(gt[c]),
      'pred': len(pred[c])
    }
  return m

#inputs are two dicts
#  m1 = to_dict(types1, spans1)
#  m2 = to_dict(types2, spans2)
# res = essay_f1(m1, m2)
def essay_f1(gt, pred, return_dict=True):
  m = prepare_f1(gt, pred)
  res = calc_f1(m, return_dict=return_dict)
  return res

