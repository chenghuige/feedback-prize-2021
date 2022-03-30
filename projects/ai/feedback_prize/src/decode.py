#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   decode.py
#        \author   chenghuige  
#          \date   2022-02-02 07:11:29.737689
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import numpy as np
import pandas as pd
from src.get_preds import *

def get_start_end(span):
  if isinstance(span, str):
    l = span.split()
    s, e = int(l[0]), int(l[-1])
    e += 1
  else:
    s, e = span
  return s, e

def calc_f1(m, return_dict=True):
  f1_scores = []
  res = {}
  for c in CLASSES:
    assert m[c]['match'] <= m[c]['pred']
    assert m[c]['match'] <= m[c]['gt']
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

def calc_intersect(gt, pred):
  s = min(gt[1], pred[1]) - max(gt[0], pred[0]) 
  return max(0, s)

def match_score(gt, pred):
  intersect = calc_intersect(gt, pred)
  return intersect / ((gt[1] - gt[0]) + (pred[1] - pred[0]) - intersect)

def best_match(gts, pred):
  best_score = 0
  for gt in gts:
    score = match_score(gt, pred)
    if score > best_score:
      best_score = score
    if best_score == 1:
      break
  return best_score

#  gts: [[18, 34], [34, 36], [36, 38], [38, 40], [40, 46], [242, 253]]
#     preds: [[18, 34],
#             [34, 36],
#             [36, 40],
#             [41, 46],
#             [46, 64],
#             [101, 120],
#             [173, 180],
#             [236, 253],
#             [298, 332]]
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

def essay_f1(gt, pred, return_dict=False):
  m = prepare_f1(gt, pred)
  res = calc_f1(m, return_dict=return_dict)
  return res

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

def list_to_dict(res_list):
  m = {}
  for cls in CLASSES:
    m[cls] = []
  for res in res_list:
    if res['cls'] == 0:
      continue
    m[id2dis[res['cls']]].append([res['start'], res['end']])
  return m

def get_gt_dict(label, id):
  # label = pd.read_feather('../input/feedback-prize-2021/train_en.fea')
  d = label[label.id == id]
  types = [id2dis[x] for x in d.para_type.values]
  spans = list(zip(d.start.values, d.end.values))
  gt_dict = to_dict(types, spans)
  return gt_dict
 
def decode(x):
  pred = x['preds']
  total = len(pred)
  probs = x['probs'] 
  start_prob = x['start_probs'] 
  pre_type = None
  preds_list = []
  preds = [] 
  pre_scores = np.zeros_like(probs[0])
  
  types = []
  res_list = []
  pre_probs = None
  pre_is_sep = False
  for i in range(total):    
   
    is_sep = False
    sep_prob = start_prob[i][1]
    if i > 0:
      pre_cls = np.argmax(pre_probs)
      pre_prob = pre_probs[pre_cls]
      pre_type = id2dis[pre_cls]
      last_cls = pred[i - 1]
      last_type = id2dis[last_cls]
      now_cls = pred[i]
      now_type = id2dis[now_cls]
      now_prob = probs[i][now_cls]
  
      if now_cls != last_cls:  
        if pre_type == 'Lead':
          if sep_prob > P['sep_prob_Lead']:
            is_sep = True    
        elif pre_type == 'Position':
          if sep_prob > P['sep_prob_Position']:
            is_sep = True      
        elif pre_type == 'Rebuttal':
          if sep_prob > P['sep_prob_Rebuttal']:
            is_sep = True     
        elif pre_type == 'Claim':
          if sep_prob > P['sep_prob_Claim']:
            is_sep = True     
        elif pre_type == 'Counterclaim':
          if sep_prob > P['sep_prob_Counterclaim']:
            is_sep = True    
        elif pre_type == 'Nothing':
          if sep_prob > P['sep_prob_Nothing']:
            is_sep = True    
        elif pre_type == 'Evidence':
          if sep_prob > P['sep_prob_Evidence']:
            is_sep = True   
        else:
          if sep_prob > P['sep_prob']:
            is_sep = True
        
        if pre_type == 'Lead':
          if pre_probs[last_cls] > P['pre_prob_Lead']:
            is_sep = True
        elif pre_type == 'Position':
          if pre_probs[last_cls] > P['pre_prob_Position']:
            is_sep = True
        elif pre_type == 'Rebuttal':
          if pre_probs[last_cls] > P['pre_prob_Rebuttal']:
            is_sep = True
        elif pre_type == 'Claim':
          if pre_probs[last_cls] > P['pre_prob_Claim']:
            is_sep = True
        elif pre_type == 'Counterclaim':
          if pre_probs[last_cls] > P['pre_prob_Counterclaim']:
            is_sep = True
        elif pre_type == 'Evidence':
          if pre_probs[last_cls] > P['pre_prob_Evidence']:
            is_sep = True
        elif pre_type == 'Nothing':
          if pre_probs[last_cls] > P['pre_prob_Nothing']:
            is_sep = True
        else:      
          if pre_probs[last_cls] > P['pre_prob']:
            is_sep = True
        
        if pre_prob == 0:
          is_sep = True
      else:
        if pre_type == 'Claim':
          is_sep = sep_prob >= P['sep_eq_prob_Claim']
        elif pre_type == 'Evidence':
          is_sep = sep_prob >= P['sep_eq_prob_Evidence']
        # elif pre_type == 'Lead':
        #   is_sep = sep_prob >= P['sep_eq_prob_Lead']
        # elif pre_type == 'Position':
        #   is_sep = sep_prob >= P['sep_eq_prob_Position']
        # elif pre_type == 'Counterclaim':
        #   is_sep = sep_prob >= P['sep_eq_prob_Counterclaim']
        # elif pre_type == 'Rebuttal':
        #   is_sep = sep_prob >= P['sep_eq_prob_Rebuttal']      
        # elif pre_type == 'Nothing':
        #   is_sep = sep_prob >= P['sep_eq_prob_Nothing']
        else:
          is_sep = sep_prob > P['sep_eq_prob']

    if is_sep:
      
      if preds:  
        res = {
              'cls': pre_cls,
              'len': len(preds),
              'start': i - len(preds),
              'end': i,
              'token_prob': pre_scores,
              'sep_prob': start_prob[i][1],
              'sep_logits': x['start_logits'][i],
              'token_logits': x['pred'][i - len(preds):i].mean(0),
            }
        if pre_type != 'Nothing':
          if pre_probs.max() > proba_thresh[pre_type]:
            res_list.append(res)
          else:
            res['cls'] = 0
            res_list.append(res)
        else:
          res_list.append(res)
    
        preds = []
        pre_scores = np.zeros_like(probs[0])
              
    pre_scores += probs[i] 
    pre_probs = gezi.softmax(pre_scores)
    preds.append(str(i))
    
  i = total
  if preds:
    pre_cls = np.argmax(pre_scores)
    pre_type = id2dis[pre_cls]
    res = {
              'cls': pre_cls,
              'len': len(preds),
              'start': i - len(preds),
              'end': i,
              'token_prob': pre_scores,
              'sep_prob': 1.,
              'sep_logits': np.asarray([0., 1.]),
              'token_logits': x['pred'][i - len(preds):i].mean(0),
            }
    if pre_type != 'Nothing':
      if pre_probs.max() > proba_thresh[pre_type]:
        res_list.append(res)
      else:
        res['cls'] = 0
        res_list.append(res)
    else:
      res_list.append(res)

  return res_list

def decodes_(x, fold=None, folds=5):  
  total = len(x['id'])
  res = {}
  xs = gezi.batch2list(x)
  ids_list, types_list, preds_list = [], [], []
  for i in tqdm(range(len(xs)), desc='get_preds', leave=False):
    x = xs[i]
    id = x['id']
    if fold is not None:
      if i % folds != fold:
        continue
    vote = decode(x)
    res[id] = vote  
  return res

def decodes(x, folds=5):  
  if FLAGS.pymp:
    try:
      dfs = Manager().dict()
      with pymp.Parallel(folds) as p:
        for i in p.range(folds):
          dfs[i] = decodes_(x, fold=i, folds=folds)
      res = gezi.merge_dict_values(dfs)
      return res
    except Exception as e:
      ic(e)
      FLAGS.pymp = False
  if not FLAGS.pymp:
    res = decodes_(x)
    return res
  