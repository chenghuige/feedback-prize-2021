#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   get_preds.py
#        \author   chenghuige  
#          \date   2022-02-07 07:26:10.410197
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import S

import sys 
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pymp 
from multiprocessing import Pool, Manager, cpu_count
import gezi
from gezi import tqdm, logging
from src.config import *

min_thresh = {
    "Lead": 9,
    "Position": 5,
    "Evidence": 14,
    "Claim": 3,
    "Concluding Statement": 11,
    "Counterclaim": 6,
    "Rebuttal": 4,
    "Nothing": 1,
}

proba_thresh = {
    "Nothing": 0.5,
    "Lead": 0.99,
    "Position": 0.6478988761165504,
    "Evidence": 0.99,
    "Claim": 0.3655395042560211,
    "Concluding Statement": 0.99,
    "Counterclaim": 0.5,
    "Rebuttal": 0.55,  
}

# P =  {
#       'pre_prob': 0.9970760068793366,
#       'sep_prob': 0.6808957083325543,
#       'sep_prob2': 0.43956461588178136,
#       }
# P = {'pre_prob': 0.9994445358985654, 'sep_prob': 0.677696887096806, 'sep_prob2': 0.34290279696674325}
# 'sep_prob': 0.6108554732371797, 'sep_prob3': 0.5865994197464139
#  {'sep_prob': 0.5974120144164955, 'sep_prob3': 0.5883697552286733, 'sep_prob4': 0.9233073692544926}
# 'sep_prob': 0.5777702605120832, 'sep_prob3': 0.6102685909267511, 'sep_prob4': 0.9214272278994625
P = {
  'sep_eq_prob_Claim': 0.5357903723123728,
  'sep_eq_prob_Evidence':  0.5834419868375141,  
  'sep_eq_prob':  0.9714243800020556,
  
  'sep_prob': 0.3451363769175534,
  'sep_prob_Lead':  0.4346759767173439,
  'sep_prob_Position':  0.3628750668870852,
  'sep_prob_Rebuttal': 0.3451363769175534,
  'sep_prob_Claim':  0.3995825234294613,
  'sep_prob_Evidence':  0.4193187245819171,
  'sep_prob_Counterclaim': 0.3524418653217463,
  'sep_prob_Nothing': 0.3451363769175534,
  
  'pre_prob':  0.9992753247380775,
  'pre_prob_Position':  0.9662340310006595,
  'pre_prob_Lead':  0.5285479463891315,
  'pre_prob_Claim':  0.9992753247380775,
  'pre_prob_Counterclaim':  0.9992753247380775,
  'pre_prob_Evidence':  0.9992753247380775,
  'pre_prob_Rebuttal':  0.9992753247380775,
  'pre_prob_Nothing':  0.9992753247380775,
}
# P = {
#   # 'sep_prob': 0.6397724115245825,
#   'sep_eq_prob_Claim': 0.5357903723123728,
#   'sep_eq_prob_Evidence':  0.5834419868375141,  
#   'sep_eq_prob_Lead':  0.9714243800020556,
#   'sep_eq_prob_Position':  0.9170439409181261,
#   'sep_eq_prob_Rebuttal':   0.6474809505077193,
#   'sep_eq_prob_Counterclaim': 0.8406116943276105,
#   'sep_eq_prob_Nothing':  0.8439459369690111,
#   'sep_eq_prob':  0.9213809053492297,
  
#   'sep_prob': 0.05083744937841328,
#   'sep_prob_Lead':  0.4346759767173439,
#   'sep_prob_Position':  0.3628750668870852,
#   'sep_prob_Rebuttal': 0.3451363769175534,
#   'sep_prob_Claim':  0.3995825234294613,
#   'sep_prob_Evidence':  0.4193187245819171,
#   'sep_prob_Counterclaim': 0.3524418653217463,
#   'sep_prob_Nothing': 0.3451363769175534,
  
#   'pre_prob':  0.9998627078512725,
#   'pre_prob_Position':  0.9662340310006595,
#   'pre_prob_Lead':  0.5398793973974411,
#   'pre_prob_Claim':  0.9895083515125076,
#   'pre_prob_Counterclaim':  0.9130269198804363,
#   'pre_prob_Evidence': 0.9999326871628939,
#   'pre_prob_Rebuttal':  0.9992102755913583,
#   'pre_prob_Nothing':  0.9807534120804502,
# }
def get_pred_reverse(x, post_adjust=True, pred_info=None, return_score=False):
  P = {'pre_prob': 0.8617871977191984, 'sep_prob': 0.6128679723792144, 'sep_prob2': 0.4774318242353257}
  pred = list(x['preds'])[::-1]
  total = len(pred)
  # by prob not logit
  probs = list(x['probs'])[::-1]
  # probs = x['pred']
  start_prob = list(x['start_probs'])[::-1]
  pre_type = None
  # predictionstring list
  preds_list = []
  # store each pred word_id for one precitionstring
  preds = [] 
  pre_scores = np.zeros_like(probs[0])
  
  types = []
  scores = []
  pre_probs = None
  for i in range(total):    
    pre_scores += probs[i] 
    pre_probs = gezi.softmax(pre_scores)
    # pre_probs = pre_scores / len(preds)
    preds.append(str(total - 1 - i))
    
    is_sep = False
    if np.absolute(pre_scores).sum() == 0:
      pre_cls = 0
    else:
      pre_cls = np.argmax(pre_scores)
    pre_prob = pre_probs[pre_cls]
    pre_type = id2dis[pre_cls]
    now_cls = pred[i]
    now_type = id2dis[now_cls]
    now_prob = probs[i][now_cls]
    sep_prob = start_prob[i][1]
       
    if i < total - 1:
      is_sep = sep_prob >= P['sep_prob']
      if pred[i] != pred[i + 1]:              
        if sep_prob > P['sep_prob2']:
          is_sep = True
                            
        if pre_probs[pred[i + 1]] > P['pre_prob']:
          is_sep = True
        # if pre_probs[pred[i]] > proba_thresh[pred[i]]: 
        #   is_sep = True
    if i == total - 1:
      is_sep = True
      sep_prob = 1.
    if is_sep:
      if preds:  
        if pre_type != 'Nothing':
          if pre_probs.max() > proba_thresh[pre_type]:
            # if not (pre_type == 'Rebuttal' and not 'Counterclaim' in types):
            preds_list.append(' '.join(preds[::-1]))
            types.append(pre_type)
            scores.append((pre_probs.max() + sep_prob) * len(preds)) 
          else:
            scores.append((pre_probs[0] + sep_prob) * len(preds))
        else:
          scores.append((pre_probs[0] + sep_prob) * len(preds))
          
        preds = []
        pre_scores = np.zeros_like(probs[0])
  
  scores = scores or [0]
  score = np.sum(scores) / total
  
  if return_score:
    return types, preds_list, score
  else:
    return types, preds_list
              
# start_info = gezi.load('../working/offline/60/0/l2/start.pkl')
# starts = {}
# starts_info = gezi.batch2list(start_info)
# for info in starts_info:
#   starts[info['id']] = info['gt']
    
def get_pred_bystart(x, pred_info=None, return_score=False):
  # gezi.save(x, '../working/d2.pkl')  
  pred = x['preds']
  total = len(x['pred'])
  # by prob not logit
  probs = x['probs'] 
  # probs = x['pred']
  start_prob = x['start_probs'] 
  pre_type = None
  cls_probs = [1.] * 8
  if 'cls_probs' in x:
    cls_probs = x['cls_probs'] 
  # predictionstring list
  preds_list = []
  # store each pred word_id for one precitionstring
  preds = [] 
  pre_scores = np.zeros_like(probs[0])
  
  types = []
  
  scores = []
  pre_probs = None
  
  pre_type = 'Nothing'
  pre_cls = 0
  
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
      
      # if x['id'] in starts:
      #   is_sep = starts[x['id']][i] > 0.45
      if False:
        pass
      else:
        if not FLAGS.post_adjust: 
          is_sep = sep_prob > 0.5     
        else:
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
        if not FLAGS.out_overlap:
          if pre_type != 'Nothing':
            if pre_probs.max() > proba_thresh[pre_type]:
              preds_list.append(' '.join(preds))
              types.append(pre_type)
              scores.append((pre_probs.max() + sep_prob) * len(preds)) 
            else:
              scores.append((pre_probs[0] + sep_prob) * len(preds))
          else:
            scores.append((pre_probs[0] + sep_prob) * len(preds))
        else:
          count = 0
          for cls_ in range(1, NUM_CLASSES):
            if pre_probs[cls_] > proba_thresh[id2dis[cls_]]: 
              preds_list.append(' '.join(preds))
              types.append(id2dis[cls_])
              count += 1
      
        preds = []
        pre_scores = np.zeros_like(probs[0])
              
    pre_scores += probs[i] 
    pre_probs = gezi.softmax(pre_scores)
    preds.append(str(i))
    
  sep_prob = 1.
  if preds:
    pre_cls = np.argmax(pre_probs)
    pre_type = id2dis[pre_cls]
      
    if pre_type != 'Nothing':
      if pre_probs.max() > proba_thresh[pre_type]:
        preds_list.append(' '.join(preds))
        types.append(pre_type)
        scores.append((pre_probs.max() + sep_prob) * len(preds))
      else:
        scores.append((pre_probs[0] + sep_prob) * len(preds))
    else:
      scores.append((pre_probs[0] + sep_prob) * len(preds))
  
  scores = scores or [0]
  score = np.sum(scores) / total
  if return_score:
    return types, preds_list, score
  else:
    return types, preds_list

def get_preds_(x, votes=None, selected_ids=None, fold=None, folds=5):  
  # ic(post_adjust)
  pred_fn = None
  if votes is None:
    if FLAGS.token2word:
      if FLAGS.pred_method == 'end':
        pred_fn = get_pred_byend
      elif FLAGS.pred_method == 'se':
        pred_fn = get_pred_byse
      elif FLAGS.pred_method == 'start': #by default
        pred_fn = get_pred_bystart
      elif FLAGS.pred_method == 'reverse':
        pred_fn = get_pred_reverse
      elif FLAGS.pred_method == 'bi':
        pred_fn = get_pred_bidirectional
      else:
        raise ValueError(FLAGS.pred_method)
    else:
      if FLAGS.pred_method == 'end':
        pred_fn = get_pred_byend2
      elif FLAGS.pred_method == 'se':
        pred_fn = get_pred_byse2
      elif FLAGS.pred_method == 'start':
        pred_fn = get_pred_bystart2
      else:
        raise ValueError(FLAGS.pred_method)
  else:
    pred_fn = lambda x: get_pred_byvote(x, votes, weights=None)
  # ic(pred_fn)

  total = len(x['id'])
  # with gezi.Timer('get_preds'):
  # ic(FLAGS.openmp)
  ids_list, types_list, preds_list = [], [], []
  xs = gezi.batch2list(x)
  for i in tqdm(range(total), desc='get_preds', leave=False):
    id = xs[i]['id']
    if selected_ids is not None and id not in selected_ids:
      continue
    if fold is not None:
      if i % folds != fold:
        continue
    types, preds = pred_fn(xs[i])
    ids_list.extend([id] * len(types))
    types_list.extend(types)
    preds_list.extend(preds)
 
  m = {
    'id': ids_list,
    'class': types_list,
    'predictionstring': preds_list
  }

  df = pd.DataFrame(m)
    
  return df

def get_pred_bidirectional(x):
  types1, preds_list1, score1 = get_pred_bystart(x, return_score=True)
  types2, preds_list2, score2 = get_pred_reverse(x, return_score=True)
  if score2 > score1:
    return types2, preds_list2
  else:
    return types1, preds_list1

def get_preds(x, votes=None, selected_ids=None, folds=5):  
  if selected_ids is not None:
    return get_preds_(x, selected_ids)
  else:
    if FLAGS.pymp:
      try:
        dfs = Manager().dict()
        with pymp.Parallel(folds) as p:
          for i in p.range(folds):
            dfs[i] = get_preds_(x, votes=votes, fold=i, folds=folds)
        return pd.concat(dfs.values())
      except Exception as e:
        ic(e)
        FLAGS.pymp = False
    if not FLAGS.pymp:
      return get_preds_(x, votes=votes)

def get_preds_votes_(x, selected_ids=None, fold=None, folds=5):  
  total = len(x['id'])
  res = {}
  xs = gezi.batch2list(x)
  ids_list, types_list, preds_list = [], [], []
  for i in tqdm(range(len(xs)), desc='get_preds', leave=False):
    x = xs[i]
    id = x['id']
    if selected_ids is not None and id not in selected_ids:
      continue
    if fold is not None:
      if i % folds != fold:
        continue
    vote = get_pred_dict(x)
    res[id] = vote  
  return res

def get_preds_votes(x, selected_ids=None, folds=5):  
  if selected_ids is not None:
    return get_preds_votes_(x, selected_ids)
  else:
    if FLAGS.pymp:
      try:
        dfs = Manager().dict()
        with pymp.Parallel(folds) as p:
          for i in p.range(folds):
            dfs[i] = get_preds_votes_(x, fold=i, folds=folds)
        res = gezi.merge_dict_values(dfs)
        return res
      except Exception as e:
        ic(e)
        FLAGS.pymp = False
    if not FLAGS.pymp:
      res = get_preds_votes_(x)
      return res

## post deal rules
## short evidence merge offline incrase 1-2k
# https://www.kaggle.com/kaggleqrdl/tensorflow-longformer-ner-postprocessing
def link_evidence(oof):
  if not len(oof):
    return oof
  
  def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])
  
  thresh = 1
  idu = oof['id'].unique()
  
  eoof = oof[oof['class'] == "Evidence"]
  neoof = oof[oof['class'] != "Evidence"]
  eoof.index = eoof[['id', 'class']]
  for thresh2 in range(26, 27, 1):
    retval = []
    for idv in tqdm(idu, desc='link_evidence', leave=False):
      for c in ['Evidence']:
        q = eoof[(eoof['id'] == idv)]
        if len(q) == 0:
          continue
        pst = []
        for r in q.itertuples():
          pst = [*pst, -1,  *[int(x) for x in r.predictionstring.split()]]
        start = 1
        end = 1
        for i in range(2, len(pst)):
          cur = pst[i]
          end = i
          if  ((cur == -1) and ((pst[i + 1] > pst[end - 1] + thresh) or (pst[i + 1] - pst[start] > thresh2))):
            retval.append((idv, c, jn(pst, start, end)))
            start = i + 1
        v = (idv, c, jn(pst, start, end + 1))
        retval.append(v)
    roof = pd.DataFrame(retval, columns=['id', 'class', 'predictionstring'])
    roof = roof.merge(neoof, how='outer')
    return roof
  
def get_pred_dict(x, pred_info=None, return_nothing=True):
  pred = x['preds']
  total = len(pred)
  # by prob not logit
  probs = x['probs'] 
  # probs = x['pred']
  start_prob = x['start_probs'] 
  pre_type = None
  # predictionstring list
  preds_list = []
  # store each pred word_id for one precitionstring
  preds = [] 
  pre_scores = np.zeros_like(probs[0])
  
  types = []
  res = {}
  pre_probs = None
  for i in range(total):    
   
    is_sep = False
    if i > 0:
      pre_cls = np.argmax(pre_scores)
      pre_prob = pre_probs[pre_cls]
      pre_type = id2dis[pre_cls]
      last_type = id2dis[pred[i - 1]]
      now_cls = pred[i]
      now_type = id2dis[now_cls]
      now_prob = probs[i][now_cls]
      if np.absolute(start_prob[i]).sum() == 0:
        is_sep = True
      else:            
        is_sep = start_prob[i][1] >= P['sep_prob']
        if i > 0:
          if pred[i] != pred[i - 1]:              
            if start_prob[i][1] > P['sep_prob2']:
              is_sep = True
                                
            if pre_probs[pred[i - 1]] > P['pre_prob']:
              is_sep = True
    if is_sep:
      if preds:  
        if pre_type != 'Nothing':
          if pre_probs.max() > proba_thresh[pre_type]:
            res[(i - len(preds), i)] = pre_cls

        preds = []
        pre_scores = np.zeros_like(probs[0])
              
    pre_scores += probs[i] 
    pre_probs = gezi.softmax(pre_scores)
    # pre_probs = pre_scores / len(preds)
    preds.append(str(i))
    
  i = total
  if preds:
    pre_cls = np.argmax(pre_scores)
    pre_type = id2dis[pre_cls]
      
    if pre_type != 'Nothing':
      if pre_probs.max() > proba_thresh[pre_type]:
        res[(i - len(preds), i)] = pre_cls

  return res

def get_pred_byvote(x, votes, weights=None):
  if weights is None:
    weights = [1] * len(votes)
    weights[0] = 1.1
  counter = Counter()
  for i, vote in enumerate(votes):
    for item in vote[x['id']]:
      counter[(item['start'], item['end'])] += weights[i]

  words = np.asarray([0] * x['num_words'])
  types, pred_list = [], []
  for item, _ in counter.most_common(100):
    start, end = item
    if words[start:end].sum() / (end - start) < 0.1:
      probs = gezi.softmax(x['probs'][start:end].sum(0))
      cls_ = np.argmax(probs)
      if probs[cls_] > proba_thresh[id2dis[cls_]]:
        words[start:end] = 1
        types.append(id2dis[cls_])
        pred_list.append(' '.join(map(str, range(start, end))))
    if words.min() > 0:
      break

  return types, pred_list

      