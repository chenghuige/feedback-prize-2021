#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval.py
#        \author   chenghuige
#          \date   2021-01-09 17:51:06.853603
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')

import os
import gezi
from gezi import logging

from absl import app, flags

FLAGS = flags.FLAGS

import math
import numpy as np
import pandas as pd
import glob
import pymp
from multiprocessing import Pool, Manager, cpu_count
from collections import OrderedDict
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, recall_score, log_loss

from gezi import logging, tqdm
from gezi.plot import display
import melt as mt

from src import config
from src.config import *
from src.util import *
from src.metrics import *
from src.get_preds import *

# CODE FROM : Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(row):
  """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
  set_pred = set(row.predictionstring_pred.split(' '))
  set_gt = set(row.predictionstring_gt.split(' '))
  # Length of each and intersection
  len_gt = len(set_gt)
  len_pred = len(set_pred)
  inter = len(set_gt.intersection(set_pred))
  overlap_1 = inter / len_gt
  overlap_2 = inter / len_pred
  return [overlap_1, overlap_2]


def calc_f1_v1(gt_df, pred_df, return_dict=False):
  """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
  gt_df = gt_df[['id','discourse_type','predictionstring']] \
      .reset_index(drop=True).copy()
  if not 'class' in pred_df.columns:
    pred_df = pred_df.rename({'discourse_type': 'class'}, axis=1)
  pred_df = pred_df[['id','class','predictionstring']] \
      .reset_index(drop=True).copy()
  pred_df['pred_id'] = pred_df.index
  gt_df['gt_id'] = gt_df.index
  # Step 1. all ground truths and predictions for a given class are compared.
  joined = pred_df.merge(gt_df,
                         left_on=['id', 'class'],
                         right_on=['id', 'discourse_type'],
                         how='outer',
                         suffixes=('_pred', '_gt'))
  joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
  joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

  joined['overlaps'] = joined.apply(calc_overlap, axis=1)

  # 2. If the overlap between the ground truth and prediction is >= 0.5,
  # and the overlap between the prediction and the ground truth >= 0.5,
  # the prediction is a match and considered a true positive.
  # If multiple matches exist, the match with the highest pair of overlaps is taken.
  joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
  joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

  joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >=
                                                          0.5)
  joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
  tp_pred_ids = joined.query('potential_TP') \
      .sort_values('max_overlap', ascending=False) \
      .groupby(['id','predictionstring_gt']).first()['pred_id'].values

  # 3. Any unmatched ground truths are false negatives
  # and any unmatched predictions are false positives.
  fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

  matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
  unmatched_gt_ids = [
      c for c in joined['gt_id'].unique() if c not in matched_gt_ids
  ]

  # Get numbers of each type
  TP = len(tp_pred_ids)
  FP = len(fp_pred_ids)
  FN = len(unmatched_gt_ids)
  #calc microf1
  f1_score = TP / (TP + 0.5 * (FP + FN))
  if not return_dict:
    return f1_score
  else:
    ret = {'f1': f1_score}
    ret['acc'] = TP / (TP + FP)
    ret['recall'] = TP / (TP + FN)
    ret['cm'] = np.asarray([
                  [TP, FP],
                  [FN, TP + FP - FN]  
                ])
    return ret
  
# https://www.kaggle.com/cpmpml/faster-metric-computation/notebook
def calc_overlap2(pred, gt):
  """
  Calculates the overlap between prediction and
  ground truth and overlap percentages used for determining
  true positives.
  """
  set_pred = set(pred.split(' '))
  set_gt = set(gt.split(' '))
  # Length of each and intersection
  len_gt = len(set_gt)
  len_pred = len(set_pred)
  inter = len(set_gt & set_pred)
  overlap_1 = inter / len_gt
  overlap_2 = inter/ len_pred
  return (overlap_1, overlap_2)

def score_feedback_comp2(gt_df, pred_df, return_dict=False):
  """
  A function that scores for the kaggle
      Student Writing Competition
      
  Uses the steps in the evaluation page here:
      https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
  """
  gt_df = gt_df[['id','discourse_type','predictionstring']].reset_index(drop=True)
  if not 'class' in pred_df.columns:
    pred_df = pred_df.rename({'discourse_type': 'class'}, axis=1)
  pred_df = pred_df[['id','class','predictionstring']].reset_index(drop=True)
  pred_df['pred_id'] = pred_df.index
  gt_df['gt_id'] = gt_df.index
  
  # Step 1. all ground truths and predictions for a given class are compared.
  joined = pred_df.merge(gt_df,
                          left_on=['id','class'],
                          right_on=['id','discourse_type'],
                          how='outer',
                          suffixes=('_pred','_gt')
                        )
  joined['predictionstring_gt'].fillna(' ', inplace=True)
  joined['predictionstring_pred'].fillna(' ', inplace=True)

  overlaps = [calc_overlap2(*args) for args in zip(joined.predictionstring_pred, 
                                                    joined.predictionstring_gt)]
  
  # 2. If the overlap between the ground truth and prediction is >= 0.5, 
  # and the overlap between the prediction and the ground truth >= 0.5,
  # the prediction is a match and considered a true positive.
  # If multiple matches exist, the match with the highest pair of overlaps is taken.
  joined['potential_TP'] = [(overlap[0] >= 0.5 and overlap[1] >= 0.5) \
                            for overlap in overlaps]
  joined['max_overlap'] = [max(*overlap) for overlap in overlaps]
  tp_pred_ids = joined.loc[joined.potential_TP, 
                            ['max_overlap', 'id','predictionstring_gt', 'pred_id']]\
      .sort_values('max_overlap', ascending=False) \
      .groupby(['id','predictionstring_gt'])['pred_id'].first()

  # 3. Any unmatched ground truths are false negatives
  # and any unmatched predictions are false positives.
  fp_pred_ids = set(joined['pred_id'].unique()) - set(tp_pred_ids)

  matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
  unmatched_gt_ids = set(joined['gt_id'].unique()) -  set(matched_gt_ids)

  # Get numbers of each type
  TP = len(tp_pred_ids)
  FP = len(fp_pred_ids)
  FN = len(unmatched_gt_ids)
  #calc microf1
  f1_score = TP / (TP + 0.5 * (FP + FN))
  if not return_dict:
    return f1_score
  else:
    ret = {'f1': f1_score}
    ret['acc'] = TP / (TP + FP)
    ret['recall'] = TP / (TP + FN)
    ret['cm'] = np.asarray([
                  [TP, FP],
                  [FN, TP + FP - FN]  
                ])
    return ret
    
calc_f1 = score_feedback_comp2
    
def calc_metrics_(df_gt, df_pred, f1_only=False):
  f1s = []
  res = {'f1/Overall': 0}
  if (not len(df_gt)) or (not len(df_pred)):
    return res
  for c in tqdm(CLASSES, leave=False, desc='calc_f1'):
    try:
      ret = calc_f1(df_gt[df_gt['discourse_type'] == c],
                    df_pred[df_pred['class'] == c], return_dict=True)
      res[f'f1/{c}'] = ret['f1']
    except Exception:
      continue
    if not f1_only:
      res[f'acc/{c}'] = ret['acc']
      res[f'recall/{c}'] = ret['recall']
      # res[f'cm/{c}'] = ret['cm']
    f1s.append(ret['f1'])
  res['f1/Overall'] = np.mean(f1s)
  # res = gezi.dict_rename(res, 'Concluding Statement', 'Concluding')
  return res

def calc_metrics(df_gt, df_pred, f1_only=False, pymp=False):
  if pymp:
    try:
      res = Manager().dict()
      res['f1/Overall'] = 0.
      if (not len(df_gt)) or (not len(df_pred)):
        return res
      with pymp.Parallel(len(CLASSES)) as p:
        for i in p.range(len(CLASSES)):
          c = CLASSES[i]
          ret = calc_f1(df_gt[df_gt['discourse_type'] == c],
                        df_pred[df_pred['class'] == c], return_dict=True)
          res[f'f1/{c}'] = ret['f1']
          if not f1_only:
            res[f'acc/{c}'] = ret['acc']
            res[f'recall/{c}'] = ret['recall']
            # res[f'cm/{c}'] = ret['cm']
      res['f1/Overall'] = np.mean([res[f'f1/{c}'] for c in CLASSES])
      # res = gezi.dict_rename(res, 'Concluding Statement', 'Concluding')
      res = dict(res)
      return res
    except Exception:
      return calc_metrics_(df_gt, df_pred, f1_only) 
  else:
    return calc_metrics_(df_gt, df_pred, f1_only)

def get_metrics(df_gt, x, res={}, prefix='', df_input=False, is_last=False, df_pred2=None, folds=5):
  if not df_input:
    df_pred = get_preds(x, folds=folds)
    if FLAGS.link_evidence:
      df_pred = link_evidence(df_pred)
    gezi.set('df_pred', df_pred)
    if is_last:
      df_pred.to_csv(f'{FLAGS.model_dir}/valid_pred.csv', index=False)
  else:
    df_pred = x
  df_pred = pd.merge(df_pred,
                  df_gt[['id', 'num_words']].drop_duplicates(),
                  on=['id'],
                  how='left')
  # 不要用df_pred.id因为有可能某些id pred没有输出结果!
  if not FLAGS.max_eval:
    df_gt = df_gt[df_gt.id.isin(set(x['id']))]
    if FLAGS.lens is not None:
      if FLAGS.lens == 0:
        ids = [id for id, num_words in zip(x['id'], x['num_words']) if num_words <= 400]
      elif FLAGS.lens == 1:
        ids = [id for id, num_words in zip(x['id'], x['num_words']) if num_words > 400 and num_words <= 800]
      else:
        ids = [id for id, num_words in zip(x['id'], x['num_words']) if num_words > 800]
      df_gt = df_gt[df_gt.id.isin(set(ids))]
  else:
    x['id'].sort()
    ids = x['id'][:FLAGS.max_eval]
    # ids = x['id'][6:7]
    # ids = x['id'][33:34]
    ic(len(ids), ids)
    l = gezi.batch2list(x)
    # ic([x for x in l if x['id'] in ids])
    # d_ = pd.DataFrame(x)
    # ic(d_[d_.id.isin(ids)])
    df_gt = df_gt[df_gt.id.isin(set(ids))]
    df_pred = df_pred[df_pred.id.isin(set(ids))]
    ic(df_pred)
    
  df_gt_ = df_gt.copy()
  df_pred_ = df_pred.copy()
  df_gt_['discourse_type'] = 'Claim'
  df_pred_['class'] = 'Claim'
  res['f1/Binary'] = calc_f1(df_gt_, df_pred_)

  metrics = calc_metrics(df_gt, df_pred)
  gezi.set('eval_metric', metrics['f1/Overall'])
  res.update(metrics)
  
  if FLAGS.eval_len:
    try:
      metrics = calc_metrics(df_gt[df_gt.num_words <= 400], df_pred[df_pred.num_words <= 400], f1_only=True)
      metrics = gezi.dict_rename(metrics, 'f1', 'f1_400-')
      res.update(metrics)
      metrics = calc_metrics(df_gt[(df_gt.num_words > 400)&(df_gt.num_words <= 800)], df_pred[(df_pred.num_words > 400)&(df_pred.num_words <= 800)], f1_only=True)
      metrics = gezi.dict_rename(metrics, 'f1', 'f1_400-800')
      res.update(metrics)
      metrics = calc_metrics(df_gt[df_gt.num_words > 800], df_pred[df_pred.num_words > 800], f1_only=True)
      metrics = gezi.dict_rename(metrics, 'f1', 'f1_800+')
      res.update(metrics)
    except Exception as e:
      ic(e)
  
  res_ = OrderedDict()
  keys_ = [*FLAGS.show_keys, 'loss/sep', 'loss/token', 'f1/sep', 'auc/sep', 'f1/token',
           'auc/token', 'f1/Overall', 'f1_400-/Overall', 'f1_400-800/Overall', 'f1_800+/Overall', *[f'f1/{c}' for c in CLASSES]]
  FLAGS.max_metrics_show = len(keys_)
  keys = gezi.unique_list(keys_ + list(res.keys()))
  res = OrderedDict({k: res[k] for k in keys if k in res})
  if prefix:
    res = gezi.dict_prefix(res, prefix)
  return res
  
df_gt = None
def evaluate(y_true, y_pred, x, other, is_last=False):
  res = {}
  # ic(y_pred.shape, x['word_ids'].shape)
  # ic(x.keys(), other.keys())
  ## for torch only
  if not 'id' in x:
    x['id'] = gezi.get('id')
  # ic(x['id'][:3])
  global df_gt
  if df_gt is None:
    # df_gt = pd.read_feather(f'{FLAGS.idir}/train.corrected.fea')
    df_gt = pd.read_feather(f'{FLAGS.idir}/train.fea')
    eval_ids = set(x['id'])
    df_gt = df_gt[df_gt['id'].isin(eval_ids)]
    df_gt[['id', 'discourse_type',
           'predictionstring']].to_csv(f'{FLAGS.model_dir}/valid_gt.csv',
                                       index=False)
    df_gt['num_words'] = df_gt.text_.apply(lambda x:len(x.split()))
    df_gt.reset_index().to_feather(f'{FLAGS.model_dir}/valid_gt.fea')

  x['mask'] = x['mask']
  
  cls_pred = np.argmax(y_pred, axis=-1)
  if y_pred.shape[-1] > NUM_CLASSES:
    cls_pred_ = cls_pred
    cls_pred = (cls_pred / 2).astype(int)
    y_true = (y_true / 2).astype(int)
      
  #---------- word 这里近似使用token， token的分类准确率
  try:
    y, y_, w = y_true.reshape(-1), cls_pred.reshape(-1), x['mask'].reshape(-1)
    res['acc/token'] = accuracy_score(y, y_, sample_weight=w)
    res['f1/token'] = f1_score(y, y_, average='macro', sample_weight=w)
    res['loss/token'] = log_loss(y, y_pred.reshape(-1, NUM_CLASSES), sample_weight=w)
    for cls in dis2id:
      c = dis2id[cls]
      res[f'true/ratio/{cls}'] = ((y == c) * w).sum() / w.sum()
      res[f'pred/ratio/{cls}'] = ((y_ == c) * w).sum() / w.sum()
    
    y_prob = gezi.softmax(y_pred)
    def get_prob(c):
      return y_prob.reshape(-1, NUM_CLASSES)[:,c]
    
    y = [a for a, b in zip(y, w) if b]
    aucs = []
    for cls in CLASSES:
      c = dis2id[cls]
      y_cls = np.asarray([int(a == c) for a in y])
      probs = get_prob(c)
      y_cls_ = np.asarray([prob for prob, b in zip(probs, w) if b])
      auc = gezi.metrics.fast_auc(y_cls, y_cls_)
      res[f'auc/{cls}'] = auc
      aucs.append(auc)
    res['auc/token'] = np.asarray(aucs).mean()
  except Exception:
    pass
  try:
    #----------- 分割准确率, 召回率， f1
    sep_logits = None
    if 'end_logits' in other:
      sep_logits = other['end_logits']
    elif 'start_logits' in other:
      sep_logits = other['start_logits']
    if sep_logits is not None:
      seps = sep_logits[:,:,1] > sep_logits[:,:,0]
    else:
      assert FLAGS.num_classes > NUM_CLASSES
      seps = (cls_pred_ % 2 == 1).astype(int)
      
    # ic(np.sum(other['end_logits'][:,1] > other['end_logits'][:,0]))
    
    y, y_, w = x['sep'].reshape(-1), seps.reshape(-1), x['mask'].reshape(-1)
    res['acc/sep'] = accuracy_score(y, y_, sample_weight=w)
    res['recall/sep'] = recall_score(y, y_, sample_weight=w)
    res['f1/sep'] = f1_score(y, y_, sample_weight=w)
    if sep_logits is not None:
      res['loss/sep'] = log_loss(y, sep_logits.reshape(-1, 2), sample_weight=w)
      sep_probs = gezi.softmax(sep_logits)[:,:,1].reshape(-1)
      res['auc/sep'] = gezi.metrics.fast_auc(np.asarray([a for a, b in zip(y, w) if b]), np.asarray([a for a, b in zip(sep_probs, w) if b]))
      res['prob/sep'] = (sep_probs * w.astype(float)).sum() / w.sum()
    res['true/ratio/sep'] = ((y == 1) * w).sum() / w.sum()
    res['pred/ratio/sep'] = ((y_ == 1) * w).sum() / w.sum()
    
    # # ----- 分割片段数目的误差
    # true_parts = x['sep'].sum(-1)
    # pred_parts = (seps * x['mask']) .sum(-1)
    # # acc2, rmse2 均通过最终分割计算出的parts
    # res['acc/parts'] = ((pred_parts + 0.5).astype(int) == true_parts).astype(int).mean()
    # res['rmse/parts'] = mean_squared_error(true_parts, pred_parts, squared=False)
    
    # if 'parts' in other:
    #   pred_parts = other['parts']
    #   # ic(pred_parts, true_parts)
    #   res['acc/parts'] = ((pred_parts + 0.5).astype(int) == true_parts).astype(int).mean()
    #   res['rmse/parts'] = mean_squared_error(true_parts, pred_parts, squared=False)
    #   # ic(list(zip(x['para_count'], pred_parts, other['parts'])))
      
    # #----------- 段落分类准确率, f1 
    # if 'para_logits' in other:
    #   para_logits = other['para_logits']
    #   para_pred = np.argmax(para_logits, axis=-1)
    #   if para_logits.shape[-1] > NUM_CLASSES:
    #     para_pred = (para_pred / 2).astype(int)
    #   # TODO 如果16分类 是否 0,1 2,3logits累加转换之后？ 而不是直接argmax 
    #   y, y_, w = x['para_type'].reshape(-1), para_pred.reshape(-1), x['para_mask'].reshape(-1)
    #   res['acc/para'] = accuracy_score(y, y_, sample_weight=w)
    #   res['f1/para'] = f1_score(y, y_, average='macro', sample_weight=w)
  except Exception:
    pass

  x_ = {
    'id': x['id'],
    'pred': y_pred,
    'word_ids': x['word_ids'],
    'num_words': x['num_words'],
    'label': x['label'],
    'start': x['sep']
  }
  x_.update(other)
  if is_last and FLAGS.tiny:
    gezi.save(x_, f'{FLAGS.model_dir}/valid_ori.pkl')
  
  convert_res(x_)
  if is_last:
    gezi.save(x_, f'{FLAGS.model_dir}/valid.pkl')
  res = get_metrics(df_gt, x_, res, prefix='Metrics/', is_last=is_last)
  
  return res

def main(_):
  #use eval.ipynb for eval and ensmeble eval
  pass

if __name__ == '__main__':
  app.run(main)
