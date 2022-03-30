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

import itertools
import numpy as np
import torch
import gezi
from gezi import tqdm
from src.config import NUM_CLASSES
from src.get_preds import *

def calc_intersect(gt, pred):
  s = min(gt[1], pred[1]) - max(gt[0], pred[0]) 
  return max(0, s)

def is_match(gt, pred):
  intersect = calc_intersect(gt, pred)
  return intersect / (gt[1] - gt[0]) >= 0.5 and intersect / (pred[1] - pred[0]) >= 0.5

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
  for c in range(1, NUM_CLASSES):
    m[c] = {
      'match': calc_match(gt[c], pred[c]),
      'gt': len(gt[c]),
      'pred': len(pred[c])
    }
  return m

def token2word(start_logits, token_logits, starts, labels, word_ids):
  start_logits_ , token_logits_ , starts_, labels_ = [], [], [], []
  for i in range(len(word_ids)):
    if word_ids[i] < 0:
      continue
    start_logits_.append(start_logits[i])
    token_logits_.append(token_logits[i])
    starts_.append(starts[i])
    labels_.append(labels[i])
  return np.asarray(start_logits_), np.asarray(token_logits_), np.asarray(starts_), np.asarray(labels_)

def word2token(x, word_ids):
  x_ = []
  idx = 0
  for i, word_id in enumerate(word_ids):
    if word_id < 0:
      x_.append(0)
    else:
      # x_.append(x[word_id])
      x_.append(x[idx])
      idx += 1
  return x_

def decode_label(starts, labels):
  m = {}
  for c in range(1, NUM_CLASSES):
    m[c] = []
  start, c = 0, labels[0]
  for i in range(len(labels)):
    # if labels[i] != c:
    if starts[i] > 0:
    # if starts[i] > 0 or labels[i] != c:
      if i > start:
        if c:
          m[c].append([start, i])
      start = i
    c = labels[i]
  i += 1
  if i > start:
    if c:
      m[c].append([start, i])
  return m

def decode_label_all(starts, labels):
  m = {}
  for c in range(NUM_CLASSES):
    m[c] = []
  start, c = 0, labels[0]
  for i in range(len(labels)):
    # if labels[i] != c:
    if starts[i] > 0:
    # if starts[i] > 0 or labels[i] != c:
      if i > start:
        m[c].append([start, i])
      start = i
    c = labels[i]
  i += 1
  if i > start:
    m[c].append([start, i])
  return m

def greedy_decode(start_probs, token_logits):
  m = {}
  for c in range(1, NUM_CLASSES):
    m[c] = []
  start, c = 0, None
  pred_tokens = np.zeros(len(start_probs))
  starts = (start_probs[:,1] > 0.5).astype(int)
  for i in range(len(starts)):
    # if starts[i] or (i > 1 and token_logits[i].argmax() != token_logits[i - 1].argmax() and start_probs[i][1] > P['sep_prob2']):
    if starts[i]:
      starts[i] = 1
      logits = gezi.softmax(token_logits[start:i]).sum(0)
      probs = gezi.softmax(logits)
      c = probs.argmax()
      max_prob = probs[c] if FLAGS.rl_prob_thre else 1.
      if c and max_prob > proba_thresh[id2dis[c]]:
        m[c].append([start, i])
      pred_tokens[start:i] = c
      start = i
  i += 1
  if i > start:
    logits = token_logits[start:i].sum(0)
    probs = gezi.softmax(logits)
    c = probs.argmax()
    max_prob = probs[c] if FLAGS.rl_prob_thre else 1.
    if c and max_prob > proba_thresh[id2dis[c]]:
      m[c].append([start, i])
    pred_tokens[start:i] = c
  return m, starts, np.asarray(pred_tokens)

def sample_decode(start_probs, token_logits):
  m = {}
  for c in range(1, NUM_CLASSES):
    m[c] = []
  start, c = 0, None
  pred_tokens = np.zeros(len(start_probs))
  starts = []
  for i in range(len(start_probs)):
    if start_probs[i][0] == 0:
      start_ = 0
    else:
      start_ = np.random.choice(2, None, p=start_probs[i])
    starts.append(start_)
    # if start_ or (i > 1 and token_logits[i].argmax() != token_logits[i - 1].argmax() and start_probs[i][1] > P['sep_prob2']):
    #   starts[-1] = 1
    if start_:
      probs = gezi.softmax(gezi.softmax(token_logits[start:i]).sum(0))
      if FLAGS.sample_tokens:
        if probs[0] == 0:
          c = 0
        else:
          c = np.random.choice(NUM_CLASSES, None, p=probs)
      else:
        c = probs.argmax()
      max_prob = probs[c] if FLAGS.rl_prob_thre else 1.
      if c and max_prob > proba_thresh[id2dis[c]]:
        m[c].append([start, i])
      pred_tokens[start:i] = c
      start = i
  i += 1
  if i > start:
    probs = gezi.softmax(gezi.softmax(token_logits[start:i]).sum(0))
    c = np.random.choice(NUM_CLASSES, None, p=probs)
    max_prob = probs[c] if FLAGS.rl_prob_thre else 1.
    if c and max_prob > proba_thresh[id2dis[c]]:
      m[c].append([start, i])
    pred_tokens[start:i] = c
  return m, np.asarray(starts), pred_tokens

def calc_f1(m):
  f1_scores = []
  ignores = 0
  for c in range(1, NUM_CLASSES):
    TP = m[c]['match']
    FP = m[c]['pred'] - TP
    FN = m[c]['gt'] - TP
    if m[c]['gt'] == 0 and m[c]['pred'] == 0:
      f1_score = 0
      ignores += 1
    else:
      f1_score = TP / (TP + 0.5 * (FP + FN))
    f1_scores.append(f1_score)
  return np.sum(f1_scores) / (len(f1_scores) - ignores)

def calc_reward(starts_list, labels_list, start_logits_list, token_logits_list, word_ids_list):
  m = {}
  for c in range(1, NUM_CLASSES):
    m[c] = {'match': 0, 'gt': 0, 'pred': 0}
  m2 = {}
  for c in range(1, NUM_CLASSES):
    m2[c] = {'match': 0, 'gt': 0, 'pred': 0}
  # greedy_f1s = []
  # sample_f1s = []
  greedy_starts_list, sample_starts_list = [], []
  greedy_tokens_list, sample_tokens_list = [], []
  for i, (starts, labels, start_logits, token_logits, word_ids) in enumerate(zip(starts_list, labels_list, start_logits_list, token_logits_list, word_ids_list)):
    if not FLAGS.merge_tokens:
      start_logits, token_logits, starts, labels = token2word(start_logits, token_logits, starts, labels, word_ids)
    # ic(greedy_starts.shape, sample_starts.shape, starts.shape, labels.shape, start_logits.shape, token_logits.shape)
    gt = decode_label(starts, labels)
    start_probs = gezi.softmax(start_logits)
    greedy, greedy_starts, greedy_tokens = greedy_decode(start_probs, token_logits)
    if not FLAGS.merge_tokens:
      greedy_starts = word2token(greedy_starts, word_ids)
      greedy_tokens = word2token(greedy_tokens, word_ids)
    res = prepare_f1(gt, greedy)
    for c in range(1, NUM_CLASSES):
      m[c]['match'] += res[c]['match']
      m[c]['gt'] += res[c]['gt']
      m[c]['pred'] += res[c]['pred']
    # greedy_f1s.append(calc_f1(res))
    
    sample, sample_starts, sample_tokens = sample_decode(start_probs, token_logits)
    if not FLAGS.merge_tokens:
      sample_starts = word2token(sample_starts, word_ids)
      sample_tokens = word2token(sample_tokens, word_ids)
    res = prepare_f1(gt, sample)
    for c in range(1, NUM_CLASSES):
      m2[c]['match'] += res[c]['match']
      m2[c]['gt'] += res[c]['gt']
      m2[c]['pred'] += res[c]['pred']
    # sample_f1s.append(calc_f1(res))
    
    greedy_tokens_list.append(greedy_tokens)
    sample_tokens_list.append(sample_tokens)
    if not FLAGS.merge_tokens:
      greedy_starts = word2token(greedy_starts, word_ids)
      sample_starts = word2token(sample_starts, word_ids)
    greedy_starts_list.append(greedy_starts)
    sample_starts_list.append(sample_starts)
  
  greedy_f1 = calc_f1(m)
  sample_f1 = calc_f1(m2)
  return greedy_f1, sample_f1,\
         np.asarray(greedy_starts_list), np.asarray(sample_starts_list), \
         np.asarray(greedy_tokens_list), np.asarray(sample_tokens_list)
