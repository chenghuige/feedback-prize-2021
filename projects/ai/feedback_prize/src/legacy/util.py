#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2021-12-15 16:19:02.210366
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pickle import NONE

import sys 
import os

from absl import app, flags
FLAGS = flags.FLAGS

import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Pool, Manager, cpu_count
try:
  import pymp
except Exception:
  pass
import pandas as pd

import tensorflow as tf
import torch
from transformers import AutoTokenizer

import gezi
from gezi import tqdm
import melt as mt

from src.config import *
from src.torch.model import Model as TorchModel
from src.tf.model import Model as TFModel
from src.preprocess import preprocess

#https://www.kaggle.com/bacicnikola/token-to-word-mapping-implementation
def tokenize(words, tokenizer, null_id=None):
  # tokenize input text
  text = ' '.join(words)
  inputs = tokenizer(text,
                     add_special_tokens=True,
                     return_offsets_mapping=True,
                     truncation=False,
                     return_length=True)
        
  word_ids = [] # list to store token -> word mapping
  word_pos = 0 # word strating position

  tokens = inputs['input_ids'][1:-1] # exclude <s> and </s> tokens
  
  # current token positions (used for iteration)
  start = 0
  end = 1

  for _ in tokens:
    decoded_word = tokenizer.decode(tokens[start:end]).strip()
    if decoded_word == '':
      # if striped word is an empty string, that token doesn't belong to any word
      word_ids.append(null_id)
      start += 1
      end += 1
      continue
    # no match
    # continue adding tokens
    if decoded_word != words[word_pos]:
      end += 1
      word_ids.append(word_pos)
    # match    
    else:
      word_ids.append(word_pos)
      start = end
      end = start + 1
      word_pos += 1
  
  # add -1 position for the <s> and </s> tokens
  word_ids = [null_id] + word_ids + [null_id]
  inputs['word_ids'] = word_ids
  return inputs

# [SEP] or \n directly , bert模型可能词表没有\n [SEP]比较安全
# BR = '[SEP]'

def get_words(text):
  if FLAGS.remove_br:
    return text.split()
  words = []
  parts = text.split('\n')
  for part in parts:
    if part:
      words.extend(part.split())
      words.append(FLAGS.br)
    else:
      words.append(FLAGS.br)
  words = words[:-1]
  assert len(words) - text.count('\n') == len(text.split())
  return words

def fix_wordids(word_ids, words):
  if FLAGS.remove_br:
    return word_ids
  ignores = set()
  br_indexes = set()
  word_ids_ = word_ids.copy()
  for i, word_id in enumerate(word_ids): 
    if word_id is not None:
      if words[word_id] == FLAGS.br:
        # change \n word_id to None
        word_ids_[i] = None
        ignores.add(word_id)
        br_indexes.add(i)
        # 注意只追加一个如果后面连续两个换行 只用第一
        if FLAGS.merge_br == 'end':
          if i > 0 and (i - 1) not in br_indexes:
            word_ids_[i] = word_ids_[i - 1]
      else:
        word_ids_[i] -= len(ignores)
        # 注意只追溯一个 如果前面连续两个换行 只用最后一个, 因为用start判断sep 所以走这里
        if FLAGS.merge_br == 'start':
          # if i > 0 and (i - 1) in br_indexes:
          #   word_ids_[i - 1] = word_ids_[i]
          # 追溯所有换行符
          j = i 
          while j > 0 and (j - 1) in br_indexes:
            word_ids_[j - 1] = word_ids_[j]
            j -= 1
        # BI -> I
        if FLAGS.mask_inside and (i > 1 and word_ids[i] == word_ids[i - 1]):
          if FLAGS.mask_inside_method == 'first':
            word_ids_[i] = None
          else:
            word_ids_[i - 1] = None
              
  return word_ids_

def get_brs_map(word_ids, words):
  brs_map = {None: None}
  ignores = set()
  for i, word_id in enumerate(word_ids):
    if word_id is not None:
      word = words[word_id]
      if word == FLAGS.br:
        ignores.add(word_id)
        brs_map[word_id] = None
      else:
        brs_map[word_id] = word_id - len(ignores)
  return brs_map
      
def remap_wordids(word_ids, words=None, brs_map=None):
  if brs_map is None:
    brs_map = get_brs_map(word_ids, words)
  word_ids_ = [brs_map[x] for x in word_ids]
  return word_ids_

def encode(text, tokenizer):
  words = get_words(text)   
  words = [preprocess(x) for x in words]

  if not 'deberta-v' in FLAGS.backbone:
    encoding = tokenizer(words, is_split_into_words=True, truncation=False)
    word_ids = encoding.word_ids()
  else:
    encoding = tokenize(words, tokenizer)
    word_ids = encoding['word_ids']
    
  num_words = len(words)
  # [0,1) + 1
  relative_positions = [0. if x is None else 1. + x / num_words for x in word_ids]
  word_ids = fix_wordids(word_ids, words)
  
  # assert(max([x for x in word_ids if x is not None]) == len(text.split()) - 1), len(text.split())
  assert(max([x for x in word_ids if x is not None]) < len(text.split())), "try adding --br='[SEP]' or --br='[unused0]'"
  assert(min([x for x in word_ids if x is not None]) >= 0)
  # assert(min([x for x in word_ids if x is not None]) == 0)
  input_ids = encoding['input_ids']
  attention_mask = encoding['attention_mask']
  
  start = 0
  last_tokens = 0
  if FLAGS.split_method == 'end':
    last_tokens = FLAGS.max_len
  elif FLAGS.split_method == 'se':
    last_tokens = FLAGS.last_tokens
  elif FLAGS.split_method == 'mid':
    start = max(int((len(word_ids) - FLAGS.max_len) / 2), 0)
        
  word_ids = [null_wordid() if x is None else x for x in word_ids]
    
  input_ids = gezi.pad(input_ids[start:], FLAGS.max_len, tokenizer.pad_token_id, 
                       last_tokens, use_sep=True, sep_token=tokenizer.sep_token_id)
  attention_mask = gezi.pad(attention_mask[start:], FLAGS.max_len, 0, 
                            last_tokens, use_sep=True, sep_token=1)
  word_ids = gezi.pad(word_ids[start:], FLAGS.max_len, null_wordid(), 
                      last_tokens, use_sep=True, sep_token=null_wordid())
  relative_positions = gezi.pad(relative_positions[start:], FLAGS.max_len, 0., 
                       last_tokens, use_sep=True, sep_token=0)
  
  if FLAGS.force_cls:
    ## 补充前sep
    if input_ids[0] != tokenizer.cls_token_id:
      input_ids[0] = tokenizer.sep_token_id
      word_ids[0] = null_wordid()
      relative_positions[0] = 0.
    ## 补充后sep
    for i in range(len(input_ids)):
      if input_ids[i] == tokenizer.pad_token_id:
        i -= 1
        break 
    if input_ids[i] != tokenizer.eos_token_id:
      input_ids[i] = tokenizer.sep_token_id
      word_ids[i] = null_wordid()
      relative_positions[0] = 0.
  
  return input_ids, attention_mask, word_ids, relative_positions             

def get_model():
  if not FLAGS.torch:
    model = TFModel().build_model()
  else:
    model = TorchModel()
  return model

test_inputs = {}
def get_inputs(backbone, mode='test', reset=True, sort=True, double_times=0):
  #--- for online infer
  FLAGS.remove_br = False
  if backbone in test_inputs and not reset:
    return test_inputs[backbone]
  else:
    tokenizer = get_tokenizer(backbone)
    files = os.listdir(f'../input/feedback-prize-2021/{mode}')
    TEST_IDS = [f.replace('.txt','') for f in files if 'txt' in f]
    ic(len(TEST_IDS), torch.cuda.is_available(), tokenizer.padding_side)
    if len(TEST_IDS) < 10 and torch.cuda.is_available() and double_times:
      for _ in range(double_times):
        TEST_IDS += TEST_IDS
    ic(len(TEST_IDS))
    MAX_LEN = FLAGS.max_len
    input_ids = np.zeros((len(TEST_IDS), MAX_LEN), dtype='int32')
    attention_mask = np.zeros((len(TEST_IDS), MAX_LEN), dtype='int32')
    word_ids_list = []
    re_pos_list = []
    num_words_list = []
   
    for id_num in tqdm(range(len(TEST_IDS))):
      n = TEST_IDS[id_num]
      name = f'../input/feedback-prize-2021/{mode}/{n}.txt'
      txt = open(name, 'r').read()
      input_ids_, attention_mask_, word_ids, relative_positions = encode(txt, tokenizer) 
      input_ids[id_num,] = input_ids_
      attention_mask[id_num,] = attention_mask_
     
      word_ids_list.append(word_ids)
      re_pos_list.append(relative_positions)
      num_words_list.append(len(txt.split()))
    
    inputs = {
      'id': TEST_IDS,
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'word_ids': word_ids_list,
      'relative_positions': re_pos_list,
      'num_words': num_words_list,
    }    
    keys = list(inputs.keys())
    if sort:
      df = pd.DataFrame({k: list(inputs[k]) for k in keys})
      df = df.sort_values('num_words')
      inputs = {
        k: np.asarray(df[k].values) for k in keys
      }
    if not reset:
      test_inputs[backbone] = inputs
    return inputs

def null_wordid():
  # return FLAGS.max_len
  return -1

def is_wordid(x):
  return x != null_wordid() 
  
def preds_token2word(preds, word_ids, num_words, reduce_method='first'):
  word_id_ = None
  preds_ = []
  probs = np.zeros([num_words, preds[0].shape[-1]])

  def get_pred(l):
    if reduce_method == 'first':
      return l[0]
    elif reduce_method == 'last':
      return l[-1]
    elif reduce_method == 'sum':
      return sum(l)
    elif reduce_method in ['mean', 'avg']:
      return sum(l) / len(l)
    else:
      raise ValueError(reduce_method)
  
  for i, word_id in enumerate(word_ids):
    if not is_wordid(word_id):
      continue
    # if FLAGS.split_method != 'start' and i <= 512:
    #   continue
    if word_id != word_id_:
      if preds_:
        probs[word_id_] = get_pred(preds_)
        preds_ = []
      word_id_ = word_id
   
    preds_.append(preds[i])
    
  if preds_:
    probs[word_id_] = get_pred(preds_)
  return probs

def token2word(x):  
  total = len(x['pred'])
  if FLAGS.records_type == 'token':
    reduce_method = FLAGS.post_reduce_method or 'avg'
    x['pred'] = [preds_token2word(x['pred'][i], x['word_ids'][i], reduce_method=reduce_method, num_words=x['num_words'][i]) for i in tqdm(range(total), desc='pred_t2w', leave=False)]
  x['probs'] = [gezi.softmax(pred) for pred in x['pred']] 
  x['preds'] = [np.argmax(prob, axis=-1) for prob in x['probs']]
  if 'start_logits' in x  and (FLAGS.pred_method != 'end'):
    if FLAGS.records_type == 'token':
      reduce_method2 = FLAGS.post_reduce_method2 or 'first'
      x['start_logits'] = [preds_token2word(x['start_logits'][i], x['word_ids'][i], reduce_method=reduce_method2, num_words=x['num_words'][i]) for i in tqdm(range(total), desc='start_t2w', leave=False)]
    x['start_probs'] = [gezi.softmax(start_logit) for start_logit in x['start_logits']]  
  if 'end_logits' in x  and (FLAGS.pred_method != 'start'):
    if FLAGS.records_type == 'token':
      reduce_method2 = FLAGS.post_reduce_method2 or 'first'
      x['end_logits'] = [preds_token2word(x['end_logits'][i], x['word_ids'][i], reduce_method=reduce_method2, num_words=x['num_words'][i]) for i in tqdm(range(total), desc='end_t2w', leave=False)]
    x['end_probs'] = [gezi.softmax(end_logit) for end_logit in x['end_logits']]  
  
  for cls_ in classes:
    if f'{cls_}_logits' in x:
      reduce_method2 = FLAGS.post_reduce_method2 or 'first'
      x[f'{cls_}_logits'] = [preds_token2word(x[f'{cls_}_logits'][i], x['word_ids'][i], reduce_method=reduce_method2, num_words=x['num_words'][i]) for i in tqdm(range(total), desc=f'{cls_}_t2w', leave=False)]
      x[f'{cls_}_probs'] = [gezi.softmax(logit) for logit in x[f'{cls_}_logits']]  
      
  if 'cls_logits' in x:
     x['cls_probs'] = gezi.sigmoid(x['cls_logits'])

def convert(x):
  x['probs'] = gezi.softmax(x['pred'])
  x['preds'] = np.argmax(x['probs'], -1)
  if 'start_logits' in x:
    x['start_probs'] = gezi.softmax(x['start_logits'])
  if 'end_logits' in x:
    x['end_probs'] = gezi.softmax(x['end_logits'])
    # ic(np.sum(x['end_probs'][:,1] > x['end_probs'][:,0]))
  if 'cls_logits' in x:
    x['cls_probs'] = gezi.sigmoid(x['cls_logits'])
    
def convert_res(x):
  if FLAGS.token2word:
    token2word(x)
  else:
    convert(x)
    
# 似乎by logits效果更好
def ensemble_res(xs, weights=None, by_prob=False):
  x_ = xs[0].copy()
  total = len(x_['pred'])
  weights = weights or [1.] * len(xs)
  x_['pred'] = [np.stack([x['pred'][i] * weight for x, weight in zip(xs, weights)], 1).mean(axis=1) for i in range(total)]
  if by_prob:
    x_['probs'] = [np.stack([x['probs'][i] * weight for x, weight in zip(xs, weights)], 1).mean(axis=1) for i in range(total)]
  else:
    x_['probs'] = [gezi.softmax(x_['pred'][i]) for i in range(total)]
  x_['preds'] = [np.argmax(x_['probs'][i], axis=-1) for i in range(total)]
  if 'start_logits' in x_ and (FLAGS.pred_method != 'end'):
    x_['start_logits'] = [np.stack([x['start_logits'][i] * weight for x, weight in zip(xs, weights)], 1).mean(axis=1) for i in range(total)]
    if by_prob:
      x_['start_probs'] = [np.stack([x['start_probs'][i] * weight for x, weight in zip(xs, weights)], 1).mean(axis=1) for i in range(total)]
    else:
      x_['start_probs'] = [gezi.softmax(x_['start_logits'][i]) for i in range(total)]
  if 'end_logits' in x_ and (FLAGS.pred_method != 'start'):
    x_['end_logits'] = [np.stack([x['end_logits'][i] * weight for x, weight in zip(xs, weights)], 1).mean(axis=1) for i in range(total)]
    if by_prob:
      x_['end_probs'] = [np.stack([x['end_probs'][i] * weight for x, weight in zip(xs, weights)], 1).mean(axis=1) for i in range(total)]
    else:
      x_['end_probs'] = [gezi.softmax(x_['end_logits'][i]) for i in range(total)]
  return x_

class Ensembler(object):
  def __init__(self, need_sort=False):
    self.x = None
    self.need_sort = need_sort
    self.weights = []
    self.total_weight = 0
    
  def add(self, x, weight=1.):
    self.total_weight += weight
    if self.need_sort:
      inds = np.asarray(x['id']).argsort()
      for key in x:
        try:
          x[key] = x[key][inds]
        except Exception:
          # ic(key)
          x[key] = [x[key][idx] for idx in inds]
    
    # for i, (word_ids, num_words) in enumerate(zip(x['word_ids'], x['num_words'])):
    #   weights = np.zeros([num_words])
    #   ## TODO which is better 
    #   # for word_id in word_ids:
    #   #   weights[word_id] = weight
    #   weights = np.zeros([num_words]) + weight
      
    #   if len(self.weights) > i:
    #     self.weights[i] += weights
    #   else:
    #     self.weights.append(weights)
    # if FLAGS.split_method != 'start':
    #   for i in range(len(x['pred'])):
    #     for j in range(512):
    #       x['pred'][j] *= 0.
          
    if self.x is None: 
      self.x = x
      if weight != 1:
        self.x.update({
            'pred': [pred * weight for pred in x['pred']],
            'start_logits': [logit * weight for logit in x['start_logits']],
          })
        if 'cls_logits' in x:
          self.x.update({'cls_logits': [logit * weight for logit in x['cls_logits']]})
        for cls_ in classes:
          if f'{cls_}_logits' in x:
            self.x.update({f'{cls_}_logits': [logit * weight for logit in x[f'{cls_}_logits']]})
    else:
      self.x.update({
        'pred': [pred1 + pred2 * weight for pred1, pred2 in zip(self.x['pred'], x['pred'])],
        'start_logits': [pred1 + pred2 * weight for pred1, pred2 in zip(self.x['start_logits'], x['start_logits'])], 
        })
      if 'cls_logits' in x:
        self.x.update({'cls_logits': [pred1 + pred2 * weight for pred1, pred2 in zip(self.x['cls_logits'], x['cls_logits'])]})
      for cls_ in classes:
        if f'{cls_}_logits' in x:
          self.x.update({f'{cls_}_logits': [pred1 + pred2 * weight for pred1, pred2 in zip(self.x[f'{cls_}_logits'], x[f'{cls_}_logits'])]})

  # def finalize(self):
  #   x = self.x
  #   x['pred'] =  [pred / np.maximum(self.weights[i], 1.)[..., np.newaxis] for i, pred in enumerate(x['pred'])]
  #   x['probs'] = [gezi.softmax(pred) for pred in x['pred']]
  #   # del x['pred']
  #   x['start_logits'] = [logit / np.maximum(self.weights[i], 1.)[..., np.newaxis] for i, logit in enumerate(x['start_logits'])]
  #   x['start_probs'] = [gezi.softmax(logit) for logit in x['start_logits']]
  #   del x['start_logits']
  #   if 'cls_logits' in x:
  #     x['cls_logits'] = [logit / np.maximum(self.weights[i], 1.)[..., np.newaxis] for i, logit in enumerate(x['cls_logits'])]
  #     x['cls_probs'] = [gezi.sigmoid(logit) for logit in x['cls_logits']]
  #   return x
  
  def finalize(self):
    x = self.x
    weight = self.total_weight
    x['pred'] =  [pred / weight for pred in x['pred']]
    x['probs'] = [gezi.softmax(pred) for pred in x['pred']]
    # del x['pred']
    x['start_logits'] = [logit / weight for logit in x['start_logits']]
    x['start_probs'] = [gezi.softmax(logit) for logit in x['start_logits']]
    del x['start_logits']
    if 'cls_logits' in x:
      x['cls_logits'] = [logit / weight for logit in x['cls_logits']]
      x['cls_probs'] = [gezi.sigmoid(logit) for logit in x['cls_logits']]
    return x

# from https://www.kaggle.com/abhishek/two-longformers-are-better-than-1
# about 1-1.5k offline

proba_thresh = {
    "Lead": 0.99,
    "Position": 0.55,
    "Evidence": 0.7,
    "Claim": 0.55,
    "Concluding Statement": 0.99,
    "Counterclaim": 0.5,
    "Rebuttal": 0.55,
}

def get_pred_bystart(x, post_adjust=True):
  MIN_LEN = FLAGS.para_min_len  #2
  MIN_LEN2 = FLAGS.para_min_len2 #6
  MIN_EVIDENCE_LEN = 27
  NUM_CLASSES = len(id2dis)
  pred = x['preds']
  total = len(pred)
  # by prob not logit
  probs = x['probs'] 
  # probs = x['pred']
  cls_probs = x['cls_probs'] if 'cls_probs' in x else np.asarray([1.] * NUM_CLASSES)
  assert 'cls_probs' in x or FLAGS.cls_loss_rate == 0
  start_prob = x['start_probs'] if 'start_probs' in x else None
  # ic((start_prob[:,1] > start_prob[:,0]).astype(int).sum())
  word_id_ = None
  pre_type = None
  # predictionstring list
  preds_list = []
  # store each pred word_id for one precitionstring
  preds = [] 
  pre_types = []
  pre_scores = np.zeros_like(probs[0])
  
  idxs = []
  types = []
   
  if 'cls_probs' in x:
    label_count_ratio = cls_probs
    
  for i in range(total):    
    if pre_scores.shape[-1] > NUM_CLASSES:
      pre_type = id2dis[int(np.argmax(pre_scores) / 2)]
    else:
      pre_type = id2dis[np.argmax(pre_scores)]
  
    is_sep = False
    if start_prob[i].sum() == 0:
      is_sep = True
      
    if start_prob is None:
      is_sep = pred[i] % 2 == 1
    else:
      is_sep = start_prob[i][1] > 0.5
      
      if post_adjust:
        if i > 0:
          if pred[i] != pred[i - 1]:
            if start_prob[i][1] > 0.3 and pre_type in  ['Rebuttal', 'Counterclaim']:
                is_sep = True
          else:
            if pre_type == 'Evidence':
              if start_prob[i][1] < 0.55:
                is_sep = False
            
    if is_sep:
      if preds:
        pre_scores = gezi.softmax(pre_scores)  
        if pre_type != 'Nothing':
          if post_adjust:
            if len(preds) < MIN_LEN:
              # 低置信度的干脆放弃召回 更安全
              # continue
              pass
            else:
              if pre_scores.max() > proba_thresh[pre_type]:
                preds_list.append(' '.join(preds))
                types.append(pre_type)
          else:
            preds_list.append(' '.join(preds))
            types.append(pre_type)
            
        preds = []
        pre_types = []
        pre_scores = np.zeros_like(probs[0])
        
    if pre_scores.shape[-1] > NUM_CLASSES:
      pre_type = id2dis[int(np.argmax(probs[i]) / 2)]
    else:
      pre_type = id2dis[np.argmax(probs[i])]
      
    pre_types.append(pre_type)
    pre_scores += probs[i] 
    preds.append(str(i))
    
  if preds:
    pre_scores = gezi.softmax(pre_scores)
    if pre_scores.shape[-1] > NUM_CLASSES:
      pre_type = id2dis[int(np.argmax(pre_scores) / 2)]
    else:
      pre_type = id2dis[np.argmax(pre_scores)]
      
    # 结尾应该更长
    if pre_type != 'Nothing':
      if post_adjust:
        if len(preds) >= MIN_LEN2:
          if pre_scores.max() > proba_thresh[pre_type]:
            preds_list.append(' '.join(preds))
            types.append(pre_type)
      else:
        preds_list.append(' '.join(preds))
        types.append(pre_type)
  return types, preds_list


def get_preds(x):  
  
  pred_fn = None
  if FLAGS.token2word:
    if FLAGS.pred_method == 'end':
      pred_fn = get_pred_byend
    elif FLAGS.pred_method == 'se':
      pred_fn = get_pred_byse
    elif FLAGS.pred_method == 'start':
      pred_fn = get_pred_bystart
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
  # ic(pred_fn)

  total = len(x['preds'])
  # with gezi.Timer('get_preds'):
  if not FLAGS.openmp:
    ids_list, types_list, preds_list = [], [], []
    for i in tqdm(range(total), desc='get_preds', leave=False):
      id = x['id'][i]
      x_ = {}
      for key in x: 
        x_[key] = x[key][i]
      types, preds = pred_fn(x_)
      ids_list.extend([id] * len(types))
      types_list.extend(types)
      preds_list.extend(preds)
  else:
    ids_list, types_list, preds_list = Manager().list(), Manager().list(), Manager().list()  
    nw = cpu_count()
    with pymp.Parallel(nw) as p:
      for i in p.range(total):
        id = x['id'][i]
        x_ = {}
        for key in x:
          x_[key] = x[key][i]
        types, preds = pred_fn(x_)
        ids = [id] * len(types)
        ids_list.extend([id] * len(types))
        types_list.extend(types)
        preds_list.extend(preds)
    ids_list, types_list, preds_list = list(ids_list), list(types_list), list(preds_list)
 
  m = {
    'id': ids_list,
    'class': types_list,
    'predictionstring': preds_list
  }

  df = pd.DataFrame(m)
    
  return df

# by dis start and end
def get_pred_byse(x):
  MIN_LEN = FLAGS.para_min_len  #2
  MIN_LEN2 = FLAGS.para_min_len2 #6
  NUM_CLASSES = len(id2dis)
  pred = x['preds']
  total = len(pred)
  probs = x['probs']
  # probs = x['pred']
  start_prob = x['start_probs']
  end_prob = x['end_probs'] 
  word_id_ = None
  pre_type = None
  # predictionstring list
  preds_list = []
  # store each pred word_id for one precitionstring
  preds = [] 
  pre_types = []
  pre_scores = np.zeros_like(probs[0])
  
  idxs = []
  types = []
  label_count_ratio = [0.17623337623337623,
                    0.28704704704704703,
                    0.26125554125554123,
                    0.08812812812812813,
                    0.07625911625911626,
                    0.05322465322465322,
                    0.03318175318175318,
                    0.02467038467038467]
  
  label_count_ratio = np.array(label_count_ratio)
  label_count_ratio[0] *= 10
  
  has_start = False
  
  for i in range(total):    
    is_start = start_prob[i][1] > start_prob[i][0]
    if is_start:
      has_start = True
    
    if has_start:    
      pre_type = id2dis[np.argmax(probs[i])]
      pre_types.append(pre_type)
      pre_scores += probs[i]
      preds.append(str(i))
      
    is_end = end_prob[i][1] > end_prob[i][0]
    if is_end:
      if preds:
        pre_scores = gezi.softmax(pre_scores) * label_count_ratio

        if pre_scores.shape[-1] > NUM_CLASSES:
          pre_type = id2dis[int(np.argmax(pre_scores) / 2)]
        else:
          pre_type = id2dis[np.argmax(pre_scores)]
        
        if len(preds) < MIN_LEN:
          # 低置信度的干脆放弃召回
          # continue
          pass
        else:
          if pre_type != 'Nothing':
            preds_list.append(' '.join(preds))
            types.append(pre_type)
        preds = []
        pre_types = []
        pre_scores = np.zeros_like(probs[0])
        has_start = False
        
  if preds:
    # pre_scores = gezi.softmax(pre_scores) * label_count_ratio_ 
    if pre_scores.shape[-1] > NUM_CLASSES:
      pre_type = id2dis[int(np.argmax(pre_scores) / 2)]
    else:
      pre_type = id2dis[np.argmax(pre_scores)]
    
    # 结尾应该更长
    if len(preds) >= MIN_LEN2:
      if pre_type != 'Nothing':
        preds_list.append(' '.join(preds))
        types.append(pre_type)
  return types, preds_list

def get_pred_bystart2(x):
  MIN_LEN = FLAGS.para_min_len  #2
  MIN_LEN2 = FLAGS.para_min_len2 #6
  NUM_CLASSES = len(id2dis)  
  pred = x['preds']
  probs = x['probs']
  start_prob = None
  if 'start_probs' in x:
    start_prob = x['start_probs']
  word_id_ = None
  pre_type = None
  # predictionstring list
  preds_list = []
  # store each pred word_id for one precitionstring
  preds = [] 
  pre_types = []
  pre_scores = np.zeros_like(probs[0])
  
  idxs = []
  types = []
  
  # got from ./gen-records.py --stats count freq of each discourse segment
  if pre_scores.shape[-1] > NUM_CLASSES:
    label_count_ratio = [
                  0.17623337623337623,
                  0.17623337623337623,
                  0.28704704704704703,
                  0.28704704704704703,
                  0.26125554125554123,
                  0.26125554125554123,
                  0.08812812812812813,
                  0.08812812812812813,
                  0.07625911625911626,
                  0.07625911625911626,
                  0.05322465322465322,
                  0.05322465322465322,
                  0.03318175318175318,
                  0.03318175318175318,
                  0.02467038467038467,
                  0.02467038467038467]
  else:
    label_count_ratio = [0.17623337623337623,
                      0.28704704704704703,
                      0.26125554125554123,
                      0.08812812812812813,
                      0.07625911625911626,
                      0.05322465322465322,
                      0.03318175318175318,
                      0.02467038467038467]
  
  label_count_ratio = np.array(label_count_ratio)
  # label_count_ratio[0] *= 10
  
  word_ids = x['word_ids']
  for i, word_id in enumerate(word_ids):
    if not is_wordid(word_id) :
      continue  
    is_sep = False
  
    if start_prob is None:
      is_sep = pred[i] % 2 == 1
    else:
      is_sep = start_prob[i][1] > start_prob[i][0]
      
    if is_sep:
      if preds:
        pre_scores = gezi.softmax(pre_scores) * label_count_ratio  

        if pre_scores.shape[-1] > NUM_CLASSES:
          pre_type = id2dis[int(np.argmax(pre_scores) / 2)]
        else:
          pre_type = id2dis[np.argmax(pre_scores)]
        
        preds = gezi.unique_list(preds)
        if len(preds) < MIN_LEN:
          # 低置信度的干脆放弃召回
          continue
          pass
        else:
          if pre_type != 'Nothing':
            preds_list.append(' '.join(preds))
            types.append(pre_type)
        preds = []
        pre_types = []
        pre_scores = np.zeros_like(probs[0])
        
    if pre_scores.shape[-1] > NUM_CLASSES:
      pre_type = id2dis[int(np.argmax(probs[i]) / 2)]
    else:
      pre_type = id2dis[np.argmax(probs[i])]
    pre_types.append(pre_type)
    pre_scores += probs[i]
    preds.append(str(word_id))
  if preds:
    pre_scores = gezi.softmax(pre_scores) * label_count_ratio
    if pre_scores.shape[-1] > NUM_CLASSES:
      pre_type = id2dis[int(np.argmax(pre_scores) / 2)]
    else:
      pre_type = id2dis[np.argmax(pre_scores)]
    
    # 结尾应该更长
    preds = gezi.unique_list(preds)
    if len(preds) >= MIN_LEN2:
      if pre_type != 'Nothing':
        preds_list.append(' '.join(preds))
        types.append(pre_type)
  return types, preds_list

def get_pred_byend2(x):
  MIN_LEN = FLAGS.para_min_len  #2
  MIN_LEN2 = FLAGS.para_min_len2 #6
  NUM_CLASSES = len(id2dis)  
  pred = x['preds']
  probs = x['probs']
  end_prob = None
  if 'end_probs' in x:
    end_prob = x['end_probs']
  word_id_ = None
  pre_type = None
  # predictionstring list
  preds_list = []
  # store each pred word_id for one precitionstring
  preds = [] 
  pre_types = []
  pre_scores = np.zeros_like(probs[0])
  
  idxs = []
  types = []
  
  # got from ./gen-records.py --stats count freq of each discourse segment
  if pre_scores.shape[-1] > NUM_CLASSES:
    label_count_ratio = [
                  0.17623337623337623,
                  0.17623337623337623,
                  0.28704704704704703,
                  0.28704704704704703,
                  0.26125554125554123,
                  0.26125554125554123,
                  0.08812812812812813,
                  0.08812812812812813,
                  0.07625911625911626,
                  0.07625911625911626,
                  0.05322465322465322,
                  0.05322465322465322,
                  0.03318175318175318,
                  0.03318175318175318,
                  0.02467038467038467,
                  0.02467038467038467]
  else:
    label_count_ratio = [0.17623337623337623,
                      0.28704704704704703,
                      0.26125554125554123,
                      0.08812812812812813,
                      0.07625911625911626,
                      0.05322465322465322,
                      0.03318175318175318,
                      0.02467038467038467]
  
  label_count_ratio = np.array(label_count_ratio)
  # label_count_ratio[0] *= 10
  
  word_ids = x['word_ids']
  for i, word_id in enumerate(word_ids):
    if not is_wordid(word_id) :
      continue  
    
    if pre_scores.shape[-1] > NUM_CLASSES:
      pre_type = id2dis[int(np.argmax(probs[i]) / 2)]
    else:
      pre_type = id2dis[np.argmax(probs[i])]
    pre_types.append(pre_type)
    pre_scores += probs[i]
    preds.append(str(word_id))
    
    is_sep = False
    if end_prob is None:
      is_sep = pred[i] % 2 == 1
    else:
      is_sep = end_prob[i][1] > end_prob[i][0]
      
    if is_sep:
      if preds:
        pre_scores = gezi.softmax(pre_scores) * label_count_ratio  

        if pre_scores.shape[-1] > NUM_CLASSES:
          pre_type = id2dis[int(np.argmax(pre_scores) / 2)]
        else:
          pre_type = id2dis[np.argmax(pre_scores)]
        
        preds = gezi.unique_list(preds)
        if len(preds) < MIN_LEN:
          # 低置信度的干脆放弃召回
          continue
          pass
        else:
          if pre_type != 'Nothing':
            preds_list.append(' '.join(preds))
            types.append(pre_type)
        preds = []
        pre_types = []
        pre_scores = np.zeros_like(probs[0])
        
  if preds:
    pre_scores = gezi.softmax(pre_scores) * label_count_ratio
    if pre_scores.shape[-1] > NUM_CLASSES:
      pre_type = id2dis[int(np.argmax(pre_scores) / 2)]
    else:
      pre_type = id2dis[np.argmax(pre_scores)]
    
    # 结尾应该更长
    preds = gezi.unique_list(preds)
    if len(preds) >= MIN_LEN2:
      if pre_type != 'Nothing':
        preds_list.append(' '.join(preds))
        types.append(pre_type)
  return types, preds_list

def get_pred_byse2(x):
  MIN_LEN = FLAGS.para_min_len  #2
  MIN_LEN2 = FLAGS.para_min_len2 #6
  NUM_CLASSES = len(id2dis)
  pred = x['preds']
  total = len(pred)
  probs = x['probs']
  # probs = x['pred']
  start_prob = x['start_probs']
  end_prob = x['end_probs'] 
  word_id_ = None
  pre_type = None
  # predictionstring list
  preds_list = []
  # store each pred word_id for one precitionstring
  preds = [] 
  pre_types = []
  pre_scores = np.zeros_like(probs[0])
  
  idxs = []
  types = []
  label_count_ratio = [0.17623337623337623,
                    0.28704704704704703,
                    0.26125554125554123,
                    0.08812812812812813,
                    0.07625911625911626,
                    0.05322465322465322,
                    0.03318175318175318,
                    0.02467038467038467]
  
  label_count_ratio = np.array(label_count_ratio)
  # label_count_ratio[0] *= 10
  
  has_start = False
  
  for i in range(total):    
    is_start = start_prob[i][1] > start_prob[i][0]
    if is_start:
      has_start = True
    
    if has_start:    
      pre_type = id2dis[np.argmax(probs[i])]
      pre_types.append(pre_type)
      pre_scores += probs[i]
      preds.append(str(i))
      
    is_end = end_prob[i][1] > end_prob[i][0]
    if is_end:
      if preds:
        pre_scores = gezi.softmax(pre_scores) * label_count_ratio

        if pre_scores.shape[-1] > NUM_CLASSES:
          pre_type = id2dis[int(np.argmax(pre_scores) / 2)]
        else:
          pre_type = id2dis[np.argmax(pre_scores)]
        
        if len(preds) < MIN_LEN:
          # 低置信度的干脆放弃召回
          continue
          # pass
        else:
          if pre_type != 'Nothing':
            preds_list.append(' '.join(preds))
            types.append(pre_type)
        preds = []
        pre_types = []
        pre_scores = np.zeros_like(probs[0])
        has_start = False
        
  if preds:
    pre_scores = gezi.softmax(pre_scores) * label_count_ratio
    if pre_scores.shape[-1] > NUM_CLASSES:
      pre_type = id2dis[int(np.argmax(pre_scores) / 2)]
    else:
      pre_type = id2dis[np.argmax(pre_scores)]
    
    # 结尾应该更长
    if len(preds) >= MIN_LEN2:
      if pre_type != 'Nothing':
        preds_list.append(' '.join(preds))
        types.append(pre_type)
  return types, preds_list

def get_preds(x, post_adjust=True):  
  
  pred_fn = None
  if FLAGS.token2word:
    if FLAGS.pred_method == 'end':
      pred_fn = get_pred_byend
    elif FLAGS.pred_method == 'se':
      pred_fn = get_pred_byse
    elif FLAGS.pred_method == 'start':
      pred_fn = get_pred_bystart
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
  # ic(pred_fn)

  total = len(x['preds'])
  # with gezi.Timer('get_preds'):
  if not FLAGS.openmp:
    ids_list, types_list, preds_list = [], [], []
    for i in tqdm(range(total), desc='get_preds', leave=False):
      id = x['id'][i]
      x_ = {}
      for key in x: 
        x_[key] = x[key][i]
      types, preds = pred_fn(x_, post_adjust=post_adjust)
      ids_list.extend([id] * len(types))
      types_list.extend(types)
      preds_list.extend(preds)
  else:
    ids_list, types_list, preds_list = Manager().list(), Manager().list(), Manager().list()  
    nw = cpu_count()
    with pymp.Parallel(nw) as p:
      for i in p.range(total):
        id = x['id'][i]
        x_ = {}
        for key in x:
          x_[key] = x[key][i]
        types, preds = pred_fn(x_)
        ids = [id] * len(types)
        ids_list.extend([id] * len(types))
        types_list.extend(types)
        preds_list.extend(preds)
    ids_list, types_list, preds_list = list(ids_list), list(types_list), list(preds_list)
 
  m = {
    'id': ids_list,
    'class': types_list,
    'predictionstring': preds_list
  }

  df = pd.DataFrame(m)
    
  return df

def unk_aug(x, x_mask, rate=0.1, unk_id=1):
    """
    randomly make 10% words as unk
    """
    if mt.epoch() > 0:
      x_mask = x_mask.long()
      ratio = np.random.uniform(0, rate)
      mask = torch.cuda.FloatTensor(x.size(0), x.size(1)).uniform_() > ratio
      mask = mask.long()
      rmask = unk_id * (1 - mask)
      x = (x * mask + rmask) * x_mask
    return x

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

