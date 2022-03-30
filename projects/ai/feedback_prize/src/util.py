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

import sys 
import os

from absl import app, flags
FLAGS = flags.FLAGS

import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Pool, Manager, cpu_count
import pymp
 
import pandas as pd

import tensorflow as tf
import torch
from torch.utils.data import DataLoader

import gezi
from gezi import tqdm
import melt as mt
import lele

from src.config import *
from src.torch.model import Model as TorchModel
from src.tf.model import Model as TFModel
from src.preprocess import preprocess
from src.torch.dataset import Dataset

# 注意deberta-v3和electra不需要也不能用下面 本意想把!!! ??? 分开 但是容易混杂
# 's 's 'm 's 't
def split_token_ids_bypunct(token_ids, tokenizer):
  token_ids_ = []
  special_token_ids = set([FLAGS.br, tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id])
  for token_id in token_ids:
    token = tokenizer.convert_ids_to_tokens(token_id)
    if token.startswith('Ġ') or token in special_token_ids or token.isalnum() or len(token) == 1:
      token_ids_.append(token_id)
    else:
      can_split = True
      ids = []
      for x in token:
        if x.isalnum():
          can_split = False
          break
        ids.append(tokenizer.convert_tokens_to_ids(x))
      if can_split:
        # print(token)
        token_ids_.extend(ids)
      else:
        token_ids_.append(token_id)
      # token_ids_.append(tokenizer.convert_tokens_to_ids(token[-1]))
  return token_ids_

def tokenize(words, tokenizer, null_id=None):
  encoding = {}
  ## 注意理论上应该是input_ids = [tokenizer.cls_token_id] 这样和使用fast tokenizer接口结果是一样的
  ## 保留之前的0 这样会最终处理变成[SEP] 开头 和之前的操作一致 可能这个操作线上效果更好？ TODO
  # input_ids = [tokenizer.cls_token_id]
  input_ids = [0]
  word_ids = [null_id]
  # eos_token_id = tokenizer.eos_token_id or tokenizer.cls_token_id + 1
  eos_token_id = tokenizer.sep_token_id
  for i, word in enumerate(words):
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if FLAGS.split_punct:
      token_ids = split_token_ids_bypunct(token_ids, tokenizer)
    input_ids.extend(token_ids)
    word_ids.extend([i] * len(token_ids))
  input_ids.append(eos_token_id)
  word_ids.append(null_id)
  encoding['input_ids'] = input_ids
  encoding['attention_mask'] = [1] * len(input_ids)
  encoding['word_ids'] = word_ids
  return encoding

# [SEP] or \n directly , bert模型可能词表没有\n [SEP]比较安全
# BR = '[SEP]'

def get_words(text):
  if FLAGS.remove_br:
    return text.split()
  # words = []
  # parts = text.split('\n')
  # for part in parts:
  #   if part:
  #     words.extend(part.split())
  #     words.append(FLAGS.br)
  #   else:
  #     words.append(FLAGS.br)
  # words = words[:-1]
  # assert len(words) - text.count('\n') == len(text.split())
  words = text.replace('\n', f' {FLAGS.br} ').split()
  if FLAGS.ori_br:
    words = [x if x != FLAGS.br else '\n' for x in words]
  return words

def is_br(word):
  return word == FLAGS.br or word == '\n'

def fix_wordids(word_ids, words, brs_map=None):
  if FLAGS.remove_br:
    return word_ids
  ignores = set()
  br_indexes = set()
  word_ids_ = word_ids.copy()
  for i, word_id in enumerate(word_ids): 
    if word_id is not None:
      if is_br(words[word_id]):
        # change \n word_id to None
        word_ids_[i] = None
        ignores.add(word_id)
        br_indexes.add(i)
        # 注意只追加一个如果后面连续两个换行 只用第一
        if FLAGS.merge_br == 'end':
          if i > 0 and (i - 1) not in br_indexes:
            word_ids_[i] = word_ids_[i - 1]
      else:
        if not brs_map:
          word_ids_[i] -= len(ignores)
        else:
          word_ids_[i] = brs_map[word_id]
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

def get_brs_map(words):
  brs_map = {None: None}
  ignores = set()
  for word_id, word in enumerate(words):
    if is_br(word):
      ignores.add(word_id)
      brs_map[word_id] = None
    else:
      brs_map[word_id] = word_id - len(ignores)
  return brs_map
    
def remap_wordids(word_ids, words=None, brs_map=None):
  if brs_map is None:
    brs_map = get_brs_map(words)
  word_ids_ = [brs_map[x] for x in word_ids]
  return word_ids_

# def split_inputs(tokenizer, input_ids, attention_mask, word_ids, relative_positions, split_method='start'):
def split_inputs(tokenizer, input_ids, attention_mask, word_ids, split_method='start'):
  start = 0
  last_tokens = 0
  max_len = FLAGS.max_len_valid or FLAGS.max_len
  if split_method == 'end':
    last_tokens = max_len
  elif split_method == 'mid':
    start = max(int((len(word_ids) - max_len) / 2), 0)
  elif split_method == 'se':
    last_tokens = FLAGS.last_tokens
  elif split_method == 'se2':
    last_tokens = FLAGS.last_tokens2
  elif split_method == 'se3':
    last_tokens = FLAGS.last_tokens3
  elif split_method == 'se4':
    last_tokens == FLAGS.last_tokens4
  elif split_method == 'se5':
    last_tokens == FLAGS.last_tokens5
  elif split_method == 'se6':
    last_tokens == FLAGS.last_tokens6
  elif split_method == 'se7':
    last_tokens == FLAGS.last_tokens7
  elif split_method.startswith('se'):
    last_tokens = int(split_method[2:])
        
  use_sep = True if FLAGS.add_special_tokens else False
  input_ids = gezi.pad(input_ids[start:], FLAGS.max_len, tokenizer.pad_token_id, 
                       last_tokens, use_sep=use_sep, sep_token=tokenizer.sep_token_id)
  attention_mask = gezi.pad(attention_mask[start:], FLAGS.max_len, 0, 
                            last_tokens, use_sep=use_sep, sep_token=1)
  word_ids = gezi.pad(word_ids[start:], FLAGS.max_len, null_wordid(), 
                      last_tokens, use_sep=use_sep, sep_token=null_wordid())
  # relative_positions = gezi.pad(relative_positions[start:], FLAGS.max_len, 0., 
  #                      last_tokens, use_sep=use_sep, sep_token=0)
  
  if FLAGS.add_special_tokens and FLAGS.force_special and (FLAGS.stride is None):
    ## 补充前sep
    if input_ids[0] != tokenizer.cls_token_id:
      input_ids[0] = tokenizer.sep_token_id
      word_ids[0] = null_wordid()
      # relative_positions[0] = 0.
    ## 补充后sep
    # for i in range(len(input_ids)):
    #   if input_ids[i] == tokenizer.pad_token_id:
    #     i -= 1
    #     break 
    # if input_ids[i] != tokenizer.eos_token_id:
    #   input_ids[i] = tokenizer.sep_token_id
    #   word_ids[i] = null_wordid()
    #   relative_positions[0] = 0.

    res = {
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'word_ids': word_ids,
      # 'relative_positions': relative_positions
    }
    return res

def get_word_ids(words, tokenizer):
  if tokenizer.is_fast and (not FLAGS.split_punct) and (not FLAGS.custom_tokenize):
    encoding = tokenizer(words, is_split_into_words=True, truncation=False, add_special_tokens=FLAGS.add_special_tokens)
    word_ids = encoding.word_ids()
  else:
    encoding = tokenize(words, tokenizer)
    word_ids = encoding['word_ids']  
  return encoding, word_ids

# used by gen-records.py and get_inputs for infer
def encode(text, tokenizer, split_method=None, multi_inputs=None):
  split_method = split_method if split_method is not None else FLAGS.split_method
  multi_inputs = multi_inputs if multi_inputs is not None else FLAGS.multi_inputs
  if FLAGS.lower:
    text = text.lower()
  words = get_words(text)   
  # words = [preprocess(x) for x in words]
  
  if FLAGS.merge_tokens:
    words = words[:FLAGS.max_words]
  
  if FLAGS.stride is not None:
    assert not FLAGS.custom_tokenize
    encoded = tokenizer(words,
                        is_split_into_words=True,
                        return_overflowing_tokens=True,
                        stride=FLAGS.stride,
                        max_length=FLAGS.max_len,
                        padding="max_length",
                        truncation=True)
    n = len(encoded['overflow_to_sample_mapping'])
    res = {}
    brs_map = get_brs_map(words)
    for i in range(n):
      res[i] = {}
      input_ids = encoded['input_ids'][i]
      if i > 0:
        start_id = tokenizer.convert_tokens_to_ids(f'[START{i}]') if FLAGS.use_stride_id else tokenizer.sep_token_id
        input_ids[0] = start_id
      if i != n - 1:
        total = sum(encoded['attention_mask'][i])
        if total < len(input_ids):
          end_id = tokenizer.convert_tokens_to_ids(f'[END{i}]') if FLAGS.use_stride_id else tokenizer.sep_token_id
          input_ids[total] = end_id
      res[i]['input_ids'] = input_ids
      res[i]['attention_mask'] = encoded['attention_mask'][i]
      word_ids = fix_wordids(encoded.word_ids(i), words, brs_map)
      res[i]['word_ids'] = [null_wordid() if x is None else x for x in word_ids] 
      assert max(res[i]['word_ids']) < len(words), text
    return res  
  
  encoding, word_ids = get_word_ids(words, tokenizer)
    
  # num_words = len(words)
  # [0,1) + 1
  word_ids = fix_wordids(word_ids, words)
  # assert(max([x for x in word_ids if x is not None]) == len(text.split()) - 1), len(text.split())
  assert(max([x for x in word_ids if x is not None]) < len(text.split())), "try adding --br='[SEP]' or --br='[unused0]'"
  assert(min([x for x in word_ids if x is not None]) >= 0)
  # assert(min([x for x in word_ids if x is not None]) == 0)
  input_ids = encoding['input_ids']
  attention_mask = encoding['attention_mask']
  word_ids = [null_wordid() if x is None else x for x in word_ids]
  # relative_positions = [0. if x is None else 1. + x / num_words for x in word_ids]
   
  res = {}
  split_methods = [split_method]
  if FLAGS.multi_inputs:
    split_methods += FLAGS.multi_inputs_srcs
  
  for i, split_method in enumerate(split_methods):
    res[i] = {}
    res[i].update(split_inputs(tokenizer, 
                                input_ids, 
                                attention_mask, 
                                word_ids, 
                                # relative_positions,
                                split_method=split_method))
  if multi_inputs:
    return res
  else:
    return res[0]        

def get_model():
  if not FLAGS.torch:
    model = TFModel().build_model()
  else:
    model = TorchModel()
  return model

# for inference
def get_inputs(backbone, mode='test', sort=True, double_times=0, split_method=None, multi_inputs=None, test_ids=None, df=None):
  split_method = split_method if split_method is not None else FLAGS.split_method
  multi_inputs = multi_inputs if multi_inputs is not None else FLAGS.multi_inputs

  #--- for online infer
  FLAGS.remove_br = False
  tokenizer = get_tokenizer(backbone)
  files = os.listdir(f'../input/feedback-prize-2021/{mode}')
  if df is None:
    test_ids = [f.replace('.txt','') for f in files if 'txt' in f] if test_ids is None else test_ids
    ic(len(test_ids), torch.cuda.is_available(), tokenizer.padding_side)
    if len(test_ids) < 10 and torch.cuda.is_available() and double_times:
      for _ in range(double_times):
        test_ids += test_ids
  else:
    test_ids = list(df.essay_id)
  ic(len(test_ids))
  MAX_LEN = FLAGS.max_len
  num_words_list = []
  
  inputs = {}
  ids, num_words = [], []
  for i, tid in tqdm(enumerate(test_ids), total=len(test_ids)):
    if df is None:
      name = f'../input/feedback-prize-2021/{mode}/{tid}.txt'
      text = open(name, 'r').read()
    else:
      text = df.iloc[i].essay
    res = encode(text, tokenizer, split_method=split_method, multi_inputs=multi_inputs) 
    num_words.append(len(text.split()))
    ids.append(tid)
    if 0 in res:
      if FLAGS.stride is None:
        for idx in res:
          if idx not in inputs:
            inputs[idx] = {}
          for key in res[idx]:
            if key not in inputs[idx]:
              inputs[idx][key] = [res[idx][key]]
            else:
              inputs[idx][key].append(res[idx][key])
      else:
        for idx in res:
          for key in res[idx]:
            if key not in inputs:
              inputs[key] = [res[idx][key]]
            else:
              inputs[key].append(res[idx][key])
          if idx > 0:
            ids.append(tid)
            num_words.append(len(text.split()))
    else:
      for key in res:
        if key not in inputs:
          inputs[key] = [res[key]]
        else:
          inputs[key].append(res[key])
          
  if 0 in inputs:
    for idx in inputs:
      inputs[idx].update({
        'id': ids,
      'num_words': num_words
      })
      # inputs[idx]['num_tokens'] = [x.sum() for x in  inputs[idx]['attention_mask']]
  else:
    inputs.update({
      'id': ids,
      'num_words': num_words
    })
    # inputs['num_tokens'] = [x.sum() for x in  inputs['attention_mask']]
    
  if 0 in inputs:
    for idx in inputs:
      for key in inputs[idx]:
        inputs[idx][key] = np.asarray(inputs[idx][key])
  else:
    for key in inputs:
      inputs[key] = np.asarray(inputs[key])    
    
  if sort:
    # inds = np.asarray(TEST_IDS).argsort()
    inds = inputs[0]['num_words'].argsort() if 0 in inputs else inputs['num_words'].argsort()
    if 0 in inputs:
      for idx in inputs:
        for key in inputs[idx]:
          inputs[idx][key] = np.asarray([inputs[idx][key][idx_] for idx_ in inds])
    else:
      for key in inputs:
        inputs[key] = np.asarray([inputs[key][idx_] for idx_ in inds])
      
  if 0 in inputs:
    inputs['id'] = inputs[0]['id']
    inputs['num_words'] = inputs[0]['num_words']
    inputs['word_ids'] = inputs[0]['word_ids']
  
  # ic(inputs)
  return inputs

def null_wordid():
  return -1

def is_wordid(x):
  return x != null_wordid() 
  
def preds_token2word(preds, word_ids, num_words, reduce_method='first'):
  word_id_ = -1
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

  if not FLAGS.merge_tokens:
    for i, word_id in enumerate(word_ids):
      if not is_wordid(word_id):
        continue
      # if FLAGS.split_method != 'start' and i <= 512:
      #   continue
      if word_id != word_id_:
        # # 如果是起始word word_id_ == -1 不计入start概率信息，另外SE模式 会造成 word_id > word_id_ + 1 这个时候也是起始位置 不计入start概率信息
        # if (word_id == -1 or word_id > word_id_ + 1)  and preds.shape[-1] == 2:
        #   word_id_ = word_id
        #   continue
        if preds_:
          probs[word_id_] = get_pred(preds_)
          preds_ = []
        word_id_ = word_id
    
      preds_.append(preds[i])
      
    if preds_:
      probs[word_id_] = get_pred(preds_)
  else:
    for i in range(num_words):
      if i < len(preds):
        probs[i] = preds[i]    
  return probs

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

def merge_res(x):
  if FLAGS.stride is None:
    return x
  x_ = {}
  for key in x:
    x_[key] = []
  m = defaultdict(list)
  if FLAGS.stride_combiner == 'mean':
    weights = {}
  inds = np.asarray(x['id']).argsort()
  for key in x:
    try:
      x[key] = x[key][inds]
    except Exception:
      x[key] = [x[key][idx] for idx in inds]
  for i, id in enumerate(x['id']):
    if FLAGS.stride_combiner == 'mean':
      weight = (np.absolute(x['pred'][i]).sum(-1, keepdims=True) > 0).astype(int)
    if id not in m:
      for key in x:
        x_[key].append(x[key][i])
      if FLAGS.stride_combiner == 'mean':
        weights[id] = weight
    else:
      x_['pred'][-1] += x['pred'][i]
      x_['start_logits'][-1] += x['start_logits'][i]
      if FLAGS.stride_combiner == 'mean':
        weights[id] += weight
    m[id].append(i)
  # TODo 还有一种是按照实际贡献平均
  if FLAGS.stride_combiner == 'divide':
    for i in range(len(x_['id'])):
      id = x_['id'][i]
      x_['pred'][i] /= len(m[id])
  elif FLAGS.stride_combiner == 'mean':
    for i in range(len(x_['id'])):
      id = x_['id'][i]
      x_['pred'][i] /= np.clip(weights[id], 1, None)
  
  for key in x:
    x[key] = x_[key]

def token2word(x):  
  total = len(x['pred'])
  if FLAGS.records_type == 'token':
    reduce_method = FLAGS.post_reduce_method or 'avg'
    x['pred'] = [preds_token2word(x['pred'][i], x['word_ids'][i], reduce_method=reduce_method, num_words=x['num_words'][i]) for i in tqdm(range(total), desc='pred_t2w', leave=False)]
  if 'start_logits' in x  and (FLAGS.pred_method != 'end'):
    if FLAGS.records_type == 'token':
      reduce_method2 = FLAGS.post_reduce_method2 or 'first'
      x['start_logits'] = [preds_token2word(x['start_logits'][i], x['word_ids'][i], reduce_method=reduce_method2, num_words=x['num_words'][i]) for i in tqdm(range(total), desc='start_t2w', leave=False)]
    x['start_probs'] = [gezi.softmax(start_logit) for start_logit in x['start_logits']]  
  if 'end_logits' in x  and (FLAGS.pred_method != 'start'):
    if FLAGS.records_type == 'token':
      reduce_method2 = FLAGS.post_reduce_method2 or 'first'
      x['end_logits'] = [preds_token2word(x['end_logits'][i], x['word_ids'][i], reduce_method=reduce_method2, num_words=x['num_words'][i]) for i in tqdm(range(total), desc='end_t2w', leave=False)]
  
  for cls_ in classes:
    if f'{cls_}_logits' in x:
      reduce_method2 = FLAGS.post_reduce_method2 or 'first'
      x[f'{cls_}_logits'] = [preds_token2word(x[f'{cls_}_logits'][i], x['word_ids'][i], reduce_method=reduce_method2, num_words=x['num_words'][i]) for i in tqdm(range(total), desc=f'{cls_}_t2w', leave=False)]
      x[f'{cls_}_probs'] = [gezi.softmax(logit) for logit in x[f'{cls_}_logits']]  
      
  if 'cls_logits' in x:
     x['cls_probs'] = gezi.sigmoid(x['cls_logits'])

  merge_res(x)
  x['probs'] = [gezi.softmax(pred) for pred in x['pred']] 
  x['preds'] = [np.argmax(prob, axis=-1) for prob in x['probs']]
  if 'start_logits' in x:
    x['start_probs'] = [gezi.softmax(start_logit) for start_logit in x['start_logits']] 
  if 'end_logits' in x:
    x['end_probs'] = [gezi.softmax(end_logit) for end_logit in x['end_logits']] 
  
convert_res = token2word    

class Ensembler(object):
  def __init__(self, need_sort=False):
    self.x = None
    self.need_sort = need_sort
    self.weights = []
    self.total_weight = 0
        
  def add(self, x, weight=1., weights=None):
    # assert weight > 0, weight
    # ic(weight, weights)
    self.total_weight += weight
    if self.need_sort:
      inds = np.asarray(x['id']).argsort()
      for key in x:
        try:
          x[key] = x[key][inds]
        except Exception:
          # ic(key)
          x[key] = [x[key][idx] for idx in inds]
          
    def get_weights(weights, num_words):
      if num_words <= 400:
        return weights[0]
      if len(weights) == 2:
        return weights[1]
      if num_words <= 800:
        return weights[1]
      return weights[2]
    
    if weights is None:
      if FLAGS.ensemble_weight_per_word:
        weights = [(np.absolute(pred).sum(-1, keepdims=True) > 0).astype(float) * weight for pred in x['pred']]
      else:
        weights = [weight] * len(x['pred'])
    else:
      if FLAGS.ensemble_weight_per_word:
        weights = [(np.absolute(pred).sum(-1, keepdims=True) > 0).astype(float) * get_weights(weights, num_words) for pred, num_words in zip(x['pred'], x['num_words'])]
      else:
        weights = [get_weights(weights, num_words) for num_words in x['num_words']] 
        
    if not self.weights:
      self.weights = weights
    else:
      self.weights = [x + y for x, y in zip(self.weights, weights)]
 
    if self.x is None: 
      self.x = x
      self.x.update({
          'pred': [pred * weight for pred, weight in zip(x['pred'], weights)],
          'start_logits': [logit * weight for logit, weight in zip(x['start_logits'], weights)],
          'probs': [prob * weight for prob, weight in zip(x['probs'], weights)],
          'start_probs': [prob * weight for prob, weight in zip(x['start_probs'], weights)],
        })
      if 'cls_logits' in x:
        self.x.update({'cls_logits': [logit * weight for logit, weight in zip(x['cls_logits'], weights)]})
      for cls_ in classes:
        if f'{cls_}_logits' in x:
          self.x.update({f'{cls_}_logits': [logit * weight for logit, weight in zip(x[f'{cls_}_logits'], weights)]})
    else:
      # for i in range(len(x['pred'])):
      #   if x['num_words'][i] > 1024:
      #     ic(x['pred'][i], self.x['pred'][i])
      #     ic(x['start_logits'][i], self.x['start_logits'][i])
      #     exit(0)
      # try:
      x_ = {
        'pred': [pred1 + pred2 * weight for pred1, pred2, weight in zip(self.x['pred'], x['pred'], weights)],
        'start_logits': [pred1 + pred2 * weight for pred1, pred2, weight in zip(self.x['start_logits'], x['start_logits'], weights)], 
        'probs': [prob1 + prob2 * weight for prob1, prob2, weight in zip(self.x['probs'], x['probs'], weights)],
        'start_probs': [prob1 + prob2 * weight for prob1, prob2, weight in zip(self.x['start_probs'], x['start_probs'], weights)], 
      }
      # except Exception as e:
      #   ic(e)
      #   for i, (pred1, pred2) in enumerate(zip(self.x['pred'], x['pred'])):
      #     try:
      #       pred = pred1 + pred2
      #     except Exception:
      #       ic(i, self.x['pred'][i].shape, x['pred'][i].shape)
      self.x.update(x_)
      if 'cls_logits' in x:
        x_ = {'cls_logits': [pred1 + pred2 * weight for pred1, pred2, weight in zip(self.x['cls_logits'], x['cls_logits'], weights)]}
        self.x.update(x_)
      for cls_ in classes:
        if f'{cls_}_logits' in x:
          x_ = {f'{cls_}_logits': [pred1 + pred2 * weight for pred1, pred2, weight in zip(self.x[f'{cls_}_logits'], x[f'{cls_}_logits'], weights)]}
          self.x.update(x_)
          
  def finalize(self):
    x = self.x
    
    weight = self.total_weight
    if not self.weights:
      self.weights = [weight] * len(x['pred'])
    else:
      self.weights = [np.clip(weight, 1., None) for weight in self.weights]
    
    # assert weight > 0, weight
    x['pred'] =  [pred / weight for pred, weight in zip(x['pred'], self.weights)]
    if FLAGS.ensemble_method == 'logit':
      x['probs'] = [gezi.softmax(pred) for pred in x['pred']]
    else:
      x['probs'] =  [prob / weight for prob, weight in zip(x['probs'], self.weights)]
    x['preds'] = [np.argmax(prob, axis=-1) for prob in x['probs']]
    # del x['pred']
    x['start_logits'] = [logit / weight for logit, weight in zip(x['start_logits'], self.weights)]
    if FLAGS.ensemble_method == 'logit':
      x['start_probs'] = [gezi.softmax(logit) for logit in x['start_logits']]
    else:
      x['start_probs'] = [prob / weight for prob, weight in zip(x['start_probs'], self.weights)]
    if 'cls_logits' in x:
      x['cls_logits'] = [logit / weight for logit, weight in zip(x['cls_logits'], self.weights)]
      x['cls_probs'] = [gezi.sigmoid(logit) for logit in x['cls_logits']]
    return x
  
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
  
# dataloaders
def get_files_list(epoch=0, mark='train'):
  files_list = []
  if FLAGS.records_names:
    for records_name in FLAGS.records_names:
      # if '_ep' in records_name:
      #   FLAGS.dataset_per_epoch = True
      records_pattern = f'{FLAGS.idir}/{records_name}/train/*.tfrec'
      if epoch:
        records_pattern = records_pattern.replace('_ep0', f'_ep{epoch}')
      ic(records_pattern)
      files = gezi.list_files(records_pattern)
      
      if mark == 'train':
        if not FLAGS.online:
          files_ = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds != FLAGS.fold]
        else:
          files_ = files
      else:
        files_ = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]
      ic(files_[:2])
      if files_:
        files_list.append(files_)
  return files_list

def get_dataloaders():
  train_files_list = get_files_list()
  train_ds = Dataset(FLAGS.train_files, 'train', train_files_list, indexes=FLAGS.dataset_indexes)
  valid_files_list = [] if not FLAGS.multi_inputs else get_files_list(mark='valid')
  valid_ds = Dataset(FLAGS.valid_files, 'valid', valid_files_list)
  valid_ds2 = Dataset(FLAGS.valid_files, 'valid')
  sampler = None if FLAGS.sampler is None else lele.ImbalancedDatasetSampler(train_ds)
  num_workers = 8
  kwargs = {'num_workers': num_workers, 'pin_memory': True}  
  train_dl = DataLoader(train_ds, mt.batch_size(), shuffle=True, **kwargs)

  eval_dl = DataLoader(valid_ds, mt.eval_batch_size(), shuffle=False, **kwargs)
  valid_dl = DataLoader(valid_ds2, mt.batch_size(), shuffle=False, **kwargs)
  return train_dl, eval_dl, valid_dl

# 每轮设置一次dataloader 有比较大概率hang TODO
def get_dataset(epoch):
  assert FLAGS.records_names and FLAGS.dataset_per_epoch
  train_files_list = get_files_list(epoch)
  train_ds = Dataset(FLAGS.train_files, 'train', train_files_list, indexes=FLAGS.dataset_indexes)
  train_dl = DataLoader(train_ds, mt.batch_size(), shuffle=True, **kwargs)
  return train_dl
