#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   csv2records.py
#        \author   chenghuige  
#          \date   2020-04-12 17:56:50.100557
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
import json

from absl import app, flags
FLAGS = flags.FLAGS

import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm 
from sklearn.utils import shuffle
from collections import defaultdict
import itertools
import swifter

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import AutoTokenizer

import tensorflow as tf

import gezi
import melt as mt

from src import config
from src.config import *
from src import util

flags.DEFINE_string('mark', 'train', 'train')
flags.DEFINE_integer('buf_size', 1000, '')
flags.DEFINE_integer('num_records', None, '')
flags.DEFINE_integer('seed_', 1024, '')
flags.DEFINE_bool('stats', False, '')

"""
废弃不用 按word 麻烦速度慢效果差 有bug但是理论上这样只是处理方式不同 不会带来更好效果 不方便维护， 使用gen-records-token.py
"""

# MAX_LEN = 2048
df = None
records_dir = None
ids = None
tokenizer = None

label_counts = defaultdict(int)
label_lens = defaultdict(int)

fes_dict = Manager().dict()

def deal_label(fe):
  idx = -1
  pre = None
  MAX_PARTS = FLAGS.max_parts
  MAX_LEN = FLAGS.max_len
  fe['para_type'] = [0] * MAX_PARTS
  fe['para_len'] = [0] * MAX_PARTS
  fe['para_mask'] = [0] * MAX_PARTS
  fe['para_index'] = [0] * MAX_LEN 
  last = None
  for i in range(MAX_LEN):
    if not fe['mask'][i]:
      continue
    last = i
    if fe['label'][i] != pre or fe['dis_start'][i]:
      idx += 1
      if fe['dis_start'][i] or pre or fe['word_ids'][i] == 0:
        fe['start'][i] = 1
        fe['start2'][i] = 1
      fe['para_type'][idx] = fe['label'][i]
      fe['para_len'][idx] = 1
      fe['para_mask'][idx] = 1 
      pre = fe['label'][i]
    else:
      fe['para_len'][idx] += 1
    fe['para_index'][i] = idx + 1
  
  # 截断暂时不认为是end 这样 sum(end) 可能比parts数目少一个
  if last != MAX_LEN - 1:  
    fe['end'][last] = 1
    fe['end2'][last] = 1
  fe['para_count'] = idx + 1
  
  if FLAGS.stats:
    for i in range(fe['para_parts']):
      label_counts[fe['para_type'][i]] += 1
      label_lens[fe['para_type'][i]] += fe['para_len'][i]
      
  assert fe['para_count'] == sum(fe['start'])

def deal(index):
  df_ = df[df['id'].isin(set(ids[index]))]
  num_insts = len(df_)
  ofile = f'{records_dir}/{index}.tfrec'
  keys = []
  MAX_LEN = FLAGS.max_len
  fes = []
  with mt.tfrecords.Writer(ofile, buffer_size=FLAGS.buf_size) as writer:
    id = None
    fe = {}
    for i, row in tqdm(enumerate(df_.itertuples()), total=num_insts, desc=ofile):
      row = row._asdict()
      
      if row['id'] != id:
        id = row['id']
        if fe:
          deal_label(fe)
          if not FLAGS.stats:
            writer.write(fe)        
          fes.append({k: v for k, v in fe.items() if not isinstance(v, list)})
        fe = {}
        fe['id'] = row['id']
        fe['label'] = [0] * MAX_LEN
        fe['dis_start'] = [0] * MAX_LEN
        fe['dis_end'] = [0] * MAX_LEN
        fe['start'] = [0] * MAX_LEN
        fe['end'] = [0] * MAX_LEN
        # compat with gen records token based
        fe['dis_start2'] = [0] * MAX_LEN
        fe['dis_end2'] = [0] * MAX_LEN
        fe['start2'] = [0] * MAX_LEN
        fe['end2'] = [0] * MAX_LEN
        # words = row['text'].split()
        input_ids, attention_mask, word_ids, relative_positions = util.encode(row['text'], tokenizer)

        # if word_ids_ != word_ids:
        #   print(list(zip(word_ids_, word_ids)))
          
        # for per token
        fe['input_ids'] = input_ids
        fe['attention_mask'] = attention_mask
        fe['word_ids'] = word_ids

        # some stats
        fe['num_words'] = len(row['text_'].split())
        fe['num_tokens'] = sum(fe['attention_mask'])
        used_word_ids = [x for x in word_ids if x != util.null_wordid()]
        fe['num_covered_words'] = len(set(used_word_ids))
        assert fe['num_covered_words'] > 0
        fe['num_covered_tokens'] = len(used_word_ids)
        fe['words_covered_ratio'] = fe['num_covered_words'] / fe['num_words']
        
        # mask is for calc loss label mask
        fe['mask'] = [1] * fe['num_covered_words'] + [0] * (MAX_LEN - fe['num_covered_words'])
        fe['start_mask'] = fe['mask'].clone()
        fe['start_mask'][0] = 0
      
      pred_ids = [int(x) for x in row['predictionstring'].split()]
      # 注意有可能截断 或者整个都不在 因为限制了max len
      start = True
      last = None
      for word_id in pred_ids:
        if word_id >= MAX_LEN:
          last = None
          break
        last = word_id
        if start:
          fe['dis_start'][word_id] = 1 
          fe['dis_start2'][word_id] = 1 
          start = False
        fe['label'][word_id] = row['discourse_type_id']
          
      if last is not None:
        fe['dis_end'][last] = 1
        fe['dis_end2'][last] = 1

    if fe:
      deal_label(fe)
      if not FLAGS.stats:
        writer.write(fe)
      fes.append({k: v for k, v in fe.items() if not isinstance(v, list)})
      
  fes_dict[index] = fes
  if index == 0:
    print(fe)
    
  if FLAGS.stats:
    ic(label_counts, label_lens)
    total_labels = sum(label_counts.values())
    total_lens = sum(label_lens.values())
    ic(total_labels, total_lens)
    label_count_ratio = [label_counts[i] / total_labels for i in range(len(all_classes))]
    label_len_ratio =  [label_lens[i] / total_lens for i in range(len(all_classes))]  
    ic(label_count_ratio, label_len_ratio)
     
          
def main(_):    
  FLAGS.records_type = 'word'
  config.init_()
  
  global df, records_dir, ids, tokenizer
  np.random.seed(FLAGS.seed_)
  assert FLAGS.mark == 'train'
  ifile = f'{FLAGS.idir}/{FLAGS.mark}.fea'
  records_dir = f'{FLAGS.idir}/{FLAGS.records_name}/{FLAGS.mark}'
  with gezi.Timer('read_csv'):
    df = pd.read_feather(ifile)
  ic(df)
  
  tokenizer = AutoTokenizer.from_pretrained(FLAGS.backbone, add_prefix_space=True)
  ic(tokenizer)
    
  num_records = FLAGS.num_records or cpu_count()
  if num_records > FLAGS.folds:
    num_records = int(num_records / FLAGS.folds) * FLAGS.folds
  FLAGS.num_records = num_records
  if FLAGS.stats:
    num_records = 1
  ic(num_records)
  ids = gezi.unique_list(df['id'])
  np.random.shuffle(ids)
  # ic(ids)
  ids = np.array_split(ids, num_records)
  ic(ids[-1][:3])
  
  if num_records > 1:
    with Pool(num_records) as p:
      p.map(deal, range(num_records))
  else:
    deal(0)
    
  ic(FLAGS.mark, records_dir, mt.get_num_records_from_dir(records_dir))

  fes = list(itertools.chain(*fes_dict.values()))
  d = pd.DataFrame(fes)
  ic(d)
  d.to_csv(f'{FLAGS.idir}/{FLAGS.records_name}/records.csv', index=False)


if __name__ == '__main__':
  app.run(main)
