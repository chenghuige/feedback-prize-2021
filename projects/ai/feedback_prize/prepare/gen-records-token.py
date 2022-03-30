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

import glob
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
flags.DEFINE_integer('buf_size', 100000, '')
flags.DEFINE_integer('num_records', None, '')
flags.DEFINE_integer('seed_', 1024, '')
flags.DEFINE_bool('stats', False, '')
flags.DEFINE_bool('stratified_split', False, '')
flags.DEFINE_alias('ss', 'stratified_split')
flags.DEFINE_integer('label_shift', 0, '')
flags.DEFINE_bool('ignore_short', False, '')
flags.DEFINE_float('inst_weight', 1., '')

# MAX_LEN = 2048
df = None
records_dir = None
ids = None
tokenizer = None

text_counts = defaultdict(int)
para_counts = defaultdict(int)
para_lens = defaultdict(int)

fes_dict = Manager().dict()

def get_count(label):
  count = 1
  if FLAGS.up_sample:
    max_cls = max(label)
    if max_cls >= 6:
      count = 3
    elif max_cls >= 3:
      count = 2
  return count

def deal_label(fe):
  idx = -1
  pre = None
  MAX_PARTS = FLAGS.max_parts
  MAX_LEN = FLAGS.max_len
  fe['para_type'] = [0] * MAX_PARTS
  fe['para_len'] = [0] * MAX_PARTS
  fe['para_mask'] = [0] * MAX_PARTS
  fe['para_index'] = [0] * MAX_LEN 
  
  fe['classes'] = [0] * NUM_CLASSES
  
  visited = set()
  for i in range(MAX_LEN):
    if not fe['mask'][i]:
      continue
    if fe['word_ids'][i] in visited:
      continue

    if fe['word_ids'][i] == 0 or fe['dis_start2'][i] or (i > 1 and fe['dis_end2'][i - 1]):
      idx += 1
      fe['start'][i] = 1
      fe['start2'][i] = 1
      if FLAGS.label_inside:
        j = i
        while (j + 1 < MAX_LEN) and fe['word_ids'][j + 1] == fe['word_ids'][i]:
          fe['start'][j + 1] = 1
          j += 1
      fe['para_type'][idx] = fe['label'][i]
      fe['para_len'][idx] = 1
      fe['para_mask'][idx] = 1 
      fe['classes'][fe['label'][i]] = 1
    else:
      fe['para_len'][idx] += 1
      
    fe['para_index'][i] = idx + 1
    fe['para_count'] = idx + 1
    visisted.add(fe['word_ids'][i])
    
  visited = set()
  for i in reversed(range(MAX_LEN)):
    if not fe['mask'][i]:
      continue
    if fe['word_ids'][i] in visited:
      continue
    
    if fe['word_ids'][i] == fe['num_words'] - 1 or fe['dis_end2'][i] or (i + 1 < MAX_LEN and fe['dis_start2'][i + 1]):
      fe['end'][i] = 1
      fe['end2'][i] = 1
      if FLAGS.label_inside:
        j = i
        while (j > 1) and fe['word_ids'][j - 1] == fe['word_ids'][i]:
          fe['end'][j - 1] = 1
          j -= 1
    visisted.add(fe['word_ids'][i])
  
  if FLAGS.stats:
    for i in range(fe['para_count']):
      para_counts[fe['para_type'][i]] += 1
      para_lens[fe['para_type'][i]] += fe['para_len'][i]
    for para_type in set(fe['para_type']):
      text_counts[para_type] += 1
      
  # ic(sum(fe['start']), sum(fe['start2']), sum(fe['end']), sum(fe['end2']), fe['para_count'])
  # if fe['para_count'] != sum(fe['start2']):
  #   ic(fe['id'], fe['para_count'], sum(fe['start2']), sum(fe['dis_start']))

def deal(index):
  if not FLAGS.stratified_split:
    df_ = df[df['id'].isin(set(ids[index]))]
  else:
    df_ = df[df['kfold'] == index]
  num_insts = len(df_)
  ofile = f'{records_dir}/{index}.tfrec'
  keys = []
  MAX_LEN = FLAGS.max_len
  fes = []
  with mt.tfrecords.Writer(ofile, buffer_size=FLAGS.buf_size, shuffle=True, seed=1024) as writer:
    id = None
    fe = {}
    for i, row in tqdm(enumerate(df_.itertuples()), total=num_insts, desc=ofile):
      row = row._asdict()
      
      if row['id'] != id:
        id = row['id']
        if fe:
          deal_label(fe)
          if not FLAGS.stats:
            if not (fe['words_covered_ratio'] == 1 and FLAGS.ignore_short):
              count = get_count(fe['label'])
              for _ in range(count):
                writer.write(fe)        
          fes.append({k: v for k, v in fe.items() if not isinstance(v, list)})
        fe = {}
        fe['weight'] = FLAGS.inst_weight
        fe['id'] = row['id']
        fe['label'] = [0] * MAX_LEN
        fe['dis_start'] = [0] * MAX_LEN
        fe['dis_end'] = [0] * MAX_LEN
        fe['start'] = [0] * MAX_LEN
        fe['end'] = [0] * MAX_LEN
        fe['dis_start2'] = [0] * MAX_LEN
        fe['dis_end2'] = [0] * MAX_LEN
        fe['start2'] = [0] * MAX_LEN
        fe['end2'] = [0] * MAX_LEN
        
        input_ids, attention_mask, word_ids = util.encode(row['text_'], tokenizer)
        
        # ic(row['text_'], word_ids, [row['text_'].split()[x] for x in word_ids if x is not None])
        # exit(0)
          
        fe['input_ids'] = input_ids
        fe['attention_mask'] = attention_mask
        fe['mask'] = [int(x != None) for x in word_ids]
        
        fe['word_ids'] = [util.null_wordid() if x is None else x for x in word_ids]
        # for debug
        fe['word_ids_str'] = ' '.join([str(x) for x in fe['word_ids']])

        fe['num_words'] = len(row['text_'].split())
        fe['num_tokens'] = sum(fe['attention_mask'])
        used_word_ids = [x for x in word_ids if x != None]
        fe['num_covered_words'] = len(set(used_word_ids))
        fe['num_covered_tokens'] = len(used_word_ids)
        fe['words_covered_ratio'] = fe['num_covered_words'] / fe['num_words']
        # ic(fe['words_covered_ratio'], fe['num_covered_words'],  fe['num_words'], fe['num_covered_words'] / fe['num_words'])
           
      preds = set([int(x) for x in row['predictionstring'].split()])
      first = True
      word_id_ = None
      last = None
      for j, word_id in enumerate(word_ids):
        if word_id in preds:
          if word_id != word_id_:
            word_id_ = word_id
          else:
            if FLAGS.label_inside:
              if j > 0:
                fe['dis_start'][j] = fe['dis_start'][j - 1]
              
          # 多个token对应一个word id
          if first:
            fe['dis_start'][j] = 1
            fe['dis_start2'][j] = 1
            first = False
          fe['label'][j] = row['discourse_type_id']
          last = j
          
      if last is not None:
        fe['dis_end'][last] = 1
        fe['dis_end2'][last] = 1
        if FLAGS.label_inside:
          j = last
          while j > 0 and word_ids[j - 1] == word_ids[last]:
            fe['dis_end'][j - 1] = 1
            j -= 1
    
    if fe:
      # input_ids, attention_mask, mask, label, start, start2, end, end2
      deal_label(fe)
      if not FLAGS.stats:
        if not (fe['words_covered_ratio'] == 1 and FLAGS.ignore_short):
          count = get_count(fe['label'])
          for _ in range(count):
            writer.write(fe)
      fes.append({k: v for k, v in fe.items() if not isinstance(v, list)})
      
  fes_dict[index] = fes
  if index == 0:
    print(fe)
    
  if FLAGS.stats:
    ic(para_counts, para_lens, text_counts)
    total_labels = sum(para_counts.values())
    total_lens = sum(para_lens.values())
    ic(total_labels, total_lens)
    para_count_ratio = [para_counts[i] / total_labels for i in range(len(all_classes))]
    para_len_ratio =  [para_lens[i] / total_lens for i in range(len(all_classes))]   
    total_texts = len(fes)
    text_count_ratio = [text_counts[i] / total_texts for i in range(len(all_classes))]
    ic(para_count_ratio, para_len_ratio, text_count_ratio)
     
          
def main(_):    
  FLAGS.records_type = 'token'
  config.init_()
  assert FLAGS.rv != 2, 'using gen-records2.py instead'
  
  global df, records_dir, ids, tokenizer
  np.random.seed(FLAGS.seed_)
  assert FLAGS.mark == 'train'
  ifile = f'{FLAGS.idir}/{FLAGS.mark}.fea'
  records_dir = f'{FLAGS.idir}/{FLAGS.records_name}/{FLAGS.mark}'
  if FLAGS.clear_first:
    command = f'rm -rf {records_dir}'
    gezi.system(command)    
  else:
    assert not glob.glob(f'{records_dir}/*.tfrec'), records_dir
  with gezi.Timer('read_csv'):
    df = pd.read_feather(ifile)
  
  # df = df.sort_values('id')
  # df = df.sample(1., random_state=1024)
  # ic(df)
  
  ic(FLAGS.backbone)
  tokenizer = util.get_tokenizer(FLAGS.backbone)
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
  d = d.sort_values(['id'])
  ic(d)
  d.to_csv(f'{FLAGS.idir}/{FLAGS.records_name}/records.csv', index=False)


if __name__ == '__main__':
  app.run(main)
