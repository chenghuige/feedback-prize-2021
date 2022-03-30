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

import string
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm 
from sklearn.utils import shuffle
from collections import defaultdict
import itertools
import swifter

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
# flags.DEFINE_bool('stratified_split', True, '')
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

def get_count(classes):
  count = 1
  if FLAGS.up_sample:
    if classes[-1] == 1 or classes[-2] == 1:
      count = 2
  return count

def deal_label(fe, words):
  idx = 0
  pre = None
  MAX_PARTS = FLAGS.max_parts
  MAX_LEN = FLAGS.max_len
  # fe['para_type'] = [0] * MAX_PARTS
  # fe['para_len'] = [0] * MAX_PARTS
  # fe['para_mask'] = [0] * MAX_PARTS
  # fe['para_index'] = [0] * MAX_LEN 
  
  # fe['classes'] = [0] * NUM_CLASSES
  
  visited = set()
  prev_word = None
  prev_idx = None
  for i in range(MAX_LEN):
    if not fe['mask'][i]:
      continue
    if fe['word_ids'][i] in visited:
      continue
    
    if (fe['word_ids'][i] != 0) and fe['dis_start2'][i] or (i > 1 and fe['dis_end2'][i - 1]):
      fe['start'][i] = 1
      fe['start2'][i] = 1
      fe['word_start'][fe['word_ids'][i]] = 1
      if FLAGS.mask_more:
        if prev_word:
          if not any(prev_word.endswith(x) for x in string.punctuation):
            fe['mask'][prev_idx] = 0
            fe['word_mask'][fe['word_ids'][prev_idx]] = 0

    visited.add(fe['word_ids'][i])
    prev_word = words[fe['word_ids'][i]]
    prev_idx = i
  
  visited = set()
  for i in reversed(range(MAX_LEN)):
    if not fe['mask'][i]:
      continue
    if fe['word_ids'][i] in visited:
      continue
    if (fe['word_ids'][i] != fe['num_words'] - 1) and fe['dis_end2'][i] or (i + 1 < MAX_LEN and fe['dis_start2'][i + 1]):
      fe['end'][i] = 1
      fe['end2'][i] = 1
      fe['word_end'][fe['word_ids'][i]] = 1
    visited.add(fe['word_ids'][i])

  if FLAGS.stats:
    for i in range(fe['para_count']):
      para_counts[fe['para_type'][i]] += 1
      para_lens[fe['para_type'][i]] += fe['para_len'][i]
    for para_type in set(fe['para_type']):
      text_counts[para_type] += 1
        
  # ic(fe['para_type'], fe['classes'], fe['label'])
  # exit(0)
      
  # ic(sum(fe['start']), sum(fe['start2']), sum(fe['end']), sum(fe['end2']), fe['para_count'])
  # if fe['para_count'] != sum(fe['start2']):
  #   ic(fe['id'], fe['para_count'], sum(fe['start2']), sum(fe['dis_start']))

def deal(index):
  # if not FLAGS.stratified_split:
  #   df_ = df[df['id'].isin(set(ids[index]))]
  # else:
  #   # TODO this will make only 5 workers ...so a bit slow
  #   # df_ = df[df['kfold'] == index]
  df_ = df[df['worker'] == index]
  num_insts = len(df_)
  ofile = f'{records_dir}/{index}.tfrec'
  keys = []
  MAX_LEN = FLAGS.max_len
  MAX_WORDS = FLAGS.max_words
  MAX_PARTS = FLAGS.max_parts
  NUM_CLASSES = FLAGS.num_classes
  fes = []
  with mt.tfrecords.Writer(ofile, buffer_size=FLAGS.buf_size, shuffle=True, seed=1024) as writer:
    for i, row in tqdm(enumerate(df_.itertuples()), total=num_insts, desc=ofile, leave=False):
      row = row._asdict()
      fe = {}
      fe['fold'] = row['kfold']
      assert fe['fold'] == index % FLAGS.folds
      fe['cluster'] = row['cluster']
      cluster = fe['cluster']
      # fe['cluster_id'] = tokenizer.convert_tokens_to_ids(f'[CLUSTER{cluster}]')
      fe['weight'] = FLAGS.inst_weight
      fe['id'] = row['id']        
      row['text'] = row['text'].replace(FLAGS.br, '\n')
      words = row['text'].split()
      fe['num_words'] = len(words)

      encode_res = util.encode(row['text'], tokenizer)
      if not 0 in encode_res:
        encode_res = {0: encode_res}

      starts = row['start']
      ends = row['end']
      
      para_types = row['para_type'].copy()
      fe['classes'] = [0] * NUM_CLASSES
      fe['para_type'] = para_types
      fe['para_mask'] = [1] * len(para_types)
      fe['para_len'] = []
      fe['para_start'] = starts.copy()
      for start, end in zip(starts, ends):
        fe['para_len'].append(end - start)
      for cls_ in fe['para_type']:
        fe['classes'][cls_] = 1
      for i in range(len(fe['para_type'])):
        fe['para_mask'][i] = 1
      fe['para_type'] = gezi.pad(fe['para_type'], MAX_PARTS)
      fe['para_mask'] = gezi.pad(fe['para_mask'], MAX_PARTS)
      fe['para_len'] = gezi.pad(fe['para_len'], MAX_PARTS)
      fe['para_start'] = gezi.pad(fe['para_start'], MAX_PARTS)
  
      for idx in encode_res:
        fe['label'] = [0] * MAX_LEN
        fe['word_label'] = [0] * MAX_WORDS
        fe['dis_start'] = [0] * MAX_LEN
        fe['dis_end'] = [0] * MAX_LEN
        fe['start'] = [0.] * MAX_LEN
        fe['end'] = [0] * MAX_LEN
        fe['word_dis_start'] = [0] * MAX_WORDS
        fe['word_dis_end'] = [0] * MAX_WORDS
        fe['word_start'] = [0.] * MAX_WORDS
        fe['word_end'] = [0] * MAX_WORDS
        fe['dis_start2'] = [0] * MAX_LEN
        fe['dis_end2'] = [0] * MAX_LEN
        fe['start2'] = [0] * MAX_LEN
        fe['end2'] = [0] * MAX_LEN
        encoded = encode_res[idx]
        # ic(encoded)
        fe.update(encoded)
      
        fe['mask'] = [int(x != util.null_wordid()) for x in fe['word_ids']]
        
        fe['start_mask'] = [0 if x <= 0 else 1 for x in fe['word_ids']]
        for i in range(len(fe['word_ids'])):
          fe['start_mask'][i] = 0
          if fe['word_ids'][i] != util.null_wordid():
            break
        
        fe['end_mask'] = [0 if x < 0 else 1 for x in fe['word_ids']]
        for i in reversed(range(len(fe['word_ids']))):
          fe['end_mask'][i] = 0
          if fe['word_ids'][i] != util.null_wordid():
            break
        
        assert max(fe['word_ids']) < fe['num_words'], fe['id']
        fe['num_tokens'] = sum(fe['attention_mask'])
        used_word_ids = [x for x in fe['word_ids'] if x != util.null_wordid()]
        fe['num_covered_words'] = len(set(used_word_ids))
        fe['num_covered_tokens'] = len(used_word_ids)
        fe['words_covered_ratio'] = fe['num_covered_words'] / fe['num_words']
        if fe['words_covered_ratio'] < 1 and FLAGS.filter_records:
          continue
        # ic(fe['words_covered_ratio'], fe['num_covered_words'],  fe['num_words'], fe['num_covered_words'] / fe['num_words'])
        
        fe['word_mask'] = [0] * MAX_WORDS
        fe['word_start_mask'] = [0] * MAX_WORDS
        # fe['word_relative_positions'] = [0.] * MAX_WORDS
        first = True
        for word_id in used_word_ids:
          if word_id >= MAX_WORDS:
            continue
          fe['word_mask'][word_id] = 1
          if not first:
            fe['word_start_mask'][word_id] = 1
          first = False
        # for word_id, res_pos in zip(fe['word_ids'], fe['relative_positions']):
        #   fe['word_relative_positions'][word_id] = res_pos
        for start, end, para_type in zip(starts, ends, para_types):
          preds = set(range(start, end))
          first = True
          word_id_ = None
          last = None
          last_wordid = None
          # 由于之前测试mask_inside线上效果稳定好于label_inside所以目前只考虑mask_inside模式
          # 也就是一个word id只和一个token id（第一个）对应其余的对应-1 word_id
          for j, word_id in enumerate(fe['word_ids']):
            if word_id in preds:
              if word_id != word_id_:
                word_id_ = word_id
                  
              # 多个token对应一个word id 理论上截断应该不认为是start 但是实测影响不大 word_id == int(row['start'])
              # change roberta-squad 在线693->688 electra 692->695 TODO 是否不对roberta做这个操作
              # bart 688 -> 687 deberta 698 -> 696 ensemble 705 -> 701 可能还是多了分割 集成反而有收益？
              # 另外这个只对mid 和 end model有影响
              # if first:
              if first and word_id == start:
                fe['dis_start'][j] = 1
                fe['dis_start2'][j] = 1
                fe['word_dis_start'][word_id] = 1
                first = False
              fe['label'][j] = para_type
              fe['word_label'][word_id] = para_type
              last = j
              last_wordid = word_id
          # if last is not None:
          if last is not None and last_wordid == (end - 1):
            fe['dis_end'][last] = 1
            fe['dis_end2'][last] = 1
            fe['word_end'][last_wordid] = 1
        
        if fe:
          # input_ids, attention_mask, mask, label, start, start2, end, end2
          deal_label(fe, words)
          if not FLAGS.stats:
            if not (fe['words_covered_ratio'] == 1 and FLAGS.ignore_short):
              count = get_count(fe['classes'])
              for _ in range(count):
                assert max(fe['classes']) <= 1
                writer.write(fe)
          # fes.append({k: v for k, v in fe.items() if not isinstance(v, list)})
          fes.append(fe)
      
  fes_dict[index] = fes
  if index == 0:
    print(fes[0])
    
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
  # FLAGS.records_type = 'token'
  # FLAGS.rv = 2
  config.init_()
  
  global df, records_dir, ids, tokenizer
  np.random.seed(FLAGS.seed_)
  assert FLAGS.mark == 'train'
  ifile = f'{FLAGS.idir}/{FLAGS.mark}_flat_{FLAGS.aug}.fea'
  if FLAGS.corrected:
    ifile = ifile.replace('.fea', '.corrected.fea')
  ic(ifile)
  records_dir = f'{FLAGS.idir}/{FLAGS.records_name}/{FLAGS.mark}'
  if FLAGS.clear_first:
    command = f'rm -rf {records_dir}'
    gezi.system(command)    
  else:
    assert not glob.glob(f'{records_dir}/*.tfrec'), records_dir
  with gezi.Timer('read_csv'):
    df = pd.read_feather(ifile)
  
  # df = df.sample(1., random_state=1024)
  ic(df.sort_values('id'))
  # df = df[df.id=='0016926B079C']
  
  ic(FLAGS.backbone)
  tokenizer = get_tokenizer(FLAGS.backbone)
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
  d.reset_index().to_feather(f'{FLAGS.idir}/{FLAGS.records_name}/records.fea')


if __name__ == '__main__':
  app.run(main)
