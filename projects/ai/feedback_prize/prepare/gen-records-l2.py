#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records-stacking.py
#        \author   chenghuige  
#          \date   2022-02-23 09:14:43.486288
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os
import json
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
from itertools import chain, cycle

import gezi
from gezi import tqdm
import melt as mt
from src import config
from src.config import *
from src.ensemble_conf import mns, mns1, mns2, mns3, v1, v2, v3, v
from src.decode import *

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('mark', 'train', 'train')
flags.DEFINE_integer('buf_size', 100000, '')
flags.DEFINE_integer('num_records', None, '')
flags.DEFINE_integer('seed_', 1024, '')
flags.DEFINE_bool('stats', False, '')
# flags.DEFINE_bool('stratified_split', True, '')
flags.DEFINE_integer('label_shift', 0, '')
flags.DEFINE_bool('ignore_short', False, '')
flags.DEFINE_float('inst_weight', 1., '')

df = None
records_dir = None
num_models = None
infos = []
oofs = []
oof = {}
records = {}
votes = []
df_gt = None
tokenizer = None

def load_oof():
  ic(gezi.get_mem_gb())
  global num_models
  valid_pkl = f'../working/offline/{v}/valid.pkl'
  ic(valid_pkl)
  xs = gezi.load(valid_pkl)
  xs = gezi.batch2list(xs)
  xs = gezi.sort_list_byid(xs)
  ic(xs[0].keys())
  for x in xs:
    oof[x['id']] = x
  
  # mark = 'offline'
  # model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
  # model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
  # model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
  
  # folds = 5  
  # for model_dir in tqdm(model_dirs):
  #   info_ = [gezi.batch2list(gezi.load(f'{model_dir}/valid.pkl')) for fold in range(folds)]
  #   info_ = list(chain.from_iterable(info_))
  #   info_ = gezi.sort_list_byid(info_)
  #   infos.append(info_)
  
  # num_models = len(model_dirs)
  # for info_ in infos:
  #   oof_ = {}
  #   for x in info_:
  #     oof_[x['id']] = x
  #   oofs.append(oof_)
  # ic(gezi.get_mem_gb())

def load_votes():
  global votes
  votes = gezi.read_pickle(f'../working/offline/{v}/votes.pkl')

def load_gt():
  global df_gt
  df_gt = pd.read_feather('../input/feedback-prize-2021/train_en.fea')
  
def load_records():
  record_files = gezi.list_files(f'{FLAGS.idir}/tfrecords/train/*.tfrec*')
  ic(record_files[:2])
  dataset = mt.Dataset('valid', files=record_files)
  datas = dataset.make_batch(512, return_numpy=True)  
  ic(dataset.num_instances)
  num_steps = dataset.num_steps
  ic(num_steps)
  
  for x in tqdm(datas, total=num_steps, desc='load_records'):
    count = len(x['id'])
    ids = gezi.decode(x['id'])
    for i in range(count):
      m = {}
      for key in x:
        m[key] = list(x[key][i])
      records[ids[i][0]] = m 

def deal(index):
  df_ = df[df['worker'] == index]
  num_insts = len(df_)
  ofile = f'{records_dir}/{index}.tfrec'

  def setup(res_list, j, num_words, gt):
    fe = {}
    pred = list_to_dict(res_list)
    fe['score'] = essay_f1(gt, pred, return_dict=False)
    fe['cls'] = []
    fe['len'] = []
    fe['sep'] = []
    fe['ratio'] = []
    fe['sep_prob'] = []
    fe['token_logits'] = []
    fe['token_probs'] = []
    fe['models'] = []
    fe['cls2'] = []
    for res in res_list:
      cls_ = res['cls']
      cls_type = id2dis[cls_]
      fe['cls2'].append(tokenizer.convert_tokens_to_ids(f'[{cls_type}]'))
      fe['cls'].append(res['cls'] + 1)
      fe['len'].append(res['len'] + 1)
      fe['sep'].append(res['start'] + 1)
      fe['ratio'].append(res['len'] / num_words)
      fe['models'].append(j + 1)
      fe['sep_prob'].append(res['sep_prob'])
      fe['token_logits'].extend(res['token_logits'])
      fe['token_probs'].extend(gezi.softmax(res['token_prob']))
    for key in ['cls', 'len', 'sep', 'ratio', 'sep_prob', 'models', 'cls2']:
      fe[key] = gezi.pad(fe[key], FLAGS.max_parts)
    for key in ['token_logits', 'token_probs']:
      fe[key] = gezi.pad(fe[key], FLAGS.max_parts * FLAGS.num_classes)
    fe['num_paras'] = len(fe['cls'])
    fe['model'] = j + 1
    return fe
    
  with mt.tfrecords.Writer(ofile, buffer_size=FLAGS.buf_size, shuffle=True, seed=1024) as writer:
    for i, row in tqdm(enumerate(df_.itertuples()), total=num_insts, desc=ofile, leave=False):
      row = row._asdict()
      id = row['id']
      fold = row['kfold']
      fe = {}
      record = records[id]
      fe['id'] = id
      fe['fold'] = fold
      fe['label'] = record['word_label']
      fe['start'] = record['word_start']
      fe['mask'] = record['word_mask']
      fe['start_mask'] =  record['word_start_mask']
      fe['num_words'] = record['num_words'][0]
      fe['word_ids'] = np.asarray([-1] * FLAGS.max_words)
      fe['word_ids'][:fe['num_words']] = range(fe['num_words'])
      fe['word_ids'] = list(fe['word_ids'])
      fe['num_models'] = len(mns)
      fe['input_ids'] = record['input_ids']
      fe['attention_mask'] = record['attention_mask']
      # fe['token_logits'] = list(np.concatenate(token_oof[fold][id], axis=-1).reshape(-1))
      # fe['start_logits'] = list(np.concatenate(start_oof[fold][id], axis=-1).reshape(-1))
      # fe['token_logits'] = list(oof[id]['pred'].reshape(-1))
      # fe['token_logits'] = gezi.pad(fe['token_logits'], FLAGS.max_words * 8)
      # fe['start_logits'] = list(oof[id]['start_logits'].reshape(-1))
      # fe['start_logits'] = gezi.pad(fe['start_logits'], FLAGS.max_words * 2)
      # print(len(fe['token_logits']), fe['num_words'][0], fe['num_words'][0] * 8, FLAGS.max_words, FLAGS.max_words * 8)
      fe['token_logits'] = list(oof[id]['pred'].reshape(-1))
      fe['token_logits'] = gezi.pad(fe['token_logits'], FLAGS.max_words * 8)
      fe['start_logits'] = list(oof[id]['start_logits'].reshape(-1))
      fe['start_logits'] = gezi.pad(fe['start_logits'], FLAGS.max_words * 2)
      fe['token_probs'] = list(oof[id]['probs'].reshape(-1))
      fe['token_probs'] = gezi.pad(fe['token_probs'], FLAGS.max_words * 8)
      fe['start_probs'] = list(oof[id]['start_probs'].reshape(-1))
      fe['start_probs'] = gezi.pad(fe['start_probs'], FLAGS.max_words * 2)
      writer.write(fe)
      # gt = get_gt_dict(df_gt, id)
      # num_gts = 0
      # for key in gt:
      #   num_gts += len(gt[key])
      # fe['num_gts'] = num_gts
      # # vote = votes[0]
      # # fe0 = setup(vote[id], 0, fe['num_words'], gt)
      # # fe0 = gezi.dict_prefix(fe0, '0/')
      # # fe.update(fe0)
      # # for j, vote in enumerate(votes[1:]):
      # #   fe1 = setup(vote[id], j + 1, fe['num_words'], gt)
      # #   fe1 = gezi.dict_prefix(fe1, '1/')
      # #   fe.update(fe1)
      # #   fe['score'] = int(fe['1/score'] > fe['0/score'])
   
      # for j, vote in enumerate(votes):
      #   fe.update(setup(vote[id], j, fe['num_words'], gt))
      #   writer.write(fe)

def main(_):    
  global df, records_dir, tokenizer
  FLAGS.encode_cls = True
  config.init_()
  FLAGS.records_name = 'tfrecords.l2'
  ifile = f'{FLAGS.idir}/train_flat_{FLAGS.aug}.corrected.fea'
  records_dir = f'{FLAGS.idir}/{FLAGS.records_name}/train'
  if FLAGS.clear_first:
    command = f'rm -rf {records_dir}'
    gezi.system(command)    
  else:
    assert not glob.glob(f'{records_dir}/*.tfrec'), records_dir
  with gezi.Timer('read_csv'):
    df = pd.read_feather(ifile)

  ic(df.sort_values('id'))
  # df = df[df.id=='0016926B079C']
  tokenizer = get_tokenizer(FLAGS.backbone)
  
  load_oof()
  load_records()
  # load_votes()
  # load_gt()
  
  num_records = FLAGS.num_records or cpu_count()
  if num_records > FLAGS.folds:
    num_records = int(num_records / FLAGS.folds) * FLAGS.folds
  FLAGS.num_records = num_records
  if FLAGS.stats:
    num_records = 1
  ic(FLAGS.folds, num_records)
  
  if num_records > 1:
    with Pool(num_records) as p:
      p.map(deal, range(num_records))
  else:
    deal(0)
  
if __name__ == '__main__':
  app.run(main)
