#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import random
import itertools
import matplotlib.pyplot as plt
import plotly.express as px
import glob
import spacy
import sklearn
from collections import defaultdict, Counter
from bisect import bisect_left
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
import pymp
from multiprocessing import Pool, Manager, cpu_count
from IPython.display import display_html
from itertools import chain, cycle
import lightgbm as lgb
import sys
sys.path.append('../../../../utils')
sys.path.append('..')
import gezi
from gezi import tqdm
from src.eval import *
from src.util import *
from src import config
from src.visualize import *
from src.rewards import *
pd.set_option('display.float_format', lambda x: '%.02f' % x)


# In[3]:


# https://www.kaggle.com/chasembowers/sequence-postprocessing-v2-67-lb/notebook#Sequence-Datasets


# In[4]:


root = '../input/feedback-prize-2021'


# In[5]:


df = pd.read_feather(f'{root}/train_en.fea')


# In[6]:


MAX_SEQ_LEN = {}
df['len'] = df.end - df.start
max_lens = df.groupby('para_type')['len'].quantile(.995)
for i in range(len(ALL_CLASSES)):
  MAX_SEQ_LEN[id2dis[i]] = int(max_lens[i])
MAX_SEQ_LEN


# In[7]:


# In[8]:


d = pd.read_feather(f'{root}/para_label.fea')


# In[9]:


gts = {}
for row in tqdm(d.itertuples(), total=len(d)):
  gts[row.id] = decode_label_all(row.start, row.para_types)


# In[10]:


folds = {}
for row in tqdm(df.itertuples(), total=len(df)):
  folds[row.id] = row.kfold


# In[11]:


np.random.seed(12345)


# In[12]:


subfolds = {}
for row in tqdm(df.itertuples(), total=len(df)):
  subfolds[row.id] = np.random.randint(5)


# In[13]:


gts['A8445CABFECE']


# In[43]:


def make_dataset(info, start_thre=0.3, start_thre2=0.95, fold=None, subfold=None, infer=False):
  total = len(info['id'])
  fes = []
  scores = []
  matches = []
  for i in tqdm(range(total)):
    x = {}
    for key in info:
      x[key] = info[key][i]
    id = x['id']
    gt = gts[id]
    fe = {}
    fe['id'] = id
    fe['index'] = i
    fe['fold'] = folds[id]
    if fold is not None and fe['fold'] != fold:
      continue
    fe['subfold'] = subfolds[id]
    if subfold is not None and fe['subfold'] != subfold:
      continue
    fe['num_words'] = x['num_words']
    fe['seps'] = (x['start_probs'][:,1] > 0.5).sum()
    fe['sep_ratio'] = fe['seps'] / fe['num_words']
    num_words = x['num_words']
    for j in range(num_words):
      if j > 0 and x['start_probs'][j][1] < start_thre and x['preds'][j] == x['preds'][j - 1]:
        continue
      probs = np.zeros_like(x['probs'][j])
      probs += x['probs'][j]
      fe['start'] = j
      fe['start_ratio'] = (j + 1) / num_words
      fe['start_probs'] = x['probs'][j]
      fe['start_sep_prob'] = x['start_probs'][j][1] if j > 0 else 1.
      preds = {k: 0 for k in range(NUM_CLASSES)}
      fe['start_cls'] = x['preds'][j]
      fe['max_start_prob'] = probs.max()
      preds[x['preds'][j]] += 1
      fe['pre_cls'] = -1 if j == 0 else x['preds'][j - 1]
      fe['pre_max_prob'] = 1 if j == 0 else x['probs'][j - 1].max()
      sep_count = 0
      for k in range(j + 1, num_words):
        probs += x['probs'][k]
        fe['para_len'] = k + 1 - j
        fe['para_len_ratio'] = (k + 1 - j) / num_words
        preds[x['preds'][k]] += 1
        if k + 1 < num_words and x['start_probs'][k + 1][1] < start_thre and x['preds'][k] == x['preds'][k + 1]:
          continue
        if k + 1 < num_words and x['start_probs'][k + 1][1] > 0.5:
          sep_count += 1
        fe['end'] = k + 1
        fe['end_ratio'] = (k + 1) / num_words
        fe['end_probs'] = x['probs'][k]
        fe['max_end_prob'] = x['probs'][k + 1].max() if k + 1 < num_words else 1
        end_cls = np.argmax(x['probs'][k])
        fe['end_cls'] = end_cls
        fe['end_sep_prob'] = x['start_probs'][k + 1][1] if k + 1 < num_words else 1.
        fe['sep_add_prob'] = (fe['start_sep_prob'] + fe['end_sep_prob']) / 2.
        fe['sep_mul_prob'] = (fe['start_sep_prob'] * fe['end_sep_prob']) ** 0.5
        fe['num_classes'] = len([k for k in range(NUM_CLASSES) if preds[k] > 0])
        mean_probs = gezi.softmax(probs)
        fe['mean_probs'] = mean_probs
        fe['max_prob'] = mean_probs.max()
        top_classes = np.argsort(-mean_probs,axis=0)[:2]
        fe['top_class'] = top_classes[0]
        fe['top_class2'] = top_classes[1]
        fe['next_cls'] = -1 if k + 1  == num_words else x['preds'][k + 1]
        top_classes = top_classes[:1]
        if fe['para_len'] > 1 and fe['num_classes'] < 4:
          for cls in top_classes:
            if cls and fe['para_len'] >  MAX_SEQ_LEN[id2dis[cls]]:
              continue
            if fe['mean_probs'][cls] < proba_thresh[id2dis[cls]]:
              continue
            fe['mean_prob'] = fe['mean_probs'][cls]
            fe['start_prob'] = x['probs'][j][cls]
            fe['end_prob'] = x['probs'][k][cls]
            fe['class_ratio'] = (x['preds'][j:k+1] == cls).sum() / fe['para_len']
            # fe['class_max_prob'] = x['probs'][j:k+1][cls].max()
            # fe['class_min_prob'] = x['probs'][j:k+1][cls].min()
            fe['cls'] = cls
            fe['is_top_class'] = int(cls == fe['top_class'])
            fe['is_top_class2'] = int(cls == fe['top_class2'])
            if not infer:
              fe['score'] = best_match(gt[cls], [j, k + 1])
              fe['match'] = calc_match(gt[cls], [[j, k + 1]])
              # if not fe['match'] and np.random.randint(3) != 0:
              #   continue
              scores.append(fe['score'])
              matches.append(fe['match'])
            fes.append(fe.copy())
        if k + 1 < num_words and x['start_probs'][k + 1][1] > start_thre2:
          break
        if sep_count > 2:  
          break
  if not infer:
    ic(np.mean(scores), np.mean(matches))
  d = pd.DataFrame(fes)
  return d


# In[44]:


info = gezi.load('../working/offline/44/valid.pkl')


# In[45]:


nw = 5
ds = Manager().dict()
with pymp.Parallel(nw) as p:
  for fold in p.range(5):
    fold_ = fold
    ds[fold_] = make_dataset(info, fold=fold)


# In[46]:


dtrain = pd.concat([ds[0], ds[2], ds[3], ds[4]])


# In[71]:


dtrain['match'] = dtrain['match'].apply(lambda x: min(x, 1))


# In[47]:


dvalid = ds[1]


# In[72]:


dvalid['match'] = dvalid['match'].apply(lambda x: min(x, 1))


# In[195]:


# dtrain = d[d.subfold != 0]
# # dtrain = d0[d0.subfold == 1]
# dvalid = d[d.subfold == 0]
reg_cols =  [
              'num_words', 'start', 'start_ratio', 'end', 'end_ratio',
              'start_sep_prob', 'end_sep_prob', 'para_len', 'para_len_ratio',
              'num_classes', 'mean_prob', 
              'start_prob', 'end_prob', 
              # 'seps', 'sep_ratio', 
              'class_ratio',
              # 'sep_add_prob', 'sep_mul_prob', 
              # 'is_top_class', 'is_top_class2',
            #   'class_max_prob', 'class_min_prob'
        ]
cat_cols = [
            'cls', 
            'start_cls', 'end_cls',
            'pre_cls', 
            'next_cls'
            ]
label_col = 'match'
label_col = 'score'
cols = reg_cols + cat_cols
X_train = dtrain[cols]
y_train = dtrain[[label_col]]


# In[196]:


X_valid = dvalid[cols]
# y_valid = dvalid[[label_col]]
y_valid = dvalid[['match']]


# In[199]:


learning_rate = 0.1
num_leaves = 12
feature_fraction = 0.8
num_boost_round = 1000
params = {
          # "objective": "binary",
          # "objective": "regression" if label_col is 'score' else 'binary',
          "objective": "cross_entropy",
          "metric": "auc",
          "boosting_type": "gbdt",
          "learning_rate": learning_rate,
          "num_leaves": 12,
          "max_bin": 256,
          "feature_fraction": 0.9,
          "verbosity": 0,
          # "drop_rate": 0.1,
          "is_unbalance": True,
          # "max_drop": 50,
          "min_child_samples": 20,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "bagging_freq": 5,
          "bagging_fraction": 0.9,
          # "num_trees": 200,
          "subsample": 0.9
          }


# In[200]:


d_train = lgb.Dataset(X_train, y_train)
d_valid = lgb.Dataset(X_valid, y_valid, reference=d_train)
bst = lgb.train(params, d_train, num_boost_round, valid_sets=d_valid, 
                categorical_feature=cat_cols,
                verbose_eval=1,
                early_stopping_rounds=10)


# In[ ]:


l = list(zip(bst.feature_name(), bst.feature_importance()))
l.sort(key=lambda x: -x[1])
l


# In[ ]:


def greedy_decodes(df, cols, para_classifier):
  df['pred'] = para_classifier.predict(df[cols])
  ids = set(df.id)
  ids_list, types_list, preds_list = [], [], []
  for id in tqdm(ids):
    d = df[df.id == id]
    d = d.sort_values(['pred'], ascending=[False])
    num_words = d.num_words.values[0]
    used = np.zeros(num_words)
    for row in d.itertuples():
      start, end = row.start, row.end
      cls = row.cls
      if used[start: end].sum() == 0:
        if used.sum() == num_words:
          break
        # used[start: end] = 1
        if cls > 0:
          if row.mean_prob > proba_thresh[id2dis[cls]]:
            used[start: end] = 1
            ids_list.append(id)
            types_list.append(id2dis[cls])
            preds_list.append(' '.join([str(x) for x in range(start, end)]))
          
  df = pd.DataFrame({
    'id': ids_list,
    'class': types_list,
    'predictionstring': preds_list,
  })
  return df


# In[ ]:


df_pred = greedy_decodes(dvalid, cols, bst)


# In[ ]:


df_gt = pd.read_csv('../working/offline/44/valid_gt.csv')


# In[ ]:


res = calc_metrics(df_gt[df_gt.id.isin(set(df_pred.id))], df_pred)
ic(res['f1/Overall'])
ic(res)


# In[ ]:


# df_pred2[df_pred2.id=='4125C404E34B']


# In[ ]:


# dvalid[dvalid.id=='4125C404E34B'][['id', 'pred', 'match', 'score', 'start', 'end', 'cls', 'mean_prob', 'start_sep_prob', 'end_sep_prob']]


# In[ ]:


# gts['4125C404E34B']


# In[ ]:


# ALL_CLASSES


# In[ ]:


# df_pred2 = df_pred2[df_pred2.id.isin(set(df_pred.id))]


# In[175]:


# df_pred_fold1 = greedy_decodes(ds[1], cols, bst)


# In[176]:


# res = calc_metrics(df_gt[df_gt.id.isin(set(df_pred_fold1.id))], df_pred_fold1)
# ic(res)


# In[ ]:





# In[ ]:




