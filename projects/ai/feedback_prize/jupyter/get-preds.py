#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from src.get_preds import *
from src import config
from src.visualize import *
from src.rewards import *
pd.set_option('display.float_format', lambda x: '%.02f' % x)


# In[2]:


import optuna
from optuna import Trial

optuna.__version__


# In[3]:


root = '../input/feedback-prize-2021'


# In[4]:


df = pd.read_feather(f'{root}/train_en.fea')


# In[5]:


MAX_SEQ_LEN = {}
df['len'] = df.end - df.start
max_lens = df.groupby('para_type')['len'].quantile(.999)
for i in range(len(ALL_CLASSES)):
  MAX_SEQ_LEN[id2dis[i]] = int(max_lens[i])
MAX_SEQ_LEN


# In[6]:


MIN_SEQ_LEN = {}
df['len'] = df.end - df.start
min_lens = df.groupby('para_type')['len'].quantile(.001)
for i in range(len(ALL_CLASSES)):
  MIN_SEQ_LEN[id2dis[i]] = int(min_lens[i])
MIN_SEQ_LEN


# In[7]:


MAX_SEQ_LEN = {'Nothing': 223,
 'Claim': 67,
 'Evidence': 417,
 'Position': 74,
 'Concluding Statement': 245,
 'Lead': 274,
 'Counterclaim': 142,
 'Rebuttal': 142}


# In[8]:


MIN_SEQ_lens = {'Nothing': 1,
 'Claim': 1,
 'Evidence': 6,
 'Position': 2,
 'Concluding Statement': 5,
 'Lead': 4,
 'Counterclaim': 3,
 'Rebuttal': 3}


# In[9]:


d = pd.read_feather(f'{root}/para_label.fea')


# In[10]:


np.random.seed(12345)


# In[11]:


root = '../working/offline/44/'


# In[12]:


# mn = 'mui.deberta-v3.rl.s1'
# mn = 'mui.electra.rl.s1'
# mn = 'mui.4'
# mn = 'large.longformer.start.len1280.seq_encoder'
# root = f'../working/offline/44/0/{mn}'


# In[13]:


info = gezi.load(f'{root}/best.pkl')


# In[14]:


df_gt = pd.read_csv(f'{root}/valid_gt.csv')


# In[15]:


def get_pred_bystart(x, post_adjust=True, pred_info=None):
  MIN_LEN = 2
  MIN_LEN2 = 2
  NUM_CLASSES = len(id2dis)
  pred = x['preds']
  total = len(pred)
  # by prob not logit
  probs = x['probs'] 
  # probs = x['pred']
  start_prob = x['start_probs'] if 'start_probs' in x else None
  # ic((start_prob[:,1] > start_prob[:,0]).astype(int).sum())
  pre_type = None
  # predictionstring list
  preds_list = []
  # store each pred word_id for one precitionstring
  preds = [] 
  pre_scores = np.zeros_like(probs[0])
  
  types = []
    
  pre_probs = None
  for i in range(total):    
   
    is_sep = False
    if i > 0:
      pre_cls = np.argmax(pre_scores)
      pre_prob = pre_probs[pre_cls]
      pre_type = id2dis[pre_cls]
      now_cls = pred[i]
      now_type = id2dis[now_cls]
      now_prob = probs[i][now_cls]
      
      if start_prob[i].sum() == 0:
        is_sep = True
      else:
        is_sep = start_prob[i][1] >= P['sep_prob']
        
        if post_adjust and FLAGS.sep_rule:
          if i > 0:
            # 注意目前最高线上版本依然是按照第一个model取pred而不是ensemble的结果取pred信息，另外有adjacent rule待验证
            if pred[i] != pred[i - 1]:      
              if FLAGS.adjacent_rebuttal:
                if start_prob[i][1] > P['sep_prob2'] and (pre_type in  ['Rebuttal', 'Counterclaim']):
                  is_sep = True
               
              if FLAGS.adjacent_minthre:
                if start_prob[i][1] > P['sep_prob3']:
                  if len(preds) >= min_thresh[pre_type]:
                    is_sep = True
                                  
              if FLAGS.adjacent_rule:
                if pre_probs[pred[i - 1]] > P['pre_prob'] and probs[i][pred[i]] > P['after_prob'] and len(preds) >= min_thresh[id2dis[pred[i - 1]]]:
                  is_sep = True
              
              if FLAGS.adjacent_prob:
                if start_prob[i][1] > P['sep_prob4']:
                  is_sep = True
                  
    if is_sep:
      if preds:  
        if pre_type != 'Nothing':
          if post_adjust:
            if len(preds) < MIN_LEN:
              # 低置信度的干脆放弃召回 更安全 pass not continue
              pass
            else:
              if pre_probs.max() > proba_thresh[pre_type]:
                preds_list.append(' '.join(preds))
                types.append(pre_type)
          else:
            preds_list.append(' '.join(preds))
            types.append(pre_type)
      
        preds = []
        pre_scores = np.zeros_like(probs[0])
              
    pre_scores += probs[i] 
    pre_probs = gezi.softmax(pre_scores)
    # pre_probs = pre_scores / len(preds)
    preds.append(str(i))
    
  if preds:
    pre_type = id2dis[np.argmax(pre_scores)]
      
    # 结尾应该更长
    if pre_type != 'Nothing':
      if post_adjust:
        if len(preds) >= MIN_LEN2:
          if pre_probs.max() > proba_thresh[pre_type]:
            preds_list.append(' '.join(preds))
            types.append(pre_type)
        # else:
        #   top_classes = np.argsort(-pre_probs,axis=0)[:2]
        #   pre_cls = top_classes[1]
        #   pre_type = id2dis[pre_cls]
        #   if pre_probs[pre_cls] > proba_thresh[pre_type]:
        #     preds_list.append(' '.join(preds))
        #     types.append(pre_type)
      else:
        preds_list.append(' '.join(preds))
        types.append(pre_type)
  return types, preds_list


# In[16]:


def get_preds_(x, post_adjust=True, selected_ids=None, fold=None, folds=5):  
  # ic(post_adjust)
  pred_fn = None
  pred_fn = get_pred_bystart

  total = len(x['id'])
  # with gezi.Timer('get_preds'):
  # ic(FLAGS.openmp)
  ids_list, types_list, preds_list = [], [], []
  for i in tqdm(range(total), desc='get_preds', leave=False):
    id = x['id'][i]
    if selected_ids is not None and id not in selected_ids:
      continue
    if fold is not None:
      if i % folds != fold:
        continue
    x_ = {}
    for key in x: 
      x_[key] = x[key][i]
    types, preds = pred_fn(x_, post_adjust=post_adjust)
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

def get_preds(x, post_adjust=True, selected_ids=None, folds=5):  
  if selected_ids is not None:
    return get_preds_(x, post_adjust, selected_ids)
  else:
    try:
      dfs = Manager().dict()
      with pymp.Parallel(folds) as p:
        for i in p.range(folds):
          dfs[i] = get_preds_(x, post_adjust, fold=i, folds=folds)
      return pd.concat(dfs.values())
    except Exception:
      return get_preds_(x, post_adjust)


# In[17]:


gezi.init_flags()


# In[19]:


def objective(trial):

    # for key in proba_thresh:
    #   if key != 'Nothing':
    #     proba_thresh[key] = trial.suggest_uniform(key, 0., 1.)
    
    P['sep_prob'] = trial.suggest_float('sep_prob', 0.5, 1.)
    P['sep_prob2'] = trial.suggest_float('sep_prob2', 0., 0.6)
    P['sep_prob3'] = trial.suggest_float('sep_prob3', 0., 0.6)
    P['sep_prob4'] = trial.suggest_float('sep_prob4', 0., 0.6)
    FLAGS.sep_rule = trial.suggest_int('sep_rule',0, 1)
    FLAGS.adjacent_prob = trial.suggest_int('adjacent_prob', 0, 1)
    FLAGS.adjacent_rebuttal = trial.suggest_int('adjacent_rebuttal',0, 1)
    FLAGS.adjacent_rule = trial.suggest_int('adjacent_rule', 0, 1)
    FLAGS.adjacent_minthre = trial.suggest_int('adjacent_minthre', 0, 1)

    df_pred = get_preds(info, post_adjust=True, folds=50)
    res = calc_metrics(df_gt, df_pred)
    score = res['f1/Overall']
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

ic(study.best_value, study.best_params)


# In[ ]:


df_pred = get_preds(info, post_adjust=True, folds=50)


# In[ ]:


res = calc_metrics(df_gt, df_pred)
res


# In[ ]:


# res = calc_metrics(df_gt, link_evidence(df_pred))
# res


# In[ ]:


len(df_gt)


# In[ ]:


len(info['id'])


# In[ ]:





# In[ ]:




