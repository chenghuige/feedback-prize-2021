#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GroupKFold,  StratifiedGroupKFold, KFold
from IPython.display import display_html, display
import gezi
from gezi import tqdm


# In[2]:


df = pd.read_csv("../input/feedback-prize-2021/train.csv")
cluster = pd.read_csv("../input/feedback-prize-2021/cluster.csv")
df = df.merge(cluster, on=['id'], how='left')


# In[3]:


# 1015 1018 1028 1035 1046
#seed = 1046
# seed = 20211256
# seed = 1238923211
seed = 12768
ic(seed)
np.random.seed(seed)
dfx = df[['id', 'cluster']].drop_duplicates()
dfx['kfold'] = [np.random.randint(5) for _ in range(len(dfx))]
if 'kfold' in df.columns:
  df.drop('kfold', axis=1, inplace=True)
train = df.merge(dfx[["id", "kfold"]], on="id", how="left")
grp = train.groupby(["kfold", 'discourse_type'], as_index=False).count()
display(grp.pivot(index='discourse_type', columns='kfold', values='id').T)
display(dfx.groupby(["kfold", 'cluster'], as_index=False).count().pivot(index='cluster', columns='kfold', values='id').T)
assert len(train.groupby(["id", "kfold"], as_index=False).count()) == train["id"].nunique()
dfx.head()


# In[4]:


# K = 5
# SEED = 1111
# dfx = df[['id', 'cluster']].drop_duplicates()
# skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)
# x = df.groupby('id')['discourse_type'].agg(list).reset_index()
# x['Rebuttal'] = df['discourse_type'].apply(lambda l: int('Rebuttal' in l))
# dfx = dfx.merge(x, on='id', how='left')
# splits = list(skf.split(X=dfx, y=dfx['Rebuttal']))


# In[5]:


# folds = np.zeros(len(dfx), dtype=int)
# for i, (train_idx, val_idx) in enumerate(splits):
#   folds[val_idx] = i
# dfx['kfold'] = folds


# In[6]:


# K = 5
# # SEED = 222
# # SEED = 12356
# # SEED = 102453
# SEED = 1203
# ic(SEED)
# dfx = df[['id', 'cluster']].drop_duplicates()
# df['Rebuttal'] = (df['discourse_type'] == 'Rebuttal').astype(int)
# df['Counterclaim'] = (df['discourse_type'] == 'Counterclaim').astype(int)
# df['Rare'] = df['discourse_type'].apply(lambda x: x in ['Rebuttal', 'Counterclaim']).astype(int)
# skf = StratifiedGroupKFold(n_splits=K, random_state=SEED, shuffle=True)
# splits = list(skf.split(X=df, y=df['Rare'], groups=df['id']))


# In[7]:


# folds = np.zeros(len(df), dtype=int)
# for i, (train_idx, val_idx) in enumerate(splits):
#   folds[val_idx] = i
# df['kfold'] = folds


# In[8]:


# ic(df.Rebuttal.sum() / 5, df.Counterclaim.sum() / 5)


# In[9]:


# train = df.merge(dfx[["id", "cluster"]], on="id", how="left")
# grp = train.groupby(["kfold", 'discourse_type'], as_index=False).count()
# display(grp.pivot(index='discourse_type', columns='kfold', values='id').T)


# In[10]:


# x = grp.pivot(index='discourse_type', columns='kfold', values='id').T


# In[11]:


m = {}
# 1203
best_seed = None
best_a, best_b = 1000, 1000
def gen_folds(seed):
  global best_seed,best_a, best_b
  skf = StratifiedGroupKFold(n_splits=K, random_state=seed, shuffle=True)
  splits = list(skf.split(X=df, y=df['Rare'], groups=df['id']))
  folds = np.zeros(len(df), dtype=int)
  for i, (train_idx, val_idx) in enumerate(splits):
    folds[val_idx] = i
  df['kfold'] = folds
  train = df.merge(dfx[["id", "cluster"]], on="id", how="left")
  grp = train.groupby(["kfold", 'discourse_type'], as_index=False).count()
  x = grp.pivot(index='discourse_type', columns='kfold', values='id').T
  a = abs(x['Rebuttal'] - x['Rebuttal'].mean()).max()
  b = abs(x['Counterclaim'] - x['Counterclaim'].mean()).max()
  m[seed] = (a, b)
  if a < best_a and b < best_b:
    best_a = a
    best_b = b
    best_seed = seed
    ic(seed, best_a, best_b)


# In[12]:


# for seed in tqdm(range(1024, 2048)):
#   gen_folds(seed)


# In[14]:


# dfx = df[['id', 'cluster', 'kfold']].drop_duplicates()


# In[15]:


display(dfx.groupby(["kfold", 'cluster'], as_index=False).count().pivot(index='cluster', columns='kfold', values='id').T)


# In[16]:


dfx[['id', 'cluster', 'kfold']].to_csv('../input/feedback-prize-2021/folds.csv')


# In[17]:


dfx


# In[18]:


dfx.describe()


# In[ ]:




