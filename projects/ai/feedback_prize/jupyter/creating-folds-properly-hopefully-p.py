#!/usr/bin/env python
# coding: utf-8

# In[7]:


# !pip install -q iterative-stratification


# In[8]:


import pandas as pd
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# In[9]:


df = pd.read_feather("../input/feedback-prize-2021/train.fea")
if 'kfold' in df.columns:
   df.drop('kfold', axis=1, inplace=True)
if 'cluster' in df.columns:
   df.drop('cluster', axis=1, inplace=True)
dfx = pd.get_dummies(df, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()


# In[10]:


cluster = pd.read_csv("../input/feedback-prize-2021/cluster.csv")
dfx = dfx.merge(cluster, on=['id'], how='left')


# In[11]:


dfx.columns


# In[12]:


for i in range(15):
  dfx[f'cluster_{i}'] = dfx.cluster.apply(lambda x: int(x == i))


# In[13]:


cols = [c for c in dfx.columns if c.startswith('cluster') or c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
cols


# In[14]:


dfx = dfx[cols]
seed = 20201021
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
labels = [c for c in dfx.columns if (c != "id" and c != "cluster")]
dfx_labels = dfx[labels]
dfx["kfold"] = -1

for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
  print(len(trn_), len(val_))
  dfx.loc[val_, "kfold"] = fold

df = df.merge(dfx[["id", "kfold", "cluster"]], on="id", how="left")
print(df.kfold.value_counts())
df.to_csv("../input/feedback-prize-2021/train_folds.csv", index=False)


# In[15]:


x = df[['id', 'kfold', 'cluster']].drop_duplicates()


# In[17]:


display(df.groupby(["kfold", 'discourse_type'], as_index=False).count().pivot(index='discourse_type', columns='kfold', values='id').T)


# In[18]:


display(x.groupby(["kfold", 'cluster'], as_index=False).count().pivot(index='cluster', columns='kfold', values='id').T)


# In[19]:


x.to_csv("../input/feedback-prize-2021/folds.csv")


# In[ ]:




