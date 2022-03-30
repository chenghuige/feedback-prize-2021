#!/usr/bin/env python
# coding: utf-8

# In[1]:


from googletrans import Translator
import numpy as np
import pandas as pd
import time
import gezi
from gezi import tqdm
import sys, os
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
from src import util
pd.set_option('display.max_colwidth', 512)


# In[15]:


import random
ep = 0 if gezi.in_notebook() else int(sys.argv[1])
random.seed(1024 + ep)
np.random.seed(1024 + ep)


# In[3]:


root = '../input/feedback-prize-2021'


# In[18]:


df = pd.read_feather(f'{root}/train_en.fea')


# In[19]:


df_count = df.groupby(['id']).size().reset_index(name='num_paras')


# In[20]:


df_count['remove_idx'] = df_count.num_paras.apply(lambda x: np.random.choice(x) if x > 1 else -1)


# In[21]:


df_count


# In[22]:


df = df.merge(df_count, on='id')


# In[23]:


df.head(20)


# In[25]:


df = df[df.remove_idx != df.para_id]
df.head(5)


# In[26]:


ids = []
texts = []
id = None
words = []
for row in tqdm(df.itertuples(), total=len(df)):
  row = row._asdict()
  if row['id'] != id:
    if words:
      ids.append(id)
      texts.append(' '.join(words))
    words = []
    id = row['id']
  col = 'para'
  words.extend(str(row[col]).split(' '))
if words:
  ids.append(id)
  texts.append(' '.join(words))
d_text = pd.DataFrame({
  'id': ids,
  'text': texts
})
d_text


# In[27]:


df = pd.merge(df, d_text, on=['id'], how='inner', suffixes=('_ori', ''))
df.head()


# In[37]:


starts, ends = [], []
end = 0
id_ = None
for row in tqdm(df.itertuples(), total=len(df)):
  row = row._asdict()
  id = row['id']
  if id != id_:
    end = 0
    id_ = id
  start = end
  col = 'para' 
  end = start + len(str(row[col]).replace('[SEP]', '\n').split())
  starts.append(start)
  ends.append(end)
df['start'] = starts
df['end'] = ends
df.head()


# In[38]:


df[['id', 'para_type', 'start', 'end', 'text']].reset_index().to_feather(f'{root}/train_remove_ep{ep}.fea')


# In[39]:


gkeys = ['id', 'text', 'kfold', 'cluster', 'worker', 'part']
allkeys = gkeys + ['para_type', 'start', 'end']
dflat = df[allkeys].groupby(gkeys).agg(list).reset_index()
dflat


# In[40]:


dflat.reset_index().to_feather(f'../input/feedback-prize-2021/train_flat_remove_ep{ep}.fea')


# In[42]:


dflat[dflat.id=='423A1CA112E2']


# In[ ]:




