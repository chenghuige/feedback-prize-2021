#!/usr/bin/env python
# coding: utf-8

# In[1]:


from googletrans import Translator
import numpy as np
import random
import pandas as pd
import time
import gezi
from gezi import tqdm
import sys
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
from src import util


# In[2]:


root = '../input/feedback-prize-2021'


# In[3]:


df = pd.read_feather(f'{root}/train_en.fea')


# In[4]:


df


# In[5]:


def swap_random(seq):
  idx = range(len(seq))
  i1, i2 = random.sample(idx, 2)
  seq[i1], seq[i2] = seq[i2], seq[i1]


# In[6]:


para_id_list = []
para_ids = []
id = None
for row in tqdm(df.itertuples(), total=len(df)):
  row = row._asdict()
  if row['id'] != id:
    if para_ids:
      if len(para_ids) > 1:
        # ic(para_ids)
        swap_random(para_ids)
        # ic(para_ids)
      para_id_list.extend(para_ids)
      para_ids = []
    id = row['id']
  para_ids.append(row['para_id'])
if para_ids:
  if len(para_ids) > 1:
    swap_random(para_ids)
  para_id_list.extend(para_ids)
df['para_id'] = para_id_list
df.head(20)


# In[7]:


ids = []
ids2 = []
texts = []
id = None
words = []
for row in tqdm(df.itertuples(), total=len(df)):
  row = row._asdict()
  if row['id'] != id:
    if words:
      ids.append(id)
      ids2.append(len(ids2))
      texts.append(' '.join(words))
    words = []
    id = row['id']
  col = 'para'
  words.extend(str(row[col]).split(' '))
if words:
  ids.append(id)
  ids2.append(len(ids2))
  texts.append(' '.join(words))
d_text = pd.DataFrame({
  'id': ids,
  'id2': ids2,
  'text': texts
})
d_text


# In[8]:


df = pd.merge(df, d_text, on=['id'], how='inner', suffixes=('_ori', ''))


# In[9]:


df = df.sort_values(['id2', 'para_id'])
df.head(20)


# In[10]:


df[df.id =='423A1CA112E2']


# In[11]:


starts, ends = [], []
end = 0
for row in tqdm(df.itertuples(), total=len(df)):
  row = row._asdict()
  if row['para_id'] == 0:
    end = 0
  start = end
  col = 'para' 
  end = start + len(str(row[col]).replace('[SEP]', '\n').split())
  starts.append(start)
  ends.append(end)
df['start'] = starts
df['end'] = ends


# In[12]:


df


# In[13]:


df[['id', 'para_type', 'start', 'end', 'text', 'kfold', 'worker']].reset_index().to_feather(f'{root}/train_swap.fea')


# In[14]:


df[df.id =='423A1CA112E2']


# In[ ]:




