#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import time
import gezi
from gezi import tqdm
import sys
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
from src import util
from src.config import *
gezi.init_flags()


# In[14]:


np.random.seed(2022)


# In[15]:


df = pd.read_feather('../input/feedback-prize-2021/train.fea')


# In[39]:


id_ = 'C196DA0C7688'


# In[59]:


MAX_LEN = 2000
id = None
fes = []
fe = {}
pre_idx = 0
for i, row in tqdm(enumerate(df.itertuples()), total=len(df)):
  row = row._asdict()
  if row['id'] != id:
    if fe:
      for j in range(pre_idx, len(fe['text'].split())):
        fe['para_types'][j] = 0
      fe['para_types'] = fe['para_types'][:j + 1]
      fe['start'] = fe['start'][:j + 1]
      assert len(fe['para_types']) == len(fe['text'].split())
      assert min(fe['para_types']) >= 0
      fes.append(fe)
    fe = {}
    fe['id'] = row['id']
    fe['para_types'] = [-1] * MAX_LEN
    fe['start'] = [0] * MAX_LEN
    fe['text'] = row['text_']
    pre_idx = 0
    id = row['id']
  predictionstring = row['predictionstring']
  positions = predictionstring.split()
  start, end = int(positions[0]), int(positions[-1])
  fe['start'][start] = 1
  if start > pre_idx:
    for j in range(pre_idx, start):
      fe['para_types'][j] = 0
  for j in range(start, end + 1):
    fe['para_types'][j] =  row['discourse_type_id']
  pre_idx = end + 1
if fe:
  for j in range(pre_idx, len(fe['text'].split())):
    fe['para_types'][j] = 0
  fe['para_types'] = fe['para_types'][:j + 1]
  fe['start'] = fe['start'][:j + 1]
  assert len(fe['para_types']) == len(fe['text'].split())
  assert min(fe['para_types']) >= 0, fe['id']
  fes.append(fe)


# In[60]:


d = pd.DataFrame(fes)


# In[61]:


d[d.id==id_]['text'].values[0]


# In[62]:


d


# In[63]:


d['part'] = [np.random.randint(19) for _ in range(len(d))]


# In[64]:


gezi.init_flags()


# In[66]:


def find_span(words, start, end):
  start2, end2 = None, None
  idx = 0
  for i, word in enumerate(words):
    if word == FLAGS.br:
      continue
    else:
      if idx == start:
        start2 = i
      if idx == end:
        end2 = i
      idx += 1
  return start2, end2

fes = []
for row in tqdm(d.itertuples(), total=len(d)):
  row = row._asdict()
  para_id = 0
  pre_type = None
  words = util.get_words(row['text'])
  count += 1
  pre_idx =0
  for i, para_type in enumerate(row['para_types']):
    if (para_type != pre_type) or (row['start'][i] == 1):
      if pre_type != None:
        fe = {}
        fe['id'] = row['id']
        fe['part'] = row['part']
        fe['para_id'] = para_id
        fe['para_type'] = pre_type
        fe['start'] = pre_idx
        fe['end'] = i
        start2, end2 = find_span(words, pre_idx, i)
        fe['start2'] = start2
        fe['end2'] = end2
        fe['para'] = ' '.join(words[start2: end2])
        fes.append(fe)
        para_id += 1
        pre_idx = i
      pre_type = para_type
  i += 1
  fe = {}
  fe['id'] = row['id']
  fe['part'] = row['part']
  fe['para_id'] = para_id
  fe['para_type'] = pre_type
  fe['start'] = pre_idx
  fe['end'] = i
  start2, end2 = find_span(words, pre_idx, i)
  fe['start2'] = start2
  fe['end2'] = end2
  fe['para'] = ' '.join(words[start2: end2])
  fes.append(fe)
      


# In[67]:


d = pd.DataFrame(fes)


# In[68]:


d


# In[69]:


d[d.id==id_].para.values


# In[70]:


ids = []
texts = []
id = None
words = []
for row in tqdm(d.itertuples(), total=len(d)):
  row = row._asdict()
  if row['id'] != id:
    if words:
      ids.append(id)
      texts.append(' '.join(words))
    words = []
    id = row['id']
  words.extend(row['para'].split(' '))
if words:
  ids.append(id)
  texts.append(' '.join(words))
d_text = pd.DataFrame({
  'id': ids,
  'text': texts
})
d_text['text'] = d_text.text.apply(lambda x: x.replace('\n', FLAGS.br))


# In[71]:


d_text[d_text.id==id_].values[0]


# In[72]:


len(d_text[d_text.id==id_].text.values[0].split())


# In[73]:


len(d_text[d_text.id==id_].text.values[0].replace(FLAGS.br, '\n').split())


# In[74]:


d = pd.merge(d, d_text, on=['id'], how='left')
d['pid'] = range(len(d))
d['para'] = d.para.apply(lambda x: x.replace('\n', FLAGS.br))


# In[75]:


d = pd.merge(d, df[['id', 'kfold']].drop_duplicates(), on='id', how='inner')


# In[76]:


d['worker'] = d['part'] * 5 + d['kfold']


# In[77]:


d


# In[78]:


d.to_csv('../input/feedback-prize-2021/train_en.csv')


# In[79]:


d.reset_index().to_feather('../input/feedback-prize-2021/train_en.fea')


# In[80]:


d.describe()


# In[81]:


d.kfold.value_counts()


# In[82]:


d.part.value_counts()


# In[83]:


d.worker.value_counts()


# In[84]:


len(d.worker == 0)


# In[85]:


id = 'C196DA0C7688'
d[d.id==id]


# In[86]:


d[d.id==id].text.values[0]


# In[ ]:




