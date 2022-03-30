#!/usr/bin/env python
# coding: utf-8

# In[124]:


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


# In[125]:


root = '../input/feedback-prize-2021'


# In[126]:


df = pd.read_feather(f'{root}/train_en.fea')


# In[127]:


df


# In[128]:


lang = 'fr'
folds = 10
dlang = pd.concat(
  [pd.read_csv(f'{root}/trans_{lang}_{fold}.csv') for fold in range(folds) if os.path.exists(f'{root}/trans_{lang}_{fold}.csv')]
)
dlang = dlang.sort_values(['pid'])
dlang


# In[129]:


dlang.to_csv(f'{root}/trans_{lang}.csv', index=False)


# In[130]:


d = pd.merge(df, dlang[['pid', 'mid', 'target']], on='pid', how='left')
d = d.fillna('[NONE]')
d


# In[131]:


ic(len(d[d.mid != '[NONE]']) / len(d))


# In[132]:


d[d.para =='[NONE]']


# In[133]:


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
  col = 'para' if row['target'] == '[NONE]' else 'target'
  words.extend(str(row[col]).split(' '))
if words:
  ids.append(id)
  texts.append(' '.join(words))
d_text = pd.DataFrame({
  'id': ids,
  'text': texts
})
d_text


# In[134]:


d = pd.merge(d, d_text, on=['id'], how='inner', suffixes=('_ori', ''))


# In[135]:


starts, ends = [], []
end = 0
for row in tqdm(d.itertuples(), total=len(d)):
  row = row._asdict()
  if row['para_id'] == 0:
    end = 0
  start = end
  col = 'para' if row['target'] == '[NONE]' else 'target'
  end = start + len(str(row[col]).replace('[SEP]', '\n').split())
  starts.append(start)
  ends.append(end)
d['start'] = starts
d['end'] = ends


# In[136]:


d


# In[137]:


d[['id', 'para_type', 'start', 'end', 'text']].reset_index().to_feather(f'{root}/train_{lang}.fea')


# In[ ]:




