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
from absl import flags
FLAGS = flags.FLAGS
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
from src import util
pd.set_option('display.max_colwidth', 512)
gezi.init_flags()


# In[2]:


root = '../input/feedback-prize-2021'


# In[3]:


df = pd.read_feather(f'{root}/train_en.fea')


# In[4]:


df


# In[5]:


lang = 'zh-cn' if gezi.in_notebook() else sys.argv[1]
folds = 10
dlang = pd.concat(
  [pd.read_csv(f'{root}/trans_{lang}_{fold}.csv') for fold in range(folds) if os.path.exists(f'{root}/trans_{lang}_{fold}.csv')]
)
dlang['target'] = dlang['target'].apply(lambda x: x.replace('[Sep]', '[SEP]').replace('[SEP]', FLAGS.br))
dlang['mid'] = dlang['mid'].apply(lambda x: x.replace('[Sep]', '[SEP]').replace('[SEP]', FLAGS.br))
dlang = dlang.sort_values(['pid'])
dlang


# In[6]:


dlang.to_csv(f'{root}/trans_{lang}.csv', index=False)


# In[7]:


d = pd.merge(df, dlang[['pid', 'mid', 'target']], on='pid', how='left')
d = d.fillna('[NONE]')
d


# In[8]:


d.groupby('id').agg({
  'mid': list,
  'para_type': list,
}).reset_index().to_feather(f'{root}/{lang}.fea')


# In[9]:


ic(len(d[d.mid != '[NONE]']) / len(d))


# In[10]:


d[d.para =='[NONE]']


# In[11]:


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
  if len(str(row['para'].replace(FLAGS.br, '')).split(' ')) <= 2:
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


# In[12]:


d = pd.merge(d, d_text, on=['id'], how='inner', suffixes=('_ori', ''))


# In[13]:


starts, ends = [], []
end = 0
for row in tqdm(d.itertuples(), total=len(d)):
  row = row._asdict()
  if row['para_id'] == 0:
    end = 0
  start = end
  col = 'para' if row['target'] == '[NONE]' else 'target'
  if len(str(row['para'].replace(FLAGS.br, '')).split(' ')) <= 2:
    col = 'para'
  end = start + len(str(row[col]).replace(FLAGS.br, '\n').split())
  starts.append(start)
  ends.append(end)
d['start'] = starts
d['end'] = ends


# In[14]:


d


# In[15]:


d[['id', 'para_type', 'start', 'end', 'text', 'kfold', 'worker']].reset_index().to_feather(f'{root}/train_{lang}.fea')


# In[16]:


d.text.values[0]


# In[17]:


gkeys = ['id', 'text', 'kfold', 'cluster', 'worker', 'part']
allkeys = gkeys + ['para_type', 'start', 'end']
dflat = d[allkeys].groupby(gkeys).agg(list).reset_index()
dflat


# In[18]:


dflat.reset_index().to_feather(f'../input/feedback-prize-2021/train_flat_{lang}.fea')

