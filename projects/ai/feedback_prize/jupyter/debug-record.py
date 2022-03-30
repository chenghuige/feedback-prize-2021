#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.express as px
import glob
import spacy
from collections import defaultdict
import sys
sys.path.append('../../../../utils')
sys.path.append('..')
import gezi
from gezi import tqdm
from src.eval import *
from src.util import *
from src import config
from src.visualize import *
pd.set_option('display.float_format', lambda x: '%.02f' % x)


# In[2]:


# In[7]:


d1 = pd.read_feather('../input/feedback-prize-2021/tfrecords.deberta-v3.cased.start.len512/records.fea').sort_values('id')


# In[4]:


d2 = pd.read_feather('../input/feedback-prize-2021/bak/records.fea').sort_values('id')


# In[8]:


d1.head()


# In[6]:


d2.head()


# In[13]:


d = pd.read_feather('../input/feedback-prize-2021/train.fea')


# In[14]:


d[d.id=='00066EA9880D']


# In[3]:


get_ipython().system('ls ../input/feedback-prize-2021/tfrecords.deberta-v3.cased.start.len512')


# In[4]:


x = pd.read_feather('../input/feedback-prize-2021/tfrecords.deberta-v3.cased.start.len512.rv3/records.fea')


# In[5]:


y = pd.read_feather('../input/feedback-prize-2021/tfrecords.deberta-v3.cased.start.len512/records.fea')


# In[ ]:





# In[12]:


x['input_ids'][0]


# In[10]:


x['word_ids']


# In[7]:


y['input_ids']


# In[11]:


y['word_ids']


# In[9]:


x['input_ids'][0] != y['input_ids'][0]


# In[8]:


for i in range(len(x)):
  if(np.eq(np.asarray(x['input_ids'].values[i]), np.asarray(y['input_ids'].values[i])) ):
    print(i)


# In[ ]:


for i in range(len(x)):
  if(np.asarray(x['word_ids'].values[i]) != np.asarray(y['word_ids'].values[i])):
    print(i)


# In[ ]:


train = pd.read_feather('../input/feedback-prize-2021/train.fea')


# In[ ]:


train2 = pd.read_feather('../input/feedback-prize-2021/train_en.fea')


# In[ ]:


d = pd.read_feather('../input/feedback-prize-2021/tfrecords.roberta.cased.mi.start.len512/records.fea')


# In[ ]:


d = pd.read_feather('../input/feedback-prize-2021/tfrecords.roberta.cased.sp.ct.mi.start.len512/records.fea')


# In[ ]:


id = 'C196DA0C7688'


# In[ ]:


train2[train2.id==id].text.values[0]


# In[ ]:


text2 = train2[train2.id==id].text.values[0].replace(FLAGS.br, '\n')


# In[ ]:


text2


# In[ ]:


len(text2.split())


# In[ ]:


len(text2)


# In[ ]:


len(train2[train2.id==id].text.values[0].split())


# In[ ]:


list(d[d.id==id]['mask'])[0].sum()


# In[ ]:


len(train[train.id==id].text_.values[0].split())


# In[ ]:


d[d.id==id]['num_words']


# In[ ]:


text = train[train.id==id].text_.values[0]
text


# In[ ]:


len(text)


# In[ ]:


gezi.init_flags()
tokenizer = get_tokenizer(backbones['tiny'])


# In[ ]:


FLAGS.max_len = 512


# In[ ]:


res = encode(text, tokenizer)


# In[ ]:


list(d[d.id==id]['word_ids'])


# In[ ]:


list(d[d.id=='0000D23A521A']['mask'])[0]


# In[ ]:


list(d[d.id=='0000D23A521A']['label'].values[0]).count(7)


# In[ ]:


list(d[d.id=='0000D23A521A']['word_label'].values[0][:256]).count(7)


# In[ ]:


list(d[d.id=='0000D23A521A']['start'].values[0]).count(1)


# In[ ]:


list(d[d.id=='0000D23A521A']['word_start'].values[0]).count(1)


# In[ ]:


[i for i, x in enumerate(d[d.id=='0000D23A521A']['word_start'].values[0]) if x > 0]


# In[ ]:


[i for i, x in enumerate(d[d.id=='0000D23A521A']['word_start'].values[0]) if x > 0]


# In[ ]:


[i for i, x in enumerate(d[d.id=='0000D23A521A']['start'].values[0]) if x > 0]


# In[ ]:


d[d.id=='0000D23A521A']['word_ids'].values[0][43]


# In[ ]:


d[d.id=='0000D23A521A']['word_ids'].values[0][226]


# In[ ]:




