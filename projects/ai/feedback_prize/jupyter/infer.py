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
import sys
sys.path.append('../../../../utils')
sys.path.append('..')
import gezi
from gezi import tqdm
from src.eval import calc_f1
pd.set_option('display.float_format', lambda x: '%.0f' % x)


# In[2]:


v = '7'
# mn= 'large.lr5e-5.m2'
mn = 'pt.large'
model_dir = f'../working/online/{v}.0/{mn}'


# In[3]:


import tensorflow as tf
import torch
from absl import flags
FLAGS = flags.FLAGS
from datasets import Dataset
from src import config
from src.util import *
import melt as mt
import numpy as np
from gezi import tqdm
import gezi
import husky
import lele


# In[4]:


argv = open(f'{model_dir}/command.txt').readline().strip().split()
FLAGS(argv)
FLAGS.wandb = False
config.init()
# mt.init()
MAX_LEN = FLAGS.max_len
# TODO for longformer large seems torch can use much larger bs for inference? for torch large+32 use 13.9G 
BATCH_SIZE = 8 
model = get_model()
gezi.load_weights(model, model_dir)
try:
  ic(open(f'{model_dir}/path.txt').readline().strip())
except Exception:
  pass


# In[5]:


inputs = get_test_inputs(FLAGS.backbone)
inputs_ = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}


# In[6]:


p = gezi.predict(model, inputs_, batch_size=BATCH_SIZE)


# In[7]:


list(p.keys())


# In[8]:


p.update({
  'id': inputs['id'],
  'word_ids': inputs['word_ids']
})
convert_res(p)


# In[9]:


df = get_preds(p)
df


# In[10]:


df.to_csv(f'{model_dir}/submission.csv')


# In[ ]:





# In[ ]:




