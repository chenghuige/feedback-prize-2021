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


import tensorflow as tf
from absl import flags
FLAGS = flags.FLAGS
from transformers import AutoTokenizer, AutoModel
from src import config
from src.model import Model
from src.dataset import Dataset
from src.util import get_preds
import src.eval as ev
import melt as mt
import husky


# In[3]:


import torch
from torch import nn
from torch.nn import functional as F
import lele


# In[4]:


class Model(nn.Module):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)  
    self.backbone = AutoModel.from_pretrained(FLAGS.backbone)
    
    dim = self.backbone.config.hidden_size
    self.dense = nn.Linear(dim, FLAGS.num_classes)
    lele.keras_init(self, True, True)
    
    self.eval_keys = ['id', 'label', 'mask', 'word_ids']
    self.str_keys = ['id']
    self.out_keys = []

  def forward(self, inputs):
    x = self.backbone(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
    x = self.dense(x)
    return x
  
  def get_loss_fn(self):
    def calc_loss(y_, y):
      y_ = y_.view(-1, FLAGS.num_classes)
      y = y.view(-1)
      return nn.CrossEntropyLoss()(y_, y)
    return calc_loss


# In[5]:


FLAGS([''])
FLAGS.mts = True
FLAGS.torch = True
FLAGS.optimizer = 'bert-Adam'
config.init_()
config.init()
mt.init()


# In[6]:


fit = mt.fit
strategy = mt.distributed.get_strategy()
with strategy.scope():    
  model = Model()

  fit(model,  
    Dataset=Dataset,
    loss_fn=model.get_loss_fn(),
    eval_fn=ev.evaluate
    ) 


# In[ ]:




