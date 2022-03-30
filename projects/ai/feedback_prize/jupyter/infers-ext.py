#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.express as px
import glob
import spacy
import sys
import gc
from IPython.display import display
sys.path.append('../../../../utils')
sys.path.append('..')
import gezi
from gezi import tqdm
from src.util import *
pd.set_option('display.float_format', lambda x: '%.0f' % x)


# In[10]:
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'WARNING'
import sys
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import tensorflow as tf
import torch
from absl import flags
FLAGS = flags.FLAGS
from transformers import AutoTokenizer
from datasets import Dataset
from src import config
from src.util import *
from src.eval import *
import melt as mt
import numpy as np
from gezi import tqdm
import gezi
gezi.set_pandas_widder()
import husky
import lele


# In[11]:


from src.ensemble_conf import mns, v
root = f'../working/offline/{v}/0/'

# In[12]:


model_dirs = [f'{root}/{mn}' for mn in mns]
model_dir = model_dirs[0]
gezi.init_flags()
batch_sizes = [32] * 100
weights = [1] * 100
num_tf_models = len([x for x in model_dirs if 'tf.' in x])
num_tf_models 

df = pd.read_csv('../input/feedback-prize-2021/ext.csv')

ensembler = Ensembler()
for i, model_dir in tqdm(enumerate(model_dirs), total=len(model_dirs)):
  ic(model_dir)
  gezi.restore_configs(model_dir)
  FLAGS.wandb = False
  ic(FLAGS.torch, FLAGS.model_dir)
  ic(FLAGS.backbone, FLAGS.max_len, FLAGS.multi_inputs, FLAGS.multi_inputs_srcs, FLAGS.merge_tokens)
  model = get_model()
  gezi.load_weights(model, model_dir)
  ic(gezi.get_mem_gb())

  display(pd.read_csv(f'{model_dir}/metrics.csv'))
  inputs = get_inputs(FLAGS.backbone, mode='train', sort=True, df=df)
 
  batch_size = 32
  ic(inputs.keys())
  ic(inputs['num_words'])
  p = gezi.predict(model, inputs, batch_sizes[i], dynamic_keys=['input_ids', 'word_ids'], mask_key='attention_mask')
  # p = predict(model, inputs, batch_sizes[i])
  p.update({
    'id': inputs['id'],
    'word_ids': inputs['word_ids'],
    'num_words': inputs['num_words']
  })
  convert_res(p)
  gezi.save(p, f'{model_dir}/ext.pkl')
  ensembler.add(p, weights[i])
  if len(inputs['id']) < 1000:
    df = get_preds(p)
    ic(df)
   
  del model
  del inputs
  if FLAGS.torch:
    torch.cuda.empty_cache()
  else:
    # only the last tf model should cuda close
    if i + 1 == num_tf_models:
      cuda.select_device(0)
      cuda.close()
  gc.collect()

if len(model_dirs) > 1:
  p = ensembler.finalize() 
  gezi.save(p, f'../working/offline/{v}/0/ext.pkl')


# In[ ]:


df = get_preds(p)
display(df)
df







