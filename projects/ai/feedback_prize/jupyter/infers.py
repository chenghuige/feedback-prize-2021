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
pd.set_option('display.float_format', lambda x: '%.4f' % x)


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


# v = '47'
# mns = [
#   # 'deberta-v3.start.lovasz_loss_rate-1.',
#   # 'deberta-v3.end',
#   # 'deberta-v3.se',
#   # 'deberta-v3.mid',
#   # 'electra.start',
#   # 'electra.end',
#   # 'electra.mid',
#   # 'electra.se',
#   # 'longformer.start.len1536.seq_encoder',
#   #'deberta-v3.start.mui-end-mid',
#   'deberta-v3.start.mui-end-mid.rnn_stack.x',
#   # 'electra.start.mui-end-mid',
#   # 'deberta.start.mui-end-mid',
#   # 'longformer.start.len1536.seq_encoder.rnn_layers-2.rnn_stack',
#   # 'deberta-v3.start.mui-end-mid',
#   # 'electra.start.mui-end-mid',
#   # 'deberta.start.mui-end-mid',
#   #  'deberta-v3.start.stride-0',
#   #   'deberta-v3.start.stride-64',
#   #   'deberta-v3.start.stride-128', 
#   #   'deberta-v3.start.stride-256',
# ]
from src.ensemble_conf import mns, v, weights
root = f'../working/online/{v}/0/'
# root = f'../working/offline/{v}/0/'

folds = pd.read_csv('../input/feedback-prize-2021/folds.csv')
test_ids = folds[folds.kfold==0].id.values
test_ids.sort()
test_ids = test_ids[:1000]
# test_ids = ['BE01ACCDF251']
# test_ids = test_ids[6:7]
# test_ids = test_ids[33:34]  # 02B12E0E4025
ic(len(test_ids), test_ids)
test_ids
# In[12]:


model_dirs = [f'{root}/{mn}' for mn in mns]
model_dir = model_dirs[0]
gezi.init_flags()
batch_sizes = [32] * 100
# weights = [1] * 100
num_tf_models = len([x for x in model_dirs if 'tf.' in x])
num_tf_models 

ensembler = Ensembler(need_sort=True)
for i, model_dir in tqdm(enumerate(model_dirs), total=len(model_dirs)):
  ic(model_dir)
  gezi.restore_configs(model_dir)
  # FLAGS.fake_infer = True
  FLAGS.wandb = False
  ic(FLAGS.torch, FLAGS.model_dir)
  #FLAGS.backbone = '../input/' + FLAGS.backbone.split('/')[-1]
  ic(FLAGS.backbone, FLAGS.max_len, FLAGS.multi_inputs, FLAGS.multi_inputs_srcs, FLAGS.merge_tokens)
  model = get_model()
  gezi.load_weights(model, model_dir)
  ic(gezi.get_mem_gb())
  # gc.collect()
  # continue
  display(pd.read_csv(f'{model_dir}/metrics.csv'))
  inputs = get_inputs(FLAGS.backbone, mode='train', sort=True, double_times=0, test_ids=test_ids)
  # inputs = get_inputs(FLAGS.backbone, mode='train', sort=False, double_times=0, test_ids=test_ids)
 
  # not for roberta-large tf can use 32 bs, torch can use even larger bs, for longformer tf only 8, torch can use 32
  #   batch_size = 32 if FLAGS.torch else 8
  batch_size = 32
  # gezi.save(inputs, '../working/inputs.pkl')
  p = gezi.predict(model, inputs, batch_sizes[i], dynamic_keys=['input_ids', 'word_ids'], mask_key='attention_mask')
  # p = predict(model, inputs, batch_sizes[i])
  p.update({
    'id': inputs['id'],
    'word_ids': inputs['word_ids'],
    'num_words': inputs['num_words']
  })
  convert_res(p)
  ensembler.add(p, weights[i])
  if len(inputs['id']) < 1000:
    df = get_preds(p)
    ic(df)
    # gezi.save_pickle(p, '../working/debug_valid.pkl')
    # gezi.save_pickle(inputs, '../working/debug_inputs.pkl')
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

p = ensembler.finalize()
df_gt = pd.read_csv('../input/feedback-prize-2021/train.csv')
df_gt = df_gt[df_gt.id.isin(test_ids)]
df_gt = df_gt.sort_values('id')
df_gt['num_words'] = df_gt.id.apply(lambda id: len(open(f'../input/feedback-prize-2021/train/{id}.txt').read().split()))
res = get_metrics(df_gt, p)
ic(res)


# In[ ]:


df = get_preds(p)
display(df)
df


# In[ ]:


df.to_csv(f'{model_dir}/submission.csv')


# In[ ]:





# In[ ]:




