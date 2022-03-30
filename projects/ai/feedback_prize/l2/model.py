#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   torch_model.py
#        \author   chenghuige  
#          \date   2021-12-29 05:57:04.508940
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from transformers import AutoModel, AutoConfig
import torch
from torch import nn
from torch.nn import functional as F
try:
  from torchcrf import CRF
except Exception:
  pass
try:
  import torch_scatter
except Exception:
  pass
import lele
from gezi import tqdm
from src.config import *
from src.torch.loss import *
from src import util

class Model(nn.Module):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)  
    Linear = nn.Linear
    backbone_name = 'microsoft/deberta-v3-large'
    self.backbone = AutoModel.from_pretrained(backbone_name)
    tokenizer = get_tokenizer(backbone_name)
    self.backbone.resize_token_embeddings(len(tokenizer)) 
    
    emb_dim = 128
    # idim = FLAGS.num_classes + emb_dim * 4 + 1 + 1
    idim = FLAGS.num_classes + 2
    # self.text_dense = Linear(self.backbone.config.hidden_size, idim)
    odim = 128
    # FLAGS.rnn_dropout = 0
    RNN = getattr(nn, FLAGS.rnn_type)
    self.seq_encoder = RNN(idim, int(odim / 2), FLAGS.rnn_layers, dropout=FLAGS.rnn_dropout, bidirectional=True, batch_first=True)
    
    # self.cls_emb = nn.Embedding(10, emb_dim, padding_idx=0)
    # self.len_emb = nn.Embedding(5000, emb_dim, padding_idx=0)
    # self.sep_emb = nn.Embedding(5000, emb_dim, padding_idx=0)
    # self.model_emb = nn.Embedding(100, emb_dim, padding_idx=0)
    # self.num_words_emb = nn.Embedding(300, emb_dim, padding_idx=0)
    
    dim = odim
    # dim = dim * 2 + emb_dim * 2  
    self.dense = Linear(dim, 1)
          
  def encode(self, inputs):
    bs = inputs['token_logits'].shape[0]
    # ic(inputs['token_logits'].shape)
    
    # input_ids = inputs['input_ids']
    # attention_mask = inputs['attention_mask']
    # max_len = attention_mask.sum(1).max()
    # input_ids = input_ids[:,:max_len]
    # attention_mask = attention_mask[:,:max_len]
    # input_ids = torch.cat([inputs['cls2'], input_ids], 1)
    # attention_mask = torch.cat([torch.ones_like(inputs['cls2']), attention_mask], 1)
    
    # text_emb = self.backbone(input_ids=input_ids, attention_mask=attention_mask)[0][:,0] 
    # text_emb = self.text_dense(text_emb)
    
    # token_logits = inputs['token_logits'].view(bs, FLAGS.max_parts, FLAGS.num_classes)
    # token_probs = inputs['token_probs'].view(bs, FLAGS.max_parts, FLAGS.num_classes)
    # cls_emb = self.cls_emb(inputs['cls'])
    # len_emb = self.len_emb(inputs['len'])
    # sep_emb = self.sep_emb(inputs['sep'])
    # model_emb = self.model_emb(inputs['models'])
    # ratios = inputs['ratio'].unsqueeze(-1)
    # sep_probs = inputs['sep_prob'].unsqueeze(-1)
    # x = torch.cat([token_probs, sep_probs, cls_emb, len_emb, sep_emb, model_emb, ratios], -1)
    
    # x = torch.cat([text_emb.unsqueeze(1), x], 1)
    # # ic(x.shape)
 
    # x, _ = self.seq_encoder(x)
    # x = torch.cat([x.mean(1), torch.max(x, 1)[0], self.model_emb(inputs['model'].squeeze(-1)), self.num_words_emb((inputs['num_words'].squeeze(-1) / 10).int())], -1)
      
    # x = torch.cat([x.mean(1), torch.max(x, 1)[0]], -1)
    token_logits = inputs['token_logits'].view(bs, FLAGS.max_words, FLAGS.num_classes)
    start_logits = inputs['start_logits'].view(bs, FLAGS.max_words, 2)
    x = torch.cat([token_logits, start_logits], -1)
    x, _ = self.seq_encoder(x)
    
    res = self.dense(x)   
    res = res.squeeze(-1)      
    return res
  
  def forward(self, inputs):     
    res = self.encode(inputs)
    return res
  
  def get_loss_fn(self):
    def loss_fn(y_pred, y_true, x):
      return nn.BCEWithLogitsLoss()(y_pred, y_true)
      # return nn.MSELoss()(y_pred, y_true)
    return loss_fn
