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
    
    self.inputs = None
    if not FLAGS.backbones:
      self.backbone, self.tokenizer = self.init_backbone(FLAGS.backbone)
    else:
      self.backbones, self.tokenizers = [], []
      for backbone_name, model_name in tqdm(zip(FLAGS.backbones, FLAGS.model_names), total=len(FLAGS.model_names)):
        model_dir = os.path.join(os.path.dirname(FLAGS.model_dir), model_name)
        load_weights = not os.path.exists(f'{FLAGS.model_dir}/model.pt')
        ic(backbone_name, model_dir, load_weights)
        backbone, tokenizer = self.init_backbone(backbone_name, model_dir, load_weights=load_weights)
        self.backbones.append(backbone)
        self.tokenizers.append(tokenizer)
      self.backbones = nn.ModuleList(self.backbones)
      self.backbone, self.tokenizer = self.backbones[0], self.tokenizers[0]
      
    if FLAGS.freeze_backbone:
      lele.freeze(self.backbone)
    
    if FLAGS.learn_weights:
      num_inputs = len(FLAGS.mis) + 1
      # https://github.com/pytorch/pytorch/issues/36035 bug here..
      # self.weights = nn.ParameterList([nn.Parameter(torch.ones(FLAGS.max_len)) for _ in range(num_inputs)])
      self.weights0 = nn.Parameter(torch.ones(FLAGS.max_len))
      self.weights1 = nn.Parameter(torch.ones(FLAGS.max_len))
      self.weights2 = nn.Parameter(torch.ones(FLAGS.max_len))
      self.weights3 = nn.Parameter(torch.ones(FLAGS.max_len))
    
    dim = self.backbone.config.hidden_size
    if FLAGS.use_wordids:
      self.wordid_emb = nn.Embedding(FLAGS.max_words, dim)
    if FLAGS.use_relative_positions:
      self.position_emb = nn.Embedding(FLAGS.num_positions + 1, dim, padding_idx=0)
    #   # dim *= 2
    
    if FLAGS.seq_encoder:
      if not FLAGS.rnn_stack:
        RNN = getattr(nn, FLAGS.rnn_type)
        if not FLAGS.rnn_bi:
          self.seq_encoder = RNN(dim, dim, FLAGS.rnn_layers, dropout=FLAGS.rnn_dropout, bidirectional=False, batch_first=True)
        else:
          self.seq_encoder = RNN(dim, int(dim / 2), FLAGS.rnn_layers, dropout=FLAGS.rnn_dropout, bidirectional=True, batch_first=True)
      else:
        self.seq_encoder = lele.layers.StackedBRNN(
            input_size=dim,
            hidden_size=int(dim / 2),
            num_layers=FLAGS.rnn_layers,
            dropout_rate=FLAGS.rnn_dropout,
            dropout_output=False,
            recurrent_dropout=False,
            concat_layers=True,
            rnn_type=FLAGS.rnn_type.lower(),
            padding=True,
        )    

    if FLAGS.crf_loss_rate > 0:
      self.crf = CRF(num_tags=FLAGS.num_classes, batch_first=True)
    
    Linear = nn.Linear if not FLAGS.mdrop else lele.layers.MultiDropout
    self.dense = Linear(dim, FLAGS.num_classes)
    if FLAGS.constant_init_bias:
      self.dense.bias.data = torch.as_tensor(np.log(para_len_ratio))
      ic(self.dense.bias)
    
    # TODO 二分类改回使用BCE 
    if FLAGS.num_classes == NUM_CLASSES:
      if not FLAGS.new_start:
        self.start_dense = Linear(dim, 2)
      else:
        self.start_dense = Linear(dim * 3, 2)
      if FLAGS.constant_init_bias:
        self.start_dense.bias.data = torch.as_tensor(np.log([0.982, 0.018]))
        ic(self.start_dense.bias)
    
    if FLAGS.end_loss_rate > 0:
      self.end_dense = Linear(dim, 2)
    
    if FLAGS.cls_para:
      self.para_dense = Linear(dim, FLAGS.num_classes)
    
    if FLAGS.cls_parts:
      # self.parts_pooling = lele.layers.Pooling('att', dim)
      self.parts_dense = nn.Sequential(Linear(dim, 1), nn.Sigmoid())
      
    if FLAGS.cls_loss_rate > 0:
      self.cls_dense = Linear(dim, NUM_CLASSES)
      
    if FLAGS.binary_loss:
      self.denses = nn.ModuleDict({
        cls_: Linear(dim, 2) for cls_ in classes
      })
    
    if FLAGS.keras_init:
      for key, m in self.named_children():
        if key not in ['backbone']:
          lele.keras_init(m)

    # if FLAGS.rewards:
    #   for name, p in self.named_parameters():
    #     patterns = ['dense.', 'start_dense.', 'seq_encoder.']
    #     if any(name.startswith(x) for x in patterns):
    #       ic(name)
    #       p.requires_grad = True
    #     else:
    #       p.requires_grad = False
          
  def init_backbone(self, backbone_name, model_dir=None, load_weights=False):
    config = AutoConfig.from_pretrained(backbone_name)
    if 'bigbird' in backbone_name:
      config.update({
        ## sparse att is so slow...
        # 'attention_type': 'original_full' if not FLAGS.bird_sparse_att else 'block_sparse',
        'attention_type': 'original_full',
        'block_size': FLAGS.block_size,
        'num_random_blocks': FLAGS.n_blocks,
      })
    max_len = FLAGS.max_len + int(FLAGS.num_words_emb)
    if FLAGS.update_config_maxlen and max_len > config.max_position_embeddings:
      # +1 for num_words_emb, maybe add more for reservation
      config.update({'max_position_embeddings': max_len})
    ic(config.max_position_embeddings)
    model_dir = model_dir or FLAGS.model_dir
    ic(os.path.exists(f'{model_dir}/model.pt'))
    if os.path.exists(f'{model_dir}/model.pt'):
      print('AutoModel from config')
      backbone = AutoModel.from_config(config)
    else:
      print('AutoModel from pretrained')
      #如果预测的时候用这个 好像会有缓存机制 特别多个不同backbone不会随着Model释放而释放！
      # self.backbone = AutoModel.from_pretrained(FLAGS.backbone) 
      if FLAGS.update_config:
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7
        config.update(
            {
                # "output_hidden_states": True,  # more gpu mem needed
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )
      backbone = AutoModel.from_pretrained(backbone_name, config=config, ignore_mismatched_sizes=True)
      # ic(backbone.device)
    
    tokenizer = get_tokenizer(backbone_name)
    if FLAGS.add_br:
      backbone.resize_token_embeddings(len(tokenizer)) 
      # TODO bart not ok
      unk_id = tokenizer.unk_token_id
      with torch.no_grad(): 
        word_embeddings = lele.get_word_embeddings(backbone)
        br_id = tokenizer.convert_tokens_to_ids(FLAGS.br)
        word_embeddings.weight[br_id, :] = word_embeddings.weight[unk_id, :]               
    
    if load_weights:
      lele.load_weights(backbone, f'{model_dir}/model.pt', renames={'backbone.': ''}, eval=False)
      lele.freeze(backbone)
    
    if FLAGS.gradient_checkpointing:
      backbone.gradient_checkpointing_enable()
    return backbone, tokenizer
  
  def get_backbone(self, idx=None):
    if idx is None:
      return self.backbone, self.tokenizer
    else:
      return self.backbones[idx], self.tokenizers[idx]
                
  def groupby(self, logits, word_ids, combiner='sum'):
    word_ids_ = word_ids + 1
    word_ids_ *= (word_ids_ < FLAGS.max_words).long()
    if FLAGS.mark_end:
      num_words = self.inputs['num_words'].view(-1, 1)
      logits = torch.cat([logits, torch.ones_like(logits[:,-1:])], 1)
      word_ids_ = torch.cat([word_ids_, torch.clamp(num_words + 1, max=FLAGS.max_words - 1)], 1)
    if self.training and FLAGS.torch_scatter:
      logits = torch_scatter.scatter(logits, word_ids_.long(), 1, dim_size=FLAGS.max_words + 1, reduce=combiner)
    else:
      logits = lele.unsorted_segment_reduce(logits, word_ids_.long(), FLAGS.max_words + 1, combiner=combiner)
    return logits[:, 1:]
  
  def encode(self, inputs, groupby=False, idx=None):
    backbone, tokenizer = self.get_backbone(idx)
    if self.training:
      if FLAGS.unk_aug_rate > 0:
        inputs['input_ids'] = util.unk_aug(inputs['input_ids'], inputs['attention_mask'], FLAGS.unk_aug_rate, tokenizer.unk_token_id)
    
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    # max_len = attention_mask.sum(1).max()
    # input_ids = input_ids[:,:max_len]
    # attention_mask = attention_mask[:,:max_len]
    
    if FLAGS.num_words_emb:
      num_words_id = (inputs['num_words'] / 100).int().detach().cpu().numpy().reshape(-1)
      # in training data max num_words is 1656 so 16.56 < 17 only use 0,1...16 will be fine
      num_words_id = np.asarray(self.tokenizer.convert_tokens_to_ids([f'[NWORDS{a}]' if a < 17 else f'[NWORDS16]' for a in num_words_id])).reshape(-1, 1)
      num_words_id = torch.as_tensor(num_words_id, device=input_ids.device, dtype=input_ids.dtype)
      input_ids = torch.cat([num_words_id, input_ids], 1)
      attention_mask = torch.cat([torch.ones_like(attention_mask[:,:1]), attention_mask], 1)
    if FLAGS.use_cluster:
      cluster_id = inputs['cluster'].detach().cpu().numpy().reshape(-1)
      cluster_id = np.asarray(self.tokenizer.convert_tokens_to_ids([f'[CLUSTER{a}]' for a in cluster_id])).reshape(-1, 1)
      # input_ids = torch.cat([inputs['cluster_id'], input_ids], 1)
      cluster_id = torch.as_tensor(cluster_id, device=input_ids.device, dtype=input_ids.dtype)
      input_ids = torch.cat([cluster_id, input_ids], 1)
      attention_mask = torch.cat([torch.ones_like(attention_mask[:,:1]), attention_mask], 1)
    x = backbone(input_ids=input_ids, attention_mask=attention_mask)[0]  
    if FLAGS.num_words_emb:
      x = x[:,1:]
    if FLAGS.use_cluster:
      x = x[:,1:]
    
    if groupby:
      x = self.groupby(x, inputs['word_ids'], combiner=FLAGS.word_combiner)
    return x
  
  def forward(self, inputs):
    self.inputs = inputs
    if FLAGS.fake_infer:
      input_ids =  inputs['input_ids'] if not 0 in inputs else inputs[0]['input_ids']
      bs = input_ids.shape[0] 
      if FLAGS.merge_tokens:
        width = FLAGS.max_words
      else:
        width = input_ids.shape[1]
      return {
        'pred': torch.rand([bs, width, 8], device=input_ids.device),
        'start_logits': torch.rand([bs, width, 2], device=input_ids.device),
        }
        
    if not 0 in inputs:
      x = self.encode(inputs, groupby=FLAGS.merge_tokens)
    else:
      xs = []
      if FLAGS.scatter_method == 0:
        word_ids_list = []
        for i, inputs_ in inputs.items():   
          if isinstance(i, int):
            idx = None if not FLAGS.model_names else i
            x = self.encode(inputs_, groupby=False, idx=idx)
            if FLAGS.learn_weights:
              # x *= self.weights[i]
              x *= getattr(self, f'weights{i}').unsqueeze(-1)
            xs.append(x)
            word_ids_list.append(inputs_['word_ids'])
        # 这种做法 index 不唯一了 https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html 会有问题吗？待验证效果
        # When indices are not unique, the behavior is non-deterministic (one of the values from src will be picked arbitrarily) 
        # and the gradient will be incorrect (it will be propagated to all locations in the source that correspond to the same index)!
        # 似乎预测确实每次结果都不太一样 但是变动不是太大 TODO
        x = torch.cat(xs, 1)
        word_ids = torch.cat(word_ids_list, 1)
        x = self.groupby(x, word_ids, combiner=FLAGS.word_combiner)
      else:
        for i, inputs_ in inputs.items():   
          if isinstance(i, int):
            xs.append(self.encode(inputs_, groupby=True))
        x = torch.stack(xs, 0).sum(0)
        if FLAGS.scatter_method == 2:
          ## TODO 这里归一化可能效果好一点点 但是infer麻烦一些 需要相关信息 暂时去掉
          x /= torch.unsqueeze(inputs['word_counts'], -1)
        elif FLAGS.scatter_method == 3:
          x /= len(xs)
    
    if FLAGS.merge_tokens:
      word_mask = (torch.abs(x).sum(-1, keepdims=True) > 0).float()
    
    if FLAGS.seq_encoder:
      if FLAGS.rnn_stack:
        if not FLAGS.merge_tokens:
          mask = inputs['attention_mask'].eq(0)
        else:
          ## TODO online word_ids not ready for infer now , change to use 1 - lele.sequence_mask(inputs['num_words'])
          # mask = inputs['word_ids'].eq(util.null_wordid())
          num_words = inputs['num_words'].view(-1, 1)
          mask = lele.sequence_mask(num_words, FLAGS.max_words).logical_not()  #TODO FIXME not work 在线报错 可能因为缺少[,1]
          mask = torch.zeros_like(mask).long()
          x = self.seq_encoder(x, mask)
      else:
        x, _ = self.seq_encoder(x)
            
    logits = self.dense(x)
    res = {
      'pred': logits
    }
    logits_list = gezi.get('xs', [])
    if logits_list:
      res['logits_list'] = logits_list
    
    if FLAGS.crf_loss_rate > 0:
      res['crf_loss'] = -1. * self.crf(emissions=logits, tags=inputs['label'].long(), 
                                       mask=inputs['mask'].byte())
      if not self.training:
        res['crf_tokens'] = self.crf.decode(emissions=logits, mask=inputs['mask'].byte())
    
    if FLAGS.num_classes == NUM_CLASSES:
      #---- here
      if FLAGS.new_start:
        x_pre = torch.cat([x[:,-1:], x[:,:-1]], 1)
        x_after = torch.cat([x[:,1:], x[:,0:1]], 1)
        x = torch.cat([x_pre, x, x_after], -1)
      start_logits = self.start_dense(x)
      res['start_logits'] = start_logits
      start_logits_list = gezi.get('xs', [])
      if start_logits_list:
        res['start_logits_list'] = start_logits_list
    else:
      if FLAGS.sum_for_token:
        res['pred'] = torch.stack([
          logits[:,:,i] + logits[:,:,i+1] for i in range(4)
        ], -1)
      if FLAGS.sum_for_sep:
        a = logits[:,:,0] + logits[:,:,2] + logits[:,:,4] + logits[:,:,6]
        # B start/sep
        b = logits[:,:,1] + logits[:,:,3] + logits[:,:,5] + logits[:,:,7]
        res['start_logits'] = torch.stack([a, b], -1)
    
    if FLAGS.end_loss_rate > 0:
      res['end_logits'] = self.end_dense(x)
            
    if FLAGS.cls_loss_rate > 0:
      x_cls = x[:,0]
      res['cls_logits'] = self.cls_dense(x_cls)
    
    if FLAGS.binary_loss:
      for cls_ in classes:
        if class_weights[cls_] > 0:
         res[f'{cls_}_logits'] = self.denses[cls_](x)
        
    # # if 'id' in inputs or 'y' in inputs:
    # if 'y' in inputs:
    #   if FLAGS.cls_parts:
    #     # x_ = self.parts_pooling(x, (1 - inputs['attention_mask']).bool())
    #     x_cls = x[:,0]
    #     res['parts'] = self.parts_dense(x_cls) * FLAGS.max_parts
    #   elif 'end_logits' in res:
    #     seps = ((res['end_logits'][:,:,1] - res['end_logits'][:,:,0]) > 0).int() * inputs['mask']
    #     res['parts'] = seps.sum(axis=1).float() 
    #   elif 'start_logits' in res:
    #     seps = ((res['start_logits'][:,:,1] - res['start_logits'][:,:,0]) > 0).int() * inputs['mask']
    #     res['parts'] = seps.sum(axis=1).float() 
    #   else:
    #     assert FLAGS.num_classes > NUM_CLASSES
    #     res['parts'] = (torch.argmax(logits, -1) % 2 == 1).int().sum(-1)
    
    #   # TODO FIXME cls para部分需要再检查 有bug
    #   if FLAGS.cls_para:        
    #     x = lele.unsorted_segment_sum(x, inputs['para_index'].long(), FLAGS.max_parts + 1)
    #     x = x[:, 1:, :]
    #     res['para_logits'] = self.para_dense(x) 
    #   else:
    #     probs = F.softmax(logits)
    #     if FLAGS.cls_loss_rate > 0:
    #       # TODO FIXME
    #       x = lele.unsorted_segment_sum(probs, torch.cumsum(seps, 1), FLAGS.max_parts + 1)
    #     else:
    #       x = lele.unsorted_segment_sum(probs, inputs['para_index'].long(), FLAGS.max_parts + 1)
    #       x = x[:, 1:, :]
        
    #     if not FLAGS.sum_for_para or x.shape[-1] == NUM_CLASSES:
    #       res['para_logits'] = x
    #     else:
    #       res['para_logits'] = torch.stack([
    #         x[:,:,i] + x[:,:,i+1] for i in range(4)
    #       ], -1)
    
    
    if FLAGS.merge_tokens:
      # check use all zero all mask, can get from inputs['mask'] but for easy infer just using mask generated after groupby
      # if 'mask' in inputs:
      #   mask = inputs['mask'].unsqueeze(-1).float()
      # else: 
      for key in res:
        # TODO 特别注意如果新加输出 要注意mask 掉无用的部分wordid
        if key in ['pred', 'start_logits', 'cls_logits']:
          try:
            res[key] *= word_mask
          except Exception as e:
            ic(e, key, res[key].shape, word_mask.shape)
            exit(0)
           
    return res
  
  def calc_rdrop_loss(self, res1, res2, x):
    token_loss = lele.losses.compute_kl_loss(res1['pred'], res2['pred'], pad_mask=x['mask'].unsqueeze(-1).bool())
    start_loss = lele.losses.compute_kl_loss(res1['start_logits'], res2['start_logits'], pad_mask=x['mask'].unsqueeze(-1).bool())
    loss = token_loss * FLAGS.token_loss_rate + start_loss * FLAGS.start_loss_rate
    return loss
  
  def get_loss_fn(self):
    from src.torch.loss import calc_loss
    return calc_loss
