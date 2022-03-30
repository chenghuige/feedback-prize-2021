#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2021-12-31 03:24:41.340166
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
from absl import flags
import torch
from torch import nn
from torch.nn import functional as F
import lele
import gezi
from src.config import *
from src.rewards import calc_reward
from src.torch.decode import *

def calc_rl_loss(start_logits, token_logits, x, scalars):
  greedy_f1, sample_f1, greedy_starts, sample_starts, greedy_tokens, sample_tokens = \
                    calc_reward(x['sep'].detach().cpu().numpy(), 
                                x['label'].detach().cpu().numpy(), 
                                start_logits.detach().cpu().numpy(),
                                token_logits.detach().cpu().numpy(), 
                                x['word_ids'].detach().cpu().numpy())
  # ic(greedy_starts, sample_starts)
  greedy_starts = torch.as_tensor(greedy_starts, device=start_logits.device).long()
  sample_starts = torch.as_tensor(sample_starts, device=start_logits.device).long()
  greedy_tokens = torch.as_tensor(greedy_tokens, device=start_logits.device).long()
  sample_tokens = torch.as_tensor(sample_tokens, device=start_logits.device).long()
  
  # ic(x['sep'][0], greedy_starts[0], sample_starts[0])
    
  scalars['greedy/f1'] = greedy_f1
  scalars['sample/f1'] = sample_f1
  loss_obj = nn.CrossEntropyLoss(reduction='none')
  loss_obj2 = nn.CrossEntropyLoss(reduction='none')
  
  rate = FLAGS.rl_token_loss_rate
  bs = start_logits.shape[0]
  sample_start_loss = loss_obj(start_logits.view(-1, 2), sample_starts.view(-1))
  sample_start_loss = lele.masked_mean(sample_start_loss, x['mask'])
  scalars['sample/start/loss'] = sample_start_loss.detach().cpu().numpy()
  sample_token_loss = loss_obj2(token_logits.view(-1, FLAGS.num_classes), sample_tokens.view(-1))
  sample_token_loss = lele.masked_mean(sample_token_loss, x['mask'])
  scalars['sample/token/loss'] = sample_token_loss.detach().cpu().numpy()
  sample_loss = sample_start_loss * (1 - rate) + sample_token_loss * rate
  sample_reward = sample_f1 - greedy_f1
  scalars['sample/reward'] = sample_reward
  sample_loss *= sample_reward
  scalars['sample/loss'] = sample_loss.detach().cpu().numpy()
  
  greedy_start_loss = loss_obj(start_logits.view(-1, 2), greedy_starts.view(-1))
  greedy_start_loss = lele.masked_mean(greedy_start_loss, x['mask'])
  scalars['greedy/start/loss'] = greedy_start_loss.detach().cpu().numpy()
  greedy_token_loss = loss_obj2(token_logits.view(-1, FLAGS.num_classes), greedy_tokens.view(-1))
  greedy_token_loss = lele.masked_mean(greedy_token_loss, x['mask'])
  scalars['sample/token/loss'] = greedy_token_loss.detach().cpu().numpy()
  greedy_loss = greedy_start_loss * (1 - rate) + greedy_token_loss * rate
  greedy_reward = greedy_f1 - sample_f1
  scalars['greedy/reward'] = greedy_reward
  greedy_loss *= greedy_reward
  scalars['greedy/loss'] = greedy_loss.detach().cpu().numpy()
  
  rl_loss = ((sample_loss + greedy_loss) / 2.)
    
  rl_loss *= FLAGS.rl_loss_scale
  scalars['loss/rl'] = rl_loss.detach().cpu().numpy()
  return rl_loss
    
def calc_loss(res, y, x, step=None, epoch=None, training=None):
  scalars = {}
  y_ = res['pred']
  # step = x['step']
  # if step < 200:
  #   weight = np.zeros(NUM_CLASSES)
  #   weight[2] = 1.nn
  #   weight = torch.as_tensor(weight, dtype=torch.float32, device=y_.device)
  # else:
  #   weight = None
  
  if FLAGS.mask_rare:
    if step < 500:
      x['mask'] *= (x['label'] != 7).long()
      x['mask'] *= (x['label'] != 6).long()
  
  loss = 0.
  if FLAGS.token_loss_rate > 0:
    weight = None
    if FLAGS.class_weights:
      class_weights_ = list(map(float, FLAGS.class_weights))
      class_weights_ = torch.as_tensor(class_weights_, device=y.device)
      weight = class_weights_
    loss_obj = nn.CrossEntropyLoss(weight=weight, label_smoothing=FLAGS.token_label_smoothing or FLAGS.label_smoothing, reduction='none')
    # loss_obj = nn.CrossEntropyLoss(reduction='none')
    if not 'logits_list' in res:
      loss = loss_obj(y_.view(-1, FLAGS.num_classes), y.view(-1))
    else:
      loss_list = [loss_obj(logits.view(-1, FLAGS.num_classes), y.view(-1)) for logits in res['logits_list']]
      loss = torch.stack(loss_list, 1).mean(1)
    loss *= FLAGS.token_loss_rate
    token_loss = lele.masked_mean(loss, x['mask'])
    scalars['loss/token'] = token_loss.detach().cpu().numpy()
  
  if FLAGS.binary_loss:
    loss_obj = nn.CrossEntropyLoss(reduction='none')
    binary_loss = torch.zeros_like(loss)
    for i, cls_ in enumerate(classes):
      if class_weights[cls_] > 0:
        cls_logits_ = res[f'{cls_}_logits']
        cls_label_ = (y == i + 1).long()
        cls_logits = cls_logits_.view(-1, 2)
        cls_label = cls_label_.view(-1)
        loss_ = loss_obj(cls_logits, cls_label)
        loss_ *= class_weights[cls_]
        loss_ *= FLAGS.binary_loss_rate
        binary_loss += loss_
    loss += binary_loss
    scalars['loss/binary'] = binary_loss.mean().detach().cpu().numpy()
  
  if FLAGS.focal_loss_rate > 0:
    floss_obj = lele.losses.FocalLoss(reduction='none')
    focal_loss = floss_obj(y_.view(-1, FLAGS.num_classes), y.view(-1))
    focal_loss *= FLAGS.focal_loss_rate
    loss += focal_loss
    scalars['loss/focal'] = focal_loss.mean().detach().cpu().numpy()
  
  if FLAGS.start_loss:
    # if step < 400:
    #   start_loss_rate = 0
    # else:
    #   start_loss_rate = FLAGS.start_loss_rate
    start_loss_rate = FLAGS.start_loss_rate
    if FLAGS.start_loss_step and step < FLAGS.start_loss_step:
      start_loss_rate = 0.
    loss_obj = nn.CrossEntropyLoss(label_smoothing=FLAGS.start_label_smoothing or FLAGS.label_smoothing, reduction='none')
    # loss_obj = nn.CrossEntropyLoss(reduction='none')
    if not 'start_logits_list' in res:
      start_logits = res['start_logits'].view(-1, 2)
      if not FLAGS.soft_start:
        start_loss = loss_obj(start_logits, x['start'].long().view(-1))
      else:
        start_loss = loss_obj(start_logits, x['start'].view(-1, 2))
        # ic(start_logits[0], x['start'].view(-1, 2)[0])
    else:
      if not FLAGS.soft_start:
        loss_list = [loss_obj(logits.view(-1, 2), x['start'].long().view(-1)) for logits in res['start_logits_list']]
      else:
        loss_list = [loss_obj(logits.view(-1, 2), x['start'].view(-1, 2)) for logits in res['start_logits_list']]
      start_loss = torch.stack(loss_list, 1).mean(1)
    #mask for word id 0
    start_loss *= x['start_mask'].view(-1)
    start_loss *= start_loss_rate
    loss += start_loss
    start_loss = lele.masked_mean(start_loss, x['mask'])
    # ic(FLAGS.start_loss_rate, start_loss)
    scalars['loss/start'] = start_loss.detach().cpu().numpy()
  
  if FLAGS.end_loss_rate > 0:
    end_logits = res['end_logits'].view(-1, 2)
    end_loss = loss_obj(end_logits, x['end'].long().view(-1))
    end_loss *= x['end_mask'].view(-1)
    end_loss *= FLAGS.end_loss_rate
    loss += end_loss
    end_loss = lele.masked_mean(end_loss, x['mask'])
    scalars['loss/end'] = end_loss.detach().cpu().numpy()
                
  if FLAGS.drop_worst_rate:
    loss = torch.topk(torch.nonzero(loss.view(-1) * x['mask'].view(-1)), k=int(loss.shape[0] * (1 - FLAGS.drop_worst_rate)), largest=False)[0].mean()
  else:
    loss = lele.masked_mean(loss, x['mask'])
    
  if FLAGS.dice_loss_rate > 0.:
    dloss = lele.losses.dice_loss(F.softmax(y_, -1), lele.one_hot(y, FLAGS.num_classes), ignore_background=FLAGS.ignore_background).mean()
    dloss *= FLAGS.dice_loss_rate
    scalars['loss/dice'] = dloss.detach().cpu().numpy()
    loss += dloss

  if FLAGS.dice_loss2_rate > 0.:
    dloss = lele.losses.dice_loss2(F.softmax(y_, -1), lele.one_hot(y, FLAGS.num_classes), ignore_background=FLAGS.ignore_background).mean()
    dloss *= FLAGS.dice_loss2_rate
    scalars['loss/dice'] = dloss.detach().cpu().numpy()
    loss += dloss

  if FLAGS.lovasz_loss_rate > 0.:
    lovasz_loss = lele.losses.lovasz_softmax_flat(F.softmax(y_, -1).view(-1, FLAGS.num_classes), y.view(-1), ignore_background=FLAGS.ignore_background)
    lovasz_loss *= FLAGS.lovasz_loss_rate
    scalars['loss/lovasz'] = lovasz_loss.detach().cpu().numpy()
    loss += lovasz_loss
            
  if FLAGS.parts_loss:
    parts_loss_obj = nn.MSELoss(reduction='none')
    pred_parts = res['parts'] / float(FLAGS.max_parts)
    true_parts = x['parts_count'].float() / float(FLAGS.max_parts)
    # ic(list(zip(x['para_count'], res['parts'])))
    parts_loss = parts_loss_obj(pred_parts, true_parts).mean() 
    parts_loss *= FLAGS.parts_loss_rate
    scalars['loss/parts'] = parts_loss.detach().cpu().numpy()
    loss += parts_loss 
    
  if FLAGS.para_loss:
    para_logits = res['para_logits'].view(-1, FLAGS.num_classes)
    para_label = x['para_type'].long().view(-1)
    loss_obj3 = nn.CrossEntropyLoss(reduction='none')
    para_loss = loss_obj3(para_logits, para_label)
    para_loss = (para_loss * x['para_mask'].view(-1)).sum() / x['para_mask'].sum()
    para_loss *= FLAGS.para_loss_rate
    scalars['loss/para'] = para_loss.detach().cpu().numpy()
    loss += para_loss
    
  if FLAGS.cls_loss_rate > 0.:
    cls_loss_obj = nn.BCEWithLogitsLoss()
    # ic(list(zip(x['classes'][0], gezi.sigmoid(res['cls_logits'][0].detach().cpu().numpy()))))
    cls_loss = cls_loss_obj(res['cls_logits'], x['classes'].float())
    cls_loss *= FLAGS.cls_loss_rate
    scalars['loss/cls'] = cls_loss.detach().cpu().numpy()
    loss += cls_loss
    
  if FLAGS.crf_loss_rate > 0:
    res['crf_loss'] = res['crf_loss'].mean()
    scalars['loss/crf'] = res['crf_loss'].detach().cpu().numpy()
    loss += res['crf_loss'] * FLAGS.crf_loss_rate
  
  loss *= FLAGS.loss_scale

  if FLAGS.rewards or FLAGS.calc_rewards:
    rl_loss = calc_rl_loss(res['start_logits'], y_, x, scalars)
    if FLAGS.rewards:
      if epoch > FLAGS.rl_start_epoch:
        loss = loss * (1 - FLAGS.rl_loss_rate)  + rl_loss * FLAGS.rl_loss_rate
          
  if not training:
    scalars = gezi.dict_prefix(scalars, 'val_')
  scalars_ = gezi.get('scalars', {})
  scalars_.update(scalars)
  gezi.set('scalars', scalars_)
  
  return loss