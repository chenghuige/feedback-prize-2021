#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2021-12-15 15:26:29.100495
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app, flags

FLAGS = flags.FLAGS

import numpy as np
import gezi
import melt as mt
from transformers import AutoConfig, AutoTokenizer
# from src import util

RUN_VERSION = '61'
PREFIX = ''

# records related
flags.DEFINE_string('records_type', 'token', 'by word or by token')
flags.DEFINE_alias('rt', 'records_type')
flags.DEFINE_string('records_name', 'tfrecords', '')
flags.DEFINE_string('records_version', '', '')
flags.DEFINE_bool('filter_records', False, '')
flags.DEFINE_alias('rv', 'records_version')
flags.DEFINE_integer('max_parts', 40, 'max num of parts to split, or max paras, here para means many paras to form 1 part/para')
flags.DEFINE_string('idir', '../input/feedback-prize-2021', '')
flags.DEFINE_integer('max_len', None, '1280 pt.large.part_loss achive 680 pb, but 1024 is faster')
flags.DEFINE_integer('max_len_valid', None, 'for valid')
flags.DEFINE_integer('max_words', 2048, '')
flags.DEFINE_alias('ml', 'max_len')
flags.DEFINE_bool('mask_inside', True, '')
flags.DEFINE_alias('mi', 'mask_inside')
flags.DEFINE_bool('mask_more', False, '')
flags.DEFINE_bool('label_inside', False, '')
flags.DEFINE_alias('li', 'label_inside')
flags.DEFINE_string('merge_br', None, 'not merge as None, end merge as end, start merge as start this might be useful')
flags.DEFINE_string('br', '[BR]', 'set to [BR]')
flags.DEFINE_bool('add_br', True, '')
flags.DEFINE_bool('ori_br', False, 'use original \n instead of [BR]')
flags.DEFINE_string('xa0', '[XA0]', '')
flags.DEFINE_bool('add_special_tokens', True, '')
flags.DEFINE_bool('split_punct', False, '')
flags.DEFINE_bool('custom_tokenize', False, '')
flags.DEFINE_alias('ct', 'custom_tokenize')
flags.DEFINE_bool('ori_deberta_v2_tokenizer', False, '')
flags.DEFINE_alias('odvt', 'ori_deberta_v2_tokenizer')
flags.DEFINE_bool('no_cls', False, '')
flags.DEFINE_bool('corrected', True, '')
flags.DEFINE_bool('encode_cls', False, '')

flags.DEFINE_bool('add_prefix_space', True, '')
flags.DEFINE_bool('up_sample', False, '')
flags.DEFINE_bool('remove_br', False, '')
flags.DEFINE_string('aug', 'en', '')
flags.DEFINE_string('split_method', 'start', 'start end middle se')
flags.DEFINE_alias('sm', 'split_method')
flags.DEFINE_integer('last_tokens', 256, '')
flags.DEFINE_integer('last_tokens2', 384, '')
flags.DEFINE_integer('last_tokens3', 448, '')
flags.DEFINE_integer('last_tokens4', 480, '')
flags.DEFINE_integer('last_tokens5', 128, '')
flags.DEFINE_integer('last_tokens6', 64, '')
flags.DEFINE_integer('last_tokens7', 32, '')
flags.DEFINE_bool('normalize', False, '')
flags.DEFINE_alias('norm', 'normalize')
flags.DEFINE_bool('tolower', False, '')
flags.DEFINE_alias('lower', 'tolower')
flags.DEFINE_bool('remove_punct', False, '')
flags.DEFINE_bool('fix_misspell', False, '')
flags.DEFINE_bool('norm_punct', False, '')
flags.DEFINE_bool('force_special', True, '')
flags.DEFINE_integer('stride', None, '128')
flags.DEFINE_string('stride_combiner', 'mean', '')
flags.DEFINE_bool('use_stride_id', False, '')

flags.DEFINE_bool('tf', False, 'by default to be torch')
flags.DEFINE_bool('tf_dataset', False, '')

flags.DEFINE_string('backbone', None, '')
flags.DEFINE_list('backbones', [], '')
flags.DEFINE_list('pretrains', [], '')
flags.DEFINE_string('hug', 'deberta-v3', 'roberta-large作为基本模型速度效果均衡,electra容易崩溃')
flags.DEFINE_list('hugs', [], '')
flags.DEFINE_integer('num_classes', None, '')
flags.DEFINE_bool('mdrop', False, '')
flags.DEFINE_integer('num_positions', 3, '')
flags.DEFINE_bool('use_relative_positions', False, '')
flags.DEFINE_alias('urp', 'use_relative_positions')
flags.DEFINE_bool('num_words_emb', True, '')
flags.DEFINE_alias('nwemb', 'num_words_emb')
flags.DEFINE_bool('mark_end', True, '')

flags.DEFINE_integer('method', 2, '1: 16 class IB NothingI, NothingB ..., 2: 8 class 同时加判断sep的loss输出start_logits, 3: 按照discourse start 和 end, 4: 按照start，end')
flags.DEFINE_bool('cls_para', False, '')
flags.DEFINE_bool('cls_parts', False, '')
flags.DEFINE_bool('continue_pretrain', False, '')
flags.DEFINE_alias('cptrain', 'continue_pretrain')

flags.DEFINE_bool('sum_for_sep', False, 'BI 模型是max prob判断sep还是按sum logits来判断')
flags.DEFINE_bool('sum_for_token', False, '')
flags.DEFINE_bool('sum_for_para', False, '')

flags.DEFINE_bool('merge_tokens', False, '模型中使用word级别合并信息')

flags.DEFINE_bool('soft_start', False, '')
flags.DEFINE_string('loss_fn', 'softmax', '')
flags.DEFINE_bool('mask_loss', True, '')
flags.DEFINE_string('loss_reduction', 'mean', '')
flags.DEFINE_string('loss_method', 'per_token', '')
flags.DEFINE_bool('parts_loss', False, 'latest pt.large.max_len=1280 show 0.672 improve to 0.68 but local cv not show as much improvment, anyway change by True by defualt from v13')
flags.DEFINE_bool('start_loss', False, '')
flags.DEFINE_integer('start_loss_step', None, '')
flags.DEFINE_bool('new_start', False, '')
flags.DEFINE_bool('new_token', False, '')
flags.DEFINE_float('end_loss_rate', 0., '')
flags.DEFINE_bool('para_loss', False, '')
flags.DEFINE_float('token_loss_rate', 1., '')
flags.DEFINE_float('start_loss_rate', 10., '')
flags.DEFINE_float('parts_loss_rate', 10., '')
flags.DEFINE_float('para_loss_rate', 1., '')
flags.DEFINE_float('cls_loss_rate', 0, '')
flags.DEFINE_string('para_pooling', 'sum', '')
flags.DEFINE_bool('reuse_token_dense', False, '')
flags.DEFINE_bool('binary_loss', False, '')
flags.DEFINE_float('binary_loss_rate', 1., '')
flags.DEFINE_bool('ignore_background', False, 'ignore Nothing/background loss')
flags.DEFINE_float('drop_worst_rate', 0, '')
flags.DEFINE_list('class_weights', [], '')

flags.DEFINE_float('dice_loss_rate', 0., '')
flags.DEFINE_float('dice_loss2_rate', 0., '')
flags.DEFINE_float('focal_loss_rate', 0., '')
flags.DEFINE_float('lovasz_loss_rate', 1., '')
flags.DEFINE_alias('llr', 'lovasz_loss_rate')
flags.DEFINE_float('start_weight', 1., '')

flags.DEFINE_bool('exp', False, '')
flags.DEFINE_bool('keras_init', False, '')
flags.DEFINE_float('backbone_lr', None, '')
flags.DEFINE_float('base_lr', 1e-4, '')
flags.DEFINE_bool('half_lr', False, '注意默认改为True 为了保持之前lr设置不再变动实际lr都再/2，对应batch size也变成8')

flags.DEFINE_float('crf_loss_rate', 0., '')
flags.DEFINE_bool('mlp', False, '')

flags.DEFINE_string('word_combiner', 'mean', '')
flags.DEFINE_string('seg_reduce_method', 'sum', '')
flags.DEFINE_alias('srm', 'seg_reduce_method')
flags.DEFINE_string('mask_inside_method', 'first', 'first, last, mean, sum')
flags.DEFINE_alias('mim', 'mask_inside_method')
flags.DEFINE_string('post_reduce_method', 'first', '')
flags.DEFINE_alias('prm', 'post_reduce_method')
flags.DEFINE_string('post_reduce_method2', 'first', '')
flags.DEFINE_alias('prm2', 'post_reduce_method2')
flags.DEFINE_bool('mask_rare', False, '')

# post rule
flags.DEFINE_bool('eval_ori', False, '')
flags.DEFINE_alias('evo', 'eval_ori')
flags.DEFINE_bool('eval_len', True, '')
flags.DEFINE_bool('link_evidence', False, '')
flags.DEFINE_alias('linkev', 'link_evidence')
flags.DEFINE_bool('post_adjust', True, '')
flags.DEFINE_integer('para_min_len', 2, '')
flags.DEFINE_integer('para_min_len2', 6, 'for last seg/para')
flags.DEFINE_bool('adjacent_rule', True, '')
flags.DEFINE_bool('adjacent_rebuttal', True, '')
flags.DEFINE_bool('adjacent_prob', True, '')
flags.DEFINE_bool('adjacent_minthre', True, '')
flags.DEFINE_bool('sep_rule', True, '')
flags.DEFINE_bool('ensemble_pred', True, '')
flags.DEFINE_bool('token2word', True, 'for post deal if token2word first or not, token2word默认 更方便不同分词模型结果的集成')
flags.DEFINE_bool('out_overlap', False, '')

flags.DEFINE_string('pred_method', 'start', 'start, end, se')
flags.DEFINE_string('ensemble_method', 'prob', 'logit or prob')
flags.DEFINE_bool('ensemble_weight_per_word', True, '')
flags.DEFINE_alias('ewpw', 'ensemble_weight_per_word')
flags.DEFINE_bool('new_prob', False, '')
flags.DEFINE_bool('ensemble_weight', True, '')

flags.DEFINE_bool('use_wordids', False, '')
flags.DEFINE_bool('use_layernorm', False, '')

# flags.DEFINE_bool('mix_train', False, '')
flags.DEFINE_bool('tiny', False, '')
flags.DEFINE_alias('fast', 'tiny')
flags.DEFINE_bool('test', False, '')
flags.DEFINE_bool('pymp', True, '')
flags.DEFINE_float('min_save_score', 0.67, '')
flags.DEFINE_bool('eval_left', False, '')

flags.DEFINE_float('unk_aug_rate', 0., '')
flags.DEFINE_bool('force_cls_first', True, '')
flags.DEFINE_list('records_names', [], '')
flags.DEFINE_alias('rns', 'records_names')
flags.DEFINE_string('sampler', None, '')
flags.DEFINE_list('dataset_indexes', None, 'by default 25%')
flags.DEFINE_alias('dsi', 'dataset_indexes')
flags.DEFINE_float('aug_start_epoch', 0, '')
flags.DEFINE_alias('augse', 'aug_start_epoch')
flags.DEFINE_bool('aug_train', False, '')
flags.DEFINE_list('augs', [], '')

flags.DEFINE_bool('aug_nopunct', False, '')
flags.DEFINE_bool('aug_swap', False, '')
flags.DEFINE_bool('aug_lang', False, '')
flags.DEFINE_bool('dataset_per_epoch', False, '')
flags.DEFINE_alias('dpe', 'dataset_per_epoch')
flags.DEFINE_list('aug_rates', None, '')
flags.DEFINE_integer('max_augs', 10, '')
flags.DEFINE_bool('aug_split', False, '')
flags.DEFINE_bool('aug_lower', False, '')

flags.DEFINE_bool('best', False, '')
flags.DEFINE_bool('abhishek', True, '')
flags.DEFINE_bool('rewards', False, '')
flags.DEFINE_alias('rw', 'rewards')
flags.DEFINE_alias('rl', 'rewards')
flags.DEFINE_bool('sample_tokens', False, '')
flags.DEFINE_bool('rl_prob_thre', True, '')
flags.DEFINE_bool('calc_rewards', False, '')
flags.DEFINE_float('rl_token_loss_rate', 0., '0.1')
flags.DEFINE_float('rl_loss_scale', 1000, '')
flags.DEFINE_float('rl_start_epoch', 1, '')
flags.DEFINE_float('rl_loss_rate', 0.5, '')
flags.DEFINE_bool('weight_decay', False, '应该影响很小 或者没有正向作用')
flags.DEFINE_bool('lr_decay', True, '待验证 应该影响很小')
flags.DEFINE_bool('freeze_backbone', False, '')
flags.DEFINE_bool('learn_weights', False, '')

#for bigbird
flags.DEFINE_integer('block_size', 16, '')
flags.DEFINE_integer('n_blocks', 2, '')
flags.DEFINE_bool('bird_sparse_att', False, '')
flags.DEFINE_bool('bird_ori', False, '')

flags.DEFINE_bool('constant_init_bias', False, '')
flags.DEFINE_float('label_smoothing', 0., '')
flags.DEFINE_float('token_label_smoothing', 0., '')
flags.DEFINE_float('start_label_smoothing', 0., '')

flags.DEFINE_bool('multi_inputs', False, '')
flags.DEFINE_alias('mui', 'multi_inputs')
flags.DEFINE_list('multi_inputs_srcs', [], '')
flags.DEFINE_alias('mis', 'multi_inputs_srcs')
flags.DEFINE_string('rnn_type', 'LSTM', '')
flags.DEFINE_bool('rnn_stack', False, '')
flags.DEFINE_bool('rnn_bi', False, '')
flags.DEFINE_integer('rnn_layers', 1, '')
flags.DEFINE_float('rnn_dropout', 0.1, '')
flags.DEFINE_bool('seq_encoder', True, '')
flags.DEFINE_alias('sencoder', 'seq_encoder')

flags.DEFINE_bool('save_final', False, '')
flags.DEFINE_alias('sf', 'save_final')
flags.DEFINE_bool('save_fp16', True, '')

flags.DEFINE_bool('torch_scatter', False, '')
flags.DEFINE_integer('scatter_method', 0, '')

flags.DEFINE_bool('update_config', True, '')
flags.DEFINE_bool('update_config_maxlen', False, '')
flags.DEFINE_alias('ucm', 'update_config_maxlen')
flags.DEFINE_list('show_keys', [], '')
flags.DEFINE_bool('save_pred', False, '')
flags.DEFINE_string('save_pred_name', None, '')

# optuna
flags.DEFINE_integer('max_models', None, '')
flags.DEFINE_bool('suggest_uniform', False, '')
flags.DEFINE_bool('suggest_pre_after', True, '')
flags.DEFINE_list('ignored_folds', [], 'set [0, 1] for safe')
flags.DEFINE_integer('n_trials', 100, '')
flags.DEFINE_alias('ntr', 'n_trials')
flags.DEFINE_bool('all_models', False, '')
flags.DEFINE_alias('allm', 'all_models')
flags.DEFINE_integer('min_model_weight', 0, '')
flags.DEFINE_integer('max_model_weight', 10, '')
flags.DEFINE_bool('float_model_weight', False, '')
flags.DEFINE_integer('prob_idx', None, '')
flags.DEFINE_integer('model_idx', None, '')
flags.DEFINE_string('model_name_', None, '')
flags.DEFINE_integer('lens', None, '')

flags.DEFINE_integer('max_eval', None, '')
flags.DEFINE_bool('votes', False, '')
flags.DEFINE_bool('first_fold_only', False, '')
flags.DEFINE_alias('ffo', 'first_fold_only')
flags.DEFINE_integer('eval_folds', None, '')

flags.DEFINE_bool('fake_infer', False, '')
flags.DEFINE_bool('use_cluster', False, '')

classes = [
  'Claim', 'Evidence', 'Position', 'Concluding Statement', 'Lead', 'Counterclaim', 'Rebuttal'
]

CLASSES = classes
all_classes = ['Nothing', *classes]
ALL_CLASSES = all_classes

id2dis =  {
  0: 'Nothing',
  1: 'Claim',
  2: 'Evidence',
  3: 'Position',
  4: 'Concluding Statement',
  5: 'Lead',
  6: 'Counterclaim',
  7: 'Rebuttal'
 }

dis2id = {
  k: v for v, k in id2dis.items()
}

NUM_CLASSES = len(id2dis)

class_weights =  {
  'Claim': 0,
  'Evidence': 0,
  'Position': 0,
  'Concluding Statement': 0,
  'Lead': 0,
  'Counterclaim': 0,
  'Rebuttal': 1,
 }

# from https://www.kaggle.com/abhishek/two-longformers-are-better-than-1 but not used 
# about 1-1.5k offline

text_count_ratio = [1.0,
                    0.9572271386430679,
                    0.9971784019494677,
                    0.9844812107220726,
                    0.8474413235859947,
                    0.5964473515454662,
                    0.29267667051430035,
                    0.22944722329100936
                    ]

label_count_ratio = [
              0.17623337623337623,
              0.28704704704704703,
              0.26125554125554123,
              0.08812812812812813,
              0.07625911625911626,
              0.05322465322465322,
              0.03318175318175318,
              0.02467038467038467
              ]

para_len_ratio = [0.04887483055161287,
                  0.13275869946713112,
                  0.5403964820927039,
                  0.042693021091391656,
                  0.12092863276774386,
                  0.07492884742134961,
                  0.021118189856424607,
                  0.018301296751642427
                  ]

# 目前可选均使用large
## 1280 1536模型
# longformer  
## ------512 模型 start,end,se,mid
# roberta-squad  2gpu 3.2it/s # 较为平衡 开发主力
# electra  2gpu 3.2 it/s  #有启动后不收敛概率 如果收敛效果ok
# bart 2gpu 2.65 it/s # 效果相对差一点 集成差异
# deberta 2gpu 1.98it/s 效果相对最好 
# TODO 考虑deberta更大的模型搭配更小学习率 更小batch size

backbones = {
  'longformer': 'allenai/longformer-large-4096',  # ok 但是注意tf版本的longformer启动特别慢，虽然运行比torch稍微快一些, tf训练也更加占用显存 不能4A100训练bs16 torch可以
  'large': 'allenai/longformer-large-4096',  
  'longformer-base': 'allenai/longformer-base-4096',  # ok
  'base': 'allenai/longformer-base-4096',  
  'tiny': 'microsoft/deberta-v3-base',
  'roberta': 'roberta-large', # ok
  'mid': 'roberta-large',
  'electra': 'google/electra-large-discriminator', # ok electra版本效果不错 甚至好于roberta512 但是某些fold有很大概率loss不下降崩溃 没找到原因 但是online全量训练应该还好可以看train自己loss是否正常 在线提交融合结果也还ok
  'electra-base': 'google/electra-base-discriminator',
  'bart': 'facebook/bart-large', # okk
  'bart-base': 'facebook/bart-base',
  'large-qa': 'allenai/longformer-large-4096-finetuned-triviaqa',
  'roberta-base': 'roberta-base',
  'ro': 'roberta-base',
  'fast': 'roberta-base',
  'roberta-large': 'roberta-large',
  'bird': 'google/bigbird-roberta-large', # ok but not good
  'bigbird': 'google/bigbird-roberta-large',
  'bird-large': 'google/bigbird-roberta-large',
  'bird-base': 'google/bigbird-roberta-base',
  'pegasus': 'google/pegasus-large',
  'scico': 'allenai/longformer-scico',
  'xlnet-large': 'xlnet-large-cased',
  'xlnet': 'xlnet-large-cased',
  'robertam': 'roberta-large-mnli',
  'robertas': 'deepset/roberta-large-squad2',
  'roberta-squad': 'deepset/roberta-large-squad2',
  'reformer': 'google/reformer-enwik8', #fail
  'roformer': 'junnyu/roformer_chinese_base',
  'span': 'SpanBERT/spanbert-large-cased', # need --br='[SEP]'
  'gpt2': 'gpt2-large', # not well
  'berts': 'phiyodr/bart-large-finetuned-squad2',
  'barts': 'phiyodr/bart-large-finetuned-squad2',
  'bart-squad': 'phiyodr/bart-large-finetuned-squad2', # this is file
  'albert': 'albert-large-v2',
  'bert-cased': 'bert-large-cased',
  'bert-uncased': 'bert-large-uncased',
  'bert-squad': 'deepset/bert-large-uncased-whole-word-masking-squad2', # fail
  't5': 't5-large',
  'base-squad': 'valhalla/longformer-base-4096-finetuned-squadv1',
  'albert-squad': 'mfeb/albert-xxlarge-v2-squad2',
  'electra-squad': 'ahotrod/electra_large_discriminator_squad2_512',
  'deberta': 'microsoft/deberta-large',
  'deberta-base': 'microsoft/deberta-base',
  'deberta-xl': 'microsoft/deberta-xlarge',
  'deberta-xlarge': 'microsoft/deberta-xlarge',
  'deberta-v2': 'microsoft/deberta-v2-xlarge', 
  'deberta-v2-xlarge': 'microsoft/deberta-v2-xlarge', 
  'deberta-v2-xxlarge': 'microsoft/deberta-v2-xxlarge',
  'deberta-v3': 'microsoft/deberta-v3-large', 
  'deberta-v3-base': 'microsoft/deberta-v3-base',
  'deberta-v3-nli': 'cross-encoder/nli-deberta-v3-large',
  'deberta-xlarge-v2': 'microsoft/deberta-xlarge-v2',
  'deberta-v2-xlarge-mnli': 'microsoft/deberta-v2-xlarge-mnli',
  'deberta-v2-xlarge-cuad': 'akdeniz27/deberta-v2-xlarge-cuad',
  'xlm-roberta-large': 'xlm-roberta-large',
  'xlm': 'xlm-roberta-large',
  'unilm': 'microsoft/unilm-large-cased',
  'rembert': 'google/rembert',
  'funnel': 'funnel-transformer/large',
  'funnel-xlarge': 'funnel-transformer/xlarge',
  'tapas': 'google/tapas-large-finetuned-sqa',
}

unigrams = {'Position': 15419,
         'Evidence': 45702,
         'Claim': 50202,
         'Counterclaim': 5817,
         'Rebuttal': 4337,
         'Concluding Statement': 13505,
         'Lead': 9305,
         'Nothing': 30873}
bigrams = {'Evidence|Evidence': 4270,
         'Evidence|Claim': 17692,
         'Claim|Counterclaim': 1021,
         'Counterclaim|Rebuttal': 3588,
         'Rebuttal|Evidence': 1801,
         'Evidence|Concluding Statement': 8704,
         'Position|Nothing': 3466,
         'Nothing|Claim': 12897,
         'Claim|Evidence': 27985,
         'Evidence|Nothing': 9150,
         'Nothing|Counterclaim': 531,
         'Claim|Nothing': 9266,
         'Nothing|Evidence': 6838,
         'Concluding Statement|Nothing': 2159,
         'Claim|Claim': 10290,
         'Evidence|Position': 714,
         'Concluding Statement|Position': 228,
         'Position|Claim': 5343,
         'Position|Evidence': 1815,
         'Evidence|Counterclaim': 3363,
         'Rebuttal|Counterclaim': 228,
         'Rebuttal|Concluding Statement': 927,
         'Nothing|Position': 1246,
         'Lead|Claim': 164,
         'Counterclaim|Evidence': 1299,
         'Lead|Position': 1289,
         'Rebuttal|Claim': 702,
         'Nothing|Concluding Statement': 2298,
         'Rebuttal|Nothing': 522,
         'Claim|Concluding Statement': 915,
         'Counterclaim|Concluding Statement': 149,
         'Position|Concluding Statement': 497,
         'Evidence|Rebuttal': 478,
         'Claim|Position': 413,
         'Counterclaim|Nothing': 415,
         'Position|Counterclaim': 269,
         'Concluding Statement|Counterclaim': 102,
         'Counterclaim|Claim': 201,
         'Nothing|Rebuttal': 247,
         'Lead|Nothing': 171,
         'Lead|Evidence': 47,
         'Concluding Statement|Concluding Statement': 3,
         'Concluding Statement|Claim': 73,
         'Counterclaim|Position': 67,
         'Counterclaim|Counterclaim': 66,
         'Claim|Rebuttal': 2,
         'Lead|Counterclaim': 30,
         'Concluding Statement|Evidence': 31,
         'Rebuttal|Position': 60,
         'Rebuttal|Rebuttal': 11,
         'Evidence|Lead': 3,
         'Position|Lead': 1,
         'Position|Rebuttal': 3,
         'Lead|Lead': 1,
         'Lead|Rebuttal': 1,
         'Position|Position': 1}

def is_long_model(model_name):
  return 'longformer' in model_name or 'bird' in model_name

def get_model_type(model_name):
  if any(name in model_name for name in ['longformer-large', 'gpt2', 'xlarge']) or FLAGS.multi_inputs:
    return 'large'
  elif any(name in model_name for name in ['large', 'longformer-base']):
    return 'base'
  else:
    return 'tiny'
  
def get_model_name(hug):
  if hug in ['base', 'large']:
    return 'longformer'
  return hug

def get_tokenizer(backbone):
  if 'unilm' in backbone:
    backbone = 'bert-large-cased'
  # tokenizer = AutoTokenizer.from_pretrained(backbone, add_prefix_space=FLAGS.add_prefix_space)
  if (not 'deberta-v' in backbone) or FLAGS.ori_deberta_v2_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(backbone, add_prefix_space=FLAGS.add_prefix_space)
  else:
    from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
    tokenizer = DebertaV2TokenizerFast.from_pretrained(backbone, add_prefix_space=FLAGS.add_prefix_space)
  
  # 特别注意添加特殊符号要和tfrecord 以及infer 对齐保持一致 !!! 
  special_tokens_dict = {'additional_special_tokens':[]}
  # if FLAGS.add_br:
  special_tokens_dict['additional_special_tokens'].append(FLAGS.br)
  #不管在线和训练是否使用num_words_emb 保持占位编码一致
  # if FLAGS.num_words_emb:
  for i in range(50):
    special_tokens_dict['additional_special_tokens'].append(f'[NWORDS{i}]')
    
  # #离线tfrecord生成需要--stride和在线 --stride保持一致
  # # if FLAGS.stride is not None:
  for i in range(10):
    special_tokens_dict['additional_special_tokens'].append(f'[START{i}]')
    special_tokens_dict['additional_special_tokens'].append(f'[END{i}]')
  
  # 离线在线都是看backbone 所以一致编码
  if 'gpt' in backbone:
    special_tokens_dict.update({'pad_token': '[PAD]'})
    
  if FLAGS.use_cluster:
    for i in range(15):
      special_tokens_dict['additional_special_tokens'].append(f'[CLUSTER{i}]')
  
  if FLAGS.encode_cls:
    for cls_ in ALL_CLASSES:
      special_tokens_dict['additional_special_tokens'].append(f'[{cls_}]')
  
  tokenizer.add_special_tokens(special_tokens_dict)
  
  ic(backbone, tokenizer, tokenizer.is_fast)
  # ic(tokenizer.cls_token, tokenizer.cls_token_id, tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.eos_token, tokenizer.eos_token_id)
  return tokenizer
    
def get_backbone(backbone, hug):
  backbone = backbone or backbones[hug]
  if FLAGS.continue_pretrain:
    backbone_path = f'{FLAGS.idir}/pretrain/{backbone}'
    if os.path.exists(backbone_path):
      backbone = backbone_path
  elif os.path.exists(f'../input/{backbone}'):
    backbone = f'../input/{backbone}'
  return backbone
    
# 可能需要gen-records.py share的部分
def init_():  
  FLAGS.folds = 5
  FLAGS.fold = FLAGS.fold or 0
  
  if FLAGS.bird_ori:
    FLAGS.block_size = 64
    FLAGS.n_blocks = 3
    
  if FLAGS.tiny:
    FLAGS.hug = 'tiny'
  
  # depreciated for FLAGS.best or update it ..
  if FLAGS.best:
    FLAGS.hug = 'large'
    FLAGS.max_len = FLAGS.max_len or 1280
  
  # TODO 支持多种不同tokenizer的不同backbone并存 输入也需要考虑多种tokenizer
  FLAGS.model_names = FLAGS.model_names or FLAGS.pretrains
  if FLAGS.model_names and (not FLAGS.hugs):
    FLAGS.hugs = [FLAGS.hug] * len(FLAGS.model_names)
  
  if FLAGS.max_eval:
    FLAGS.eval_len = False
  
  if not FLAGS.hugs:
    FLAGS.backbone = get_backbone(FLAGS.backbone, FLAGS.hug)
  else:
    for hug in FLAGS.hugs:
      FLAGS.backbones.append(get_backbone(None, hug))
    FLAGS.backbone = FLAGS.backbones[0]
  ic(FLAGS.backbone, FLAGS.backbones)
    
  if 'longformer-large' in FLAGS.backbone:    
    FLAGS.tf_dataset = True #TODO HACK
    assert not FLAGS.records_names
    assert not FLAGS.multi_inputs
  
  if any(name in FLAGS.backbone for name in ['roberta', 'electra', 'albert']):
    FLAGS.num_words_emb = False
    
  if any(name in FLAGS.backbone for name in ['electra', 'deberta-v3', 'deberta-v2']):
    # assert not FLAGS.split_punct, 'Tokenizer has already split punct by default'
    FLAGS.split_punct = False
    
  if FLAGS.records_names:
    FLAGS.torch_only = True
   
  FLAGS.num_classes = NUM_CLASSES
  if FLAGS.method == 1:
    FLAGS.num_classes = NUM_CLASSES * 2 
  else:
    FLAGS.start_loss = True
    if FLAGS.method == 3 or FLAGS.method == 4:
      if FLAGS.end_loss_rate == 0:
        FLAGS.end_loss_rate = 10.
      FLAGS.pred_method = 'se'
      # FLAGS.label_inside = True
      
    if FLAGS.label_inside:
      FLAGS.post_reduce_method = 'avg'
      FLAGS.post_reduce_method2 = 'avg'
      FLAGS.mask_inside = False
      
    if FLAGS.pred_method == 'end':
      if FLAGS.end_loss_rate == 0:
        FLAGS.end_loss_rate = 10
        FLAGS.start_loss_rate = 0
      
    if FLAGS.start_loss_rate == 0:
      if FLAGS.end_loss_rate == 0:
        FLAGS.end_loss_rate = 10
      FLAGS.pred_method = 'end'
      
    if FLAGS.pred_method == 'se':
      assert FLAGS.start_loss_rate > 0 
      assert FLAGS.end_loss_rate > 0 
      
  config = AutoConfig.from_pretrained(FLAGS.backbone)
  max_len = FLAGS.max_len or 1536
  if not FLAGS.max_len:
    if not is_long_model(FLAGS.backbone):
      max_len = 512
  FLAGS.max_len = max_len
  FLAGS.max_len_valid = FLAGS.max_len_valid or FLAGS.max_len
  # if config.max_position_embeddings > 0:
  #   assert FLAGS.max_len <= config.max_position_embeddings, f'{FLAGS.max_len} {config.max_position_embeddings}'
  
  # if FLAGS.max_len > 512:
  #   FLAGS.seq_encoder = True
  
  if FLAGS.max_len > 768:
    FLAGS.last_tokens = 768
    FLAGS.last_tokens2 = 512
  elif FLAGS.max_len > 512:
    FLAGS.last_tokens = 512
    FLAGS.last_tokens2 = 256
    
  ic(FLAGS.method, FLAGS.num_classes, FLAGS.max_len, FLAGS.pred_method, FLAGS.token2word,
     FLAGS.token_loss_rate, FLAGS.start_loss_rate, FLAGS.end_loss_rate, FLAGS.parts_loss_rate, 
     FLAGS.split_method, FLAGS.post_reduce_method, FLAGS.post_reduce_method2, 
     FLAGS.num_words_emb, FLAGS.mark_end, FLAGS.last_tokens, FLAGS.last_tokens2)
  
  if FLAGS.cls_parts:
    FLAGS.parts_loss = True
  
  if FLAGS.cls_para:
    FLAGS.para_loss = True
  
  ic(FLAGS.num_classes, FLAGS.start_loss, FLAGS.parts_loss, FLAGS.para_loss)
        
  if 'longformer' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.longformer'
  elif 'bird' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.bird'
  elif 'bert-large-cased' in FLAGS.backbone or 'unilm-large-cased' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.bert-cased'
  elif 'bert-large-uncased' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.bert-uncased'
  elif 'xlm-roberta' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.xlm-roberta'
  elif 'roberta' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.roberta'
    # if not FLAGS.tf:  # TODO torch 如何处理1024？不过tf看在前部表现也是512 比1024更好 
  elif 'xlnet' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.xlnet'
  elif 'gpt2' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.gpt2'
    FLAGS.add_special_tokens = False
  elif 'macbert' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.macbert'
  elif 'electra' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.electra'
  elif 'bart' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.bart'
  elif 'albert' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.albert'
  elif 'reformer' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.reformer'
  elif 'span' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.span'
  elif 'deberta-v' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.deberta-v3'
  elif 'deberta' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.deberta'
  elif 'roformer' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.roformer'
  elif 'funnel' in FLAGS.backbone:
    FLAGS.records_name = 'tfrecords.funnel'
  elif FLAGS.hug:
    FLAGS.records_name = f'tfrecords.{FLAGS.hug}'
    
  tokenizer = get_tokenizer(FLAGS.backbone)
  gezi.set('tokenizer', tokenizer)
  ic(tokenizer, FLAGS.br)
  
  assert not (FLAGS.mask_inside and FLAGS.label_inside)
  ic(FLAGS.mask_inside, FLAGS.label_inside)
  if FLAGS.normalize:
    FLAGS.records_name += '.norm'
  if FLAGS.tolower:
    FLAGS.records_name += '.uncased'
  else:
    FLAGS.records_name += '.cased'
  # if FLAGS.remove_punct:
  #   FLAGS.records_name += '.rpunct'
  # if FLAGS.fix_misspell:
  #   FLAGS.records_name += '.fmiss'
  if FLAGS.split_punct:
    FLAGS.records_name += '.sp'
  if FLAGS.custom_tokenize:
    FLAGS.records_name += '.ct'
  
  # if FLAGS.mask_inside:
  #   FLAGS.records_name += '.mi'
  # if FLAGS.label_inside:
  #   FLAGS.records_name += '.li'
  # if FLAGS.ori_br:
  #   FLAGS.records_name += '.obr'
  if FLAGS.mask_more:
    FLAGS.records_name += '.mm'
  if FLAGS.merge_br:
    FLAGS.records_name += f'.{FLAGS.merge_br}'
  if FLAGS.up_sample:
    FLAGS.records_name += '.upsample'
  if FLAGS.remove_br:
    FLAGS.records_name += '.nobr'
  if FLAGS.odvt:
    FLAGS.records_name += '.odvt'
  if FLAGS.stride is not None:
    FLAGS.records_name += f'.stride{FLAGS.stride}'
  if FLAGS.records_type != 'token':
    FLAGS.records_name += f'.{FLAGS.records_type}'
  if FLAGS.filter_records:
    FLAGS.records_name += f'.filter'
  if FLAGS.corrected:
    FLAGS.records_name += f'.corr'
  FLAGS.records_name += f'.{FLAGS.split_method}'
  FLAGS.records_name += f'.len{FLAGS.max_len}'
  if FLAGS.records_version:
    FLAGS.records_name += f'.rv_{FLAGS.records_version}'
  if FLAGS.aug != 'en':
    gezi.set('records_name', FLAGS.records_name)
    FLAGS.records_name = FLAGS.records_name.replace(f'.{FLAGS.split_method}', f'.{FLAGS.aug}.{FLAGS.split_method}').replace('corr', '')
    
  if FLAGS.aug_nopunct:
    FLAGS.augs = ['nopunct']
    FLAGS.dsi = FLAGS.dsi or [0,0,0,1]

  if FLAGS.aug_swap:
    FLAGS.augs = ['swap']
    FLAGS.dsi = FLAGS.dsi or [0,0,0,1]

  if FLAGS.aug_lang:
    FLAGS.augs = ['de']
    FLAGS.dsi = FLAGS.dsi or [0,0,0,1]
    
  ep = 3
  # if 'longformer' in FLAGS.backbone:
  #   ep = 2
  FLAGS.ep = FLAGS.ep or ep
  if FLAGS.ep <= 1:
    FLAGS.rl_start_epoch = 0
  
  if FLAGS.augs:
    if len(FLAGS.augs) == 1:
      if FLAGS.augs[0].endswith('_ep'):
        FLAGS.augs = [f'{FLAGS.augs[0]}{i}' for i in range(FLAGS.max_augs)]
    
    ## TODO ...
    FLAGS.records_names = [FLAGS.records_name.replace(f'.{FLAGS.split_method}', f'.{aug}.{FLAGS.split_method}').replace('.corr', '') for aug in FLAGS.augs]
  
  if FLAGS.aug_split:
    augs = ['start', 'end', 'mid', 'se']
    FLAGS.augs = [aug for aug in augs if aug != FLAGS.split_method]
    FLAGS.dsi = FLAGS.dsi or [1, 1, 1, 1]
    FLAGS.records_names = [FLAGS.records_name.replace(f'.{FLAGS.split_method}', f'.{aug}') for aug in FLAGS.augs]
    
  if FLAGS.multi_inputs:
    ## now by default seq_encoder is true, to make 2048+(like 4096) infer, you could set --seq_encoder=0 and use a stride model to cover all parts, or end model to cover last words
    # FLAGS.seq_encoder = True
    assert FLAGS.seq_encoder, 'multi inputs must with --seq_encoder to merge results from tokens to word level'
    # augs = ['start', 'end', 'mid', 'se']
    if not FLAGS.multi_inputs_srcs:
      if FLAGS.max_len <= 512:
        FLAGS.multi_inputs_srcs = ['end', 'mid']
      else:
        FLAGS.multi_inputs_srcs = ['end']

    augs = FLAGS.multi_inputs_srcs
    FLAGS.augs = [aug for aug in augs if aug != FLAGS.split_method]
    FLAGS.records_names = [FLAGS.records_name.replace(f'.{FLAGS.split_method}', f'.{aug}') for aug in FLAGS.augs]
        
  if FLAGS.seq_encoder:
    FLAGS.merge_tokens = True 
    
  if FLAGS.aug_lower:
    FLAGS.augs = ['uncased']
    FLAGS.dsi = FLAGS.dsi or [1, 1]
    if 'uncased' in FLAGS.records_name:
      FLAGS.records_names = [FLAGS.records_name.replace('uncased', 'cased')]
    else:
      FLAGS.records_names = [FLAGS.records_name.replace('cased', 'uncased')]

  ic(FLAGS.augs, FLAGS.records_name, FLAGS.records_names, FLAGS.aug_start_epoch, 
     FLAGS.lower, FLAGS.custom_tokenize,
     FLAGS.multi_inputs, FLAGS.seq_encoder, FLAGS.merge_tokens, FLAGS.split_punct)
  
def init(): 
  init_()
  if FLAGS.test:
    FLAGS.mode = 'valid'
    # FLAGS.fie = 0.1
    FLAGS.num_valid = 100
    FLAGS.fast = True
    
  if FLAGS.exp:
    FLAGS.loss_scale = 100
    FLAGS.use_wordids = True
    
  # FLAGS.gpus = 4
  # FLAGS.train_scratch = True
  ## torch默认TorchOnly 走自己的Dataset全部载入内存shuffle，tf版本设置最大buffer效果更好，目前小模型稍好于torch很微弱
  FLAGS.buffer_size = 20000
  FLAGS.static_input = True
  FLAGS.cache_valid = True
  FLAGS.async_eval = True
  FLAGS.async_eval_last = True if not FLAGS.pymp else False
  FLAGS.async_valid = False
  
  # if FLAGS.abhishek:
  #   FLAGS.opt = 'bert-adamw'
    # FLAGS.mdrop = True
  
  if not FLAGS.tf:
    # make torch by default
    FLAGS.torch = True
  else:
    FLAGS.torch = False
  
  if FLAGS.torch:
    if not FLAGS.tf_dataset:
      FLAGS.torch_only = True
       
  records_pattern = f'{FLAGS.idir}/{FLAGS.records_name}/train/*.tfrec'
  files = gezi.list_files(records_pattern) 
  ic(records_pattern)
  if FLAGS.online:
    FLAGS.allnew = True
    FLAGS.train_files = files
  else:
    FLAGS.train_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds != FLAGS.fold]
  records_name = gezi.get('records_name', FLAGS.records_name)
  valid_records_pattern = records_pattern.replace(FLAGS.records_name, records_name) \
                            .replace(f'.len{FLAGS.max_len}', f'.len{FLAGS.max_len_valid}')\
                              .replace('.upsample', '').replace('.filter', '')
  # if FLAGS.stride:
  #   valid_records_pattern = valid_records_pattern.replace(f'.stride{FLAGS.stride}', '')
  # valid_records_pattern = valid_records_pattern.replace(f'.rv_fix', '')
  
  files = gezi.list_files(valid_records_pattern) 
  FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]
  
  if not FLAGS.online:
    FLAGS.train_files = [x for x in FLAGS.train_files if x not in FLAGS.valid_files]
  if FLAGS.eval_left:
    FLAGS.valid_files = [x.replace('.len', '.left.len').replace('.aug', '').replace('.sample', '') for x in FLAGS.valid_files]
    
  # if FLAGS.mix_train:
  #   records_pattern = records_pattern.replace('.len', '.left.len')
  #   train_files = gezi.list_files(records_pattern)
  #   if not FLAGS.online:
  #     train_files = [x for x in train_files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds != FLAGS.fold]
  #   FLAGS.train_files += train_files
  #   np.random.shuffle(FLAGS.train_files)
  
  if FLAGS.mix_train:
    if FLAGS.online:
      train_files = files
    else:
      train_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds != FLAGS.fold]
    np.random.shuffle(train_files)
    ## method 1 make ori train files at last electra 6239
    FLAGS.train_files += train_files  
    
  assert FLAGS.train_files, records_pattern
  assert FLAGS.valid_files, valid_records_pattern
  
  FLAGS.run_version = FLAGS.run_version or RUN_VERSION
  
  # remove from v29
  model_type = get_model_type(FLAGS.backbone)
  
  if model_type != 'large' and FLAGS.max_len <= 512 and (FLAGS.rdrop_rate == 0) and (not FLAGS.multi_inputs):
    FLAGS.gpus = FLAGS.gpus or 2
  ic(FLAGS.backbone, model_type)
    
  if FLAGS.backbone == 'roberta-base':
    FLAGS.clear_first = True
    assert not FLAGS.online
    
  if FLAGS.online:
    FLAGS.allow_train_valid = True
    FLAGS.nvs = 1
    # FLAGS.train_allnew = True
    assert FLAGS.fold == 0
      
  FLAGS.log_all_folds = True
  if FLAGS.log_all_folds or FLAGS.fold == 0:
    FLAGS.wandb = True
    if FLAGS.folds_metrics:
      FLAGS.wandb_resume = True
    FLAGS.wandb_project = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # FLAGS.wandb_project += '.v2'
  FLAGS.write_summary = True
  
  FLAGS.run_version += f'/{FLAGS.fold}'
    
  pres = ['offline', 'online']
  pre = pres[int(FLAGS.online)]
  model_dir = f'../working/{pre}/{FLAGS.run_version}/model'  
  FLAGS.model_dir = FLAGS.model_dir or model_dir
  if FLAGS.mn == 'model':
    FLAGS.mn = ''
  if not FLAGS.mn:  
    if FLAGS.hug:
      model_type = get_model_type(FLAGS.backbone)
      model_name = get_model_name(FLAGS.hug)
    FLAGS.mn = f'{model_name}'
    if FLAGS.tf:
      FLAGS.mn = f'tf.{FLAGS.mn}'
    FLAGS.mn += f'.{FLAGS.split_method}'
    # if FLAGS.merge_tokens:
    #   FLAGS.mn += f'.word-{FLAGS.word_combiner}'
    # else:
    #   FLAGS.mn += '.token'
    if not FLAGS.mask_inside:
      FLAGS.mn += '.nomi'
    if FLAGS.max_len != 512:
      FLAGS.mn += f'.len{FLAGS.max_len}'
    if FLAGS.mui:
      FLAGS.mn += '.mui-' + '-'.join(FLAGS.multi_inputs_srcs)
    mt.model_name_from_args(ignores=['tf', 'hug', 'sm', 
                                     'save_final', 'sf', 
                                     'tiny', 'tf_dataset',
                                     'split_method', 'sm',
                                    #  'merge_tokens', 
                                     'mask_inside', 'mi',
                                     'word_combiner', 'max_len',
                                     'multi_inuts', 'mui', 'multi_inputs_srcs',
                                     'show_keys', 'mis'])
    FLAGS.mn += PREFIX
  
  # bs 16 验证效果最好 8 cv提升 但是线上下降
  bs = 16
  FLAGS.bs = FLAGS.bs or bs
  eval_bs = 32 
  FLAGS.eval_batch_size = eval_bs
  
  if FLAGS.bs == 1:
    FLAGS.gpus = 1
  FLAGS.write_valid_final = True
  
  if not FLAGS.online:
    FLAGS.nvs = FLAGS.nvs or 3
  
  # if (not FLAGS.multi_inputs) and 'deberta-v3' in FLAGS.backbone:
  #   FLAGS.half_lr = False
  
  if not FLAGS.model_names:
    if FLAGS.multi_inputs:
      if all(x in FLAGS.backbone for x in ['deberta', 'large']) \
        and (len(FLAGS.multi_inputs_srcs) > 1 or FLAGS.max_len > 512):
        # # 上一版本710 唯一一个bs 8的模型 集成708->710 尝试对比 acc step=2
        # 小模型测试 bs=8 线上效果降低 但是离线看graident acc效果下降较多 暂时只能减小bs
        # if len(gezi.get_gpus()) < 8:
        FLAGS.bs = 8
        FLAGS.half_lr = True
        # if not FLAGS.half_lr:
        #   FLAGS.acc_steps = 2
        # else:
        #   FLAGS.bs = 8
    
    if FLAGS.seq_encoder:
      if 'longformer-large' in FLAGS.backbone or 'bigbird' in FLAGS.backbone:
        FLAGS.bs = 8
        FLAGS.half_lr = True
    
    if FLAGS.max_len > 1024 and 'deberta' in FLAGS.backbone:
      FLAGS.bs = 8
      FLAGS.half_lr = True
  
  if 'deberta-xlarge' in FLAGS.backbone:
    FLAGS.bs = 8
    FLAGS.half_lr = True
  
  #注意这些学习率都是之前按照bs 16设置，当前改为bs 8 对应设置了half_lr=True 实际 /2
  if not 'large' in FLAGS.backbone:
    lr = 1e-4
  else:
    lr = 2.5e-5
 
  # TODO electra仍然有较小概率崩溃1/20大约，检查设置成更小的1.0效果 测试依然electra可能崩溃 维持2 注意只要loss ok 线上electra没有问题
  # 1/31/2022 batch size 为8 之后 似乎没有出现崩溃
  if 'electra' in FLAGS.backbone:
    lr = 2e-5    
    # # HACK 默认配置fold4 容易中间崩溃 保持之前配置不变
    # if FLAGS.multi_inputs:
    #   FLAGS.half_lr = False
    #   FLAGS.bs = 16
    
  if 'deberta' in FLAGS.backbone:
    lr = 3e-5
    
  # 上一版本deberta v3 多输入采用的 half lr 之后是1e-5的设置
  if all(x in FLAGS.backbone for x in ['deberta-v3', 'large']):
    lr = 2e-5
    # if not FLAGS.multi_inputs:
    #   lr = 3e-5
  
  # TODO 是否lr太小 再对比下lr=2e-5
  if all(x in FLAGS.backbone for x in ['deberta', 'xlarge']):
    # FLAGS.half_lr = True
    lr = 2e-5
    if 'deberta-v2' in FLAGS.backbone:
      lr = 1e-5
    # lr = 5e-6 # 之前虽然有很大概率不收敛 但是最终还是收敛了 最近复现基本都不收敛了 继续调小到5e-6 另外Adam的correct_bias设置为True
    # FLAGS.backbone_lr = lr
    # FLAGS.base_lr = FLAGS.backbone_lr * 100
    # if 'deberta-xlarge' in FLAGS.backbone:
    #   FLAGS.acc_steps = 2
 
  FLAGS.lr = FLAGS.lr or lr
  # 影响非常小
  if FLAGS.lr_decay:
    FLAGS.lr_decay_power = 0.5
    
  # clip need to verify...
  # if 'deberta-v2' not in FLAGS.backbone:
  FLAGS.clip_gradients = 1.
  
  if FLAGS.seq_encoder:
    FLAGS.base_lr = 1e-3 
    FLAGS.backbone_lr = FLAGS.lr
     
  if FLAGS.half_lr:
    if FLAGS.backbone_lr is not None:
      FLAGS.base_lr /= 2.
      FLAGS.backbone_lr /= 2.
    FLAGS.lr /= 2.
    
  if FLAGS.lr_scale is not None:
    if FLAGS.backbone_lr is not None:
      FLAGS.base_lr *= FLAGS.lr_scale
      FLAGS.backbone_lr *= FLAGS.lr_scale
  # FLAGS.learning_rates = [FLAGS.lr, FLAGS.lr * 0.5]
  # if not FLAGS.online:
  #   FLAGS.seed = 1024
  
  # change from adamw back to adam
  optimizer = 'bert-adamw' 
  FLAGS.optimizer = FLAGS.optimizer or optimizer
  
  FLAGS.save_model = False
  if FLAGS.online:
    FLAGS.sie = FLAGS.ep
    FLAGS.save_final = True
  else:
    FLAGS.sie = 1e10 # not save model for offline mode FIXME tf仍然会生成。。
    if FLAGS.fold == 0:
      FLAGS.save_final = True
  
  # #HACK 之前没问题 怀疑是新fold数据 莫名引起 10%的时候验证没问题 如果按照33% valid数据predict的时候会nccl报错退出 
  # # 这个问题只有longformer-large出现原因未知 FIXME
  # if 'longformer-large' in FLAGS.backbone:
  #   if FLAGS.fold == 0:
  #     FLAGS.save_model = True
  #   assert FLAGS.tf_dataset, 'HACK fail if FLAGS.torch_only model at last step loss.backward() NCCL fail epoch 1'
  #   ic(FLAGS.backbone, FLAGS.save_model, FLAGS.nvs)
   
  # if 'electra' in FLAGS.backbone: 
  #   class_weights = np.asarray([1.] * NUM_CLASSES)
  #   class_weights[2] = 0.2
  #   class_weights = class_weights / class_weights.sum() * NUM_CLASSES
  #   gezi.set('class_weights', class_weights)
  #   ic(class_weights)
    
  
