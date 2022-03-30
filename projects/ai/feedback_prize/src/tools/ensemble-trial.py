#!/usr/bin/env python
# coding: utf-8

from cmath import asin
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.express as px
import glob
import spacy
from collections import defaultdict
import traceback
import wandb
import sys, os
sys.path.append('../../../../utils')
sys.path.append('..')
os.environ["WANDB_SILENT"] = "true"
import optuna
from optuna import Trial
from optuna.integration.wandb import WeightsAndBiasesCallback
import gezi
from gezi import tqdm
from src.eval import *
from src.util import *
from src import config
from src.config import *
from src.visualize import *
from src.ensemble_conf import *
from src.get_preds import P

pd.set_option('display.float_format', lambda x: '%.02f' % x)

# ------------------ config
show_keys = ['folds', 'fold', 'f1/Overall', 'F1/Overall', *[f'f1/{cls}' for cls in CLASSES], 'f1_400-/Overall', 'f1_400-800/Overall', 'f1_800+/Overall']
             
def main(_): 
  # P.update({'pre_prob': 0.9994445358985654, 'sep_prob': 0.677696887096806, 'sep_prob2': 0.34290279696674325})
  
  if len(sys.argv) > 1 and sys.argv[1].startswith('on'):
    mark = 'online'
  else:
    mark = 'offline'
  
  FLAGS.eval_ori = False
  # if len(sys.argv) > 1 and 'all' in sys.argv[1]:
  #   FLAGS.eval_ori = True

  folds = 5 if mark != 'online' else 1
  scores = []
  x_ = None
  root = f'../working/{mark}/{v}'
  df_gts = []
  has_missings = False
  has_bads = False
  for fold in range(folds):
    model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
    model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
    model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    if fold == 0:
      ic(model_dirs)
      # gezi.restore_configs(model_dirs[0])
    missings = [
        model_dir for model_dir in model_dirs
        if not os.path.exists(f'{model_dir}/valid.pkl')
    ]
    missings2 = [
        model_dir for model_dir in model_dirs
        if not os.path.exists(f'{model_dir}/log.txt')
    ]
    if missings:
      ic(fold, missings)
      has_missings = True
      
    bads = [
      model_dir for model_dir in model_dirs
      if os.path.exists(f'{model_dir}/metrics.csv') and pd.read_csv(f'{model_dir}/metrics.csv')['f1/Overall'].values[-1] < 0.4
    ]
    if bads:
      ic(fold, bads)
      has_bads = True
    model_dir = model_dirs[0]
    df_gt = pd.read_feather(f'{model_dir}/valid_gt.fea')
    if 'num_words' not in df_gt.columns:
      df_gt['num_words'] = df_gt.text.apply(lambda x: len(x.split()))
    df_gts.append(df_gt)
  
  df_gt = pd.concat(df_gts)
  model_dir = model_dirs[0]
  ic(has_missings, has_bads)
  assert not has_missings and not has_bads
  FLAGS.wandb_project = FLAGS.wandb_project or 'feedback_prize'
  ic(FLAGS.wandb_project)
  mns_name = str(len(mns)) + '_' + '|'.join(mns)
  wandb_dir = f'{os.path.dirname(os.path.dirname(model_dir))}/ensemble'
  config = FLAGS.flag_values_dict().copy()
  config['models'] = mns
  gezi.try_mkdir(wandb_dir)

  ignored_folds = set([int(x) for x in FLAGS.ignored_folds]) if not FLAGS.online else []
  if FLAGS.max_models:
    model_dirs = model_dirs[:FLAGS.max_models]
    ic(FLAGS.max_models, model_dirs)
    
  wandb_kwargs = {
    'project': FLAGS.wandb_project,
    'group': f'{v}/ensemble-trial' if len(mns) > 1 else f'{v}/single-trial',
    'dir': wandb_dir,
    'name': mns_name,
    'id': None,
    'config': config,
    'resume': False
  }    
  try:
    wandbc = WeightsAndBiasesCallback(metric_name="f1/Overall", wandb_kwargs=wandb_kwargs)
  except Exception:
    pass 
 
  scores_dict = {}
  scores = []
  logs = {}
  # for fold in range(folds):
  #   ic('fold:', fold)
  #   model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
  #   model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
  #   model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    
  #   ensembler = Ensembler(need_sort=True)
  #   for i, model_dir in enumerate(model_dirs):
  #     ensembler.add(gezi.load(f'{model_dir}/valid.pkl'), weights=[weights[i], weights2[i], weights3[i]])
  #   x = ensembler.finalize()
  #   res = get_metrics(df_gt, x, folds=50)
  #   ic(fold, res['f1/Overall'])
  #   scores_dict[fold] = res['f1/Overall']
  #   logs[f'metrics/{fold}'] = res['f1/Overall']
  #   if fold in ignored_folds:
  #     scores.append(res['f1/Overall'])
  # ic(scores, scores_dict)
  # ic(np.mean(scores), np.mean(list(scores_dict.values())))
  # logs['metrics/valid_mean'] = np.mean(scores)
  # logs['metrics/all_mean'] = np.mean(list(scores_dict.values()))
      
  xs = [[] for _ in range(len(model_dirs))] 
  # 所以默认2,3,4搜索 0,1验证
  for fold in range(folds):
    if not FLAGS.online:
      if fold in ignored_folds:
        ic('ignore fold:', fold)
        continue
    model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
    model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
    model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    
    for i, model_dir in enumerate(model_dirs):
      xs[i].append(gezi.load(f'{model_dir}/valid.pkl'))
  
  for i in range(len(xs)):
    xs[i] = gezi.merge_array_dicts(xs[i])
         
  ensembler = Ensembler(need_sort=True)
  for i in range(len(model_dirs)):
    ensembler.add(xs[i].copy(), weights=[weights[i], weights2[i], weights3[i]])
  x = ensembler.finalize()
  ic(len(x['id']))
  res = get_metrics(df_gt, x, folds=50)
  logs['metrics/best_val'] = res['f1/Overall']
  if FLAGS.prob_idx is None:
    logs.update(gezi.dict_prefix(P, 'params/'))
  else:
    logs.update(gezi.dict_prefix(proba_thresh, 'params/'))
  ic(scores, scores_dict)
  ic(logs)
  gezi.log_wandb(logs)
  pre_logs = logs.copy()
  
  def objective(trial):
    # for i in range(len(model_dirs)):
    #   weights[i] = trial.suggest_int(f'w0_{i}_{mns[i]}', 0, 3)
    
    suggest = trial.suggest_float if not FLAGS.suggest_uniform else trial.suggest_uniform
    # ic(suggest)
    if FLAGS.prob_idx is None:
      # pass
      # P['pre_prob'] = suggest('pre_prob', 0., 1.)
      # P['sep_eq_prob'] = suggest('sep_eq_prob', 0., 1.)
      # P['sep_eq_prob_Lead'] = suggest('sep_eq_prob_Lead', 0., 1.)
      #  P['sep_eq_prob_Position'] = suggest('sep_eq_prob_Position', 0., 1.)
      # P['sep_eq_prob_Nothing'] = suggest('sep_eq_prob_Nothing', 0., 1.)
      # P['sep_eq_prob_Rebuttal'] = suggest('sep_eq_prob_Rebuttal', 0., 1.)
      P['sep_eq_prob_Counterclaim'] = suggest('sep_eq_prob_Counterclaim', 0., 1.)
      # P['pre_prob_Lead'] = suggest('pre_prob_Lead', 0., 1.)
      # P['pre_prob_Claim'] = suggest('pre_prob_Claim', 0., 1.)
      # P['pre_prob_Position'] = suggest('pre_prob_Position', 0., 1.)
      # P['pre_prob_Evidence'] = suggest('pre_prob_Evidence', 0., 1.)
      # P['pre_prob_Counterclaim'] = suggest('pre_prob_Counterclaim', 0., 1.)
      # P['pre_prob_Rebuttal'] = suggest('pre_prob_Rebuttal', 0., 1.)
      # P['pre_prob_Nothing'] = suggest('pre_prob_Nothing', 0., 1.)
      # P['sep_prob'] = suggest('sep_prob', 0., 1.)
      # P['sep_prob2'] = suggest('sep_prob2', 0., .5)
      # P['pre_prob2'] = suggest('pre_prob2', 0.8, 1.)
      # P['sep_prob'] = suggest('sep_prob', 0.5, 1.)
      # P['sep_prob3'] = suggest('sep_prob3', 0.5, 1.)
      # P['sep_prob4'] = suggest('sep_prob4', 0.5, 1.)
      # P['sep_prob_Lead'] = suggest('sep_prob_Lead', 0., 1.)
      # P['sep_prob_Rebuttal'] = suggest('sep_prob_Rebuttal', 0., .5)
      # P['sep_prob_Claim'] = suggest('sep_prob_Claim', 0., 1.)
      # P['sep_prob_Evidence'] = suggest('sep_prob_Evidence', 0., 1.)
      # P['sep_prob_Position'] = suggest('sep_prob_Position', 0., 1.)
      # P['sep_prob_Counterclaim'] = suggest('sep_prob_Counterclaim', 0., 1.)
      # P['sep_prob_Nothing'] = suggest('sep_prob_Nothing', 0., 1.)
      # for i in range(len(ALL_CLASSES)):
      #   P[i] = suggest(str(i), 0., 1.)
      # cls_ = 'Evidence'
      # P[cls_] = suggest(cls_, 0.7, 1.)
      # for i in range(len(ALL_CLASSES)):
      #   P2[i] = suggest(str(i), 0., 1.)
      # proba_thresh[cls_] = suggest(cls_ + '.prob', 0.5, 1.)
      # for cls_ in CLASSES:
      #   proba_thresh[cls_] = suggest(cls_, 0., 1.)
      # ic(proba_thresh)
    else:
      i= FLAGS.prob_idx
      proba_thresh[all_classes[i]] = suggest(all_classes[i], 0., 1.)
      # proba_thresh[classes[4]] = suggest(classes[4], 0.98, 1.)
          
    res = get_metrics(df_gt, x, folds=50)
    score = res['f1/Overall'] if FLAGS.prob_idx is None else res[f'f1/{all_classes[i]}']
    return score
  
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=FLAGS.n_trials, callbacks=[wandbc])
  ic(study.best_value, study.best_params)
  gezi.log_wandb(gezi.dict_prefix(study.best_params, 'study/'))
      
  for key, val in study.best_params.items():
    P[key] = val

  scores_dict_ = {}
  scores = []
  for fold in range(folds):
    ic('fold:', fold)
    model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
    model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
    model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    
    ensembler = Ensembler(need_sort=True)
    for i, model_dir in enumerate(model_dirs):
      ensembler.add(gezi.load(f'{model_dir}/valid.pkl'), weights=[weights[i], weights2[i], weights3[i]])
    x = ensembler.finalize()
    res = get_metrics(df_gt, x, folds=50)
    ic(fold, scores_dict[fold], res['f1/Overall'])
    logs[f'metrics/{fold}'] = res['f1/Overall']
    if fold in ignored_folds:
      scores.append(res['f1/Overall'])
    scores_dict_[fold] = res['f1/Overall']
  logs['metrics/best_val'] = study.best_value
  logs['metrics/valid_mean'] = np.mean(scores)
  logs['metrics/all_mean'] = np.mean(list(scores_dict_.values()))
  logs.update(gezi.dict_prefix(P, 'params/'))
  ic(np.mean(scores), np.mean(list(scores_dict_.values())))
  ic(scores, scores_dict_)
  ic(pre_logs, logs)
  gezi.log_wandb(logs)

          
if __name__ == '__main__':
  app.run(main)  
  