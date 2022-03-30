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
  ic(v, root)
  if FLAGS.all_models:
    candidates = glob.glob(f'{root}/0/*/valid.pkl')
    cmns = [os.path.basename(os.path.dirname(x)) for x in candidates]
    mns = []
    for mn, ca in zip(cmns, candidates):
      is_ok = True
      for i in range(5):
        ca_ = ca.replace('/0/', f'/{i}/')
        model_dir = os.path.dirname(ca_)
        # ic(ca_, os.path.exists(ca_))
        if not os.path.exists(ca_):
          is_ok = False
          break
        if os.path.exists(f'{model_dir}/metrics.csv') and pd.read_csv(f'{model_dir}/metrics.csv')['f1/Overall'].values[-1] < 0.4:
          is_ok = False
          break
      if is_ok:
        mns.append(mn)
    ic(mns)
  else:
    from src.ensemble_conf import mns
  # exit(0)
  df_gts = []
  has_missings = False
  has_bads = False
  for fold in range(folds):
    if FLAGS.all_models:
      model_dirs = [f'../working/{mark}/{v}/{fold}/{mn}' for mn in mns]
    else:
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
    
  # wandb_kwargs = {
  #   'project': FLAGS.wandb_project,
  #   'group': f'{v}/ensemble-select',
  #   'dir': wandb_dir,
  #   'name': mns_name,
  #   'id': None,
  #   'config': config,
  #   'resume': False
  # }    
  # try:
  #   wandbc = WeightsAndBiasesCallback(metric_name="f1/Overall", wandb_kwargs=wandb_kwargs)
  # except Exception:
  #   pass
 
  global P
  global weights
  scores_dict = {}
  scores = []
  logs = {}
  for fold in range(folds):
    # ic('fold:', fold)
    if FLAGS.all_models:
      model_dirs = [f'../working/{mark}/{v}/{fold}/{mn}' for mn in mns]
    else:
      model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
      model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
      model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    
  #   ensembler = Ensembler(need_sort=True)
  #   for i, model_dir in enumerate(model_dirs):
  #     try:
  #       # ensembler.add(gezi.load(f'{model_dir}/valid.pkl'), weights=[weights[i], weights2[i]])
  #       ensembler.add(gezi.load(f'{model_dir}/valid.pkl'), weights[i])
  #     except Exception:
  #       ic(i,model_dir)
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
    if FLAGS.all_models:
      model_dirs = [f'../working/{mark}/{v}/{fold}/{mn}' for mn in mns]
    else:
      model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
      model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
      model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    
    for i, model_dir in tqdm(enumerate(model_dirs), total=len(model_dirs)):
      xs[i].append(gezi.load(f'{model_dir}/valid.pkl'))
  
  for i in range(len(xs)):
    xs[i] = gezi.merge_array_dicts(xs[i])

  # ensembler = Ensembler(need_sort=True)
  # for i in range(len(model_dirs)):
  #   # ensembler.add(xs[i].copy(), weights=[weights[i], weights2[i]])
  #   ensembler.add(xs[i].copy(), weights[i])
  # x = ensembler.finalize()
  # ic(len(x['id']))
  # res = get_metrics(df_gt, x, folds=50)
  # logs['metrics/best_val'] = res['f1/Overall']
  # logs.update(gezi.dict_prefix(P, 'params/'))
  # ic(scores, scores_dict)
  # ic(logs)
  # # gezi.log_wandb(logs)
  # pre_logs = logs.copy()
  
  # from gezi import geneticalgorithm as ga
  # def loss_func(ws):
  #   ensembler = Ensembler(need_sort=True)
  #   for i in range(len(model_dirs)):
  #     ensembler.add(xs[i].copy(), ws[i])
  #   x = ensembler.finalize()
    
  #   res = get_metrics(df_gt, x, folds=50)
  #   return 1. - res['f1/Overall']
  
  # varbound=np.array([[0,20]]*len(model_dirs))
  # model=ga(function=loss_func, dimension=len(model_dirs), variable_type='int', variable_boundaries=varbound)
  # model.run()
  # best_wt=model.output_dict['variable']
  # ic(best_wt)
  # ensembler = Ensembler(need_sort=True)
  # weights = best_wt
  # for i in range(len(model_dirs)):
  #   ensembler.add(xs[i].copy(), weights[i])
  # x = ensembler.finalize()
  # res = get_metrics(df_gt, x, folds=50)
  # ic(res)
  
  def objective(trial):
    for i in range(len(model_dirs)):
      if FLAGS.model_idx is not None:
        if i != FLAGS.model_idx:
          continue
      if FLAGS.model_name_ is not None:
        if mns[i] != FLAGS.model_name_:
          continue
      if not FLAGS.float_model_weight:
        weights[i] = trial.suggest_int(mns[i], FLAGS.min_model_weight, FLAGS.max_model_weight)
      else:
        weights[i] = trial.suggest_float(mns[i], 0, 1)

    ic(dict(zip(mns, weights)))
    
    ensembler = Ensembler(need_sort=True)
    for i in range(len(model_dirs)):
      if FLAGS.lens == None:
        ensembler.add(xs[i].copy(), weights[i])
      elif FLAGS.lens == 0:
        ensembler.add(xs[i].copy(), weights=[weights[i], 0, 0])
      elif FLAGS.lens == 1:
        ensembler.add(xs[i].copy(), weights=[0, weights[i], 0])
      elif FLAGS.lens == 2:
        ensembler.add(xs[i].copy(), weights=[0, 0, weights[i]])
    x = ensembler.finalize()
    
    res = get_metrics(df_gt, x, folds=50)
    return res['f1/Overall']

  study = optuna.create_study(direction='maximize')
  # study.optimize(objective, n_trials=FLAGS.n_trials, callbacks=[wandbc])
  study.optimize(objective, n_trials=FLAGS.n_trials)
  ic(study.best_value, study.best_params)
  # gezi.log_wandb(gezi.dict_prefix(study.best_params, 'study/'))
      
  # for key, val in study.best_params.items():
  #   P[key] = val

  # scores_dict_ = {}
  # scores = []
  # for fold in range(folds):
  #   ic('fold:', fold)
  #   if FLAGS.all_models:
  #     model_dirs = [f'../working/{mark}/{v}/{fold}/{mn}' for mn in mns]
  #   else:
  #     model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
  #     model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
  #     model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    
  #   ensembler = Ensembler(need_sort=True)
  #   for i, model_dir in enumerate(model_dirs):
  #     # ensembler.add(gezi.load(f'{model_dir}/valid.pkl'), weights=[weights[i], weights2[i]])
  #     ensembler.add(gezi.load(f'{model_dir}/valid.pkl'), weights[i])
  #   x = ensembler.finalize()
  #   res = get_metrics(df_gt, x, folds=50)
  #   ic(fold, scores_dict[fold], res['f1/Overall'])
  #   logs[f'metrics/{fold}'] = res['f1/Overall']
  #   if fold in ignored_folds:
  #     scores.append(res['f1/Overall'])
  #   scores_dict_[fold] = res['f1/Overall']
  # logs['metrics/best_val'] = study.best_value
  # logs['metrics/valid_mean'] = np.mean(scores)
  # logs['metrics/all_mean'] = np.mean(list(scores_dict_.values()))
  # logs.update(gezi.dict_prefix(P, 'params/'))
  # ic(np.mean(scores), np.mean(list(scores_dict_.values())))
  # ic(scores, scores_dict_)
  # ic(pre_logs, logs)
  # gezi.log_wandb(logs)

          
if __name__ == '__main__':
  app.run(main)  
  
