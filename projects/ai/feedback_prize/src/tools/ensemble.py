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
import gezi
from gezi import tqdm
from src.eval import *
from src.util import *
from src import config
from src.config import *
from src.visualize import *
from src.ensemble_conf import *
from src.decode import decodes
from gezi.plot import display

pd.set_option('display.float_format', lambda x: '%.02f' % x)
gezi.set_pandas_widder()

# ------------------ config
show_keys = ['folds', 'fold', 'f1/Overall', 'F1/Overall', *[f'f1/{cls}' for cls in CLASSES], 'f1_400-/Overall', 'f1_400-800/Overall', 'f1_800+/Overall']
             
def main(_): 
  # P.update({'pre_prob': 0.9994445358985654, 'sep_prob': 0.677696887096806, 'sep_prob2': 0.34290279696674325})
  
  if len(sys.argv) > 1 and sys.argv[1].startswith('on'):
    mark = 'online'
  else:
    mark = 'offline'
  
  if mark == 'online':
    FLAGS.online = True
    FLAGS.max_eval = FLAGS.max_eval or 1000
  # if FLAGS.max_eval:
  #   FLAGS.eval_len = False
  
  folds = 5 if mark != 'online' else 1
  scores = []
  x_ = None
  root = f'../working/{mark}/{v}'
  df_gts, df_preds, df_preds2, xs_ = [], [], [], []
  
  has_missings = False
  has_bads = False
  for fold in range(folds):
    model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
    model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
    model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    if fold == 0:
      ic(model_dirs)
      ## do not restore, no need if restore here --ensemble... has no effect then
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
  ic(has_missings, has_bads)
  # if has_missings or has_bads:
  #   exit(0)
  # if has_bads:
  #   exit(0)
  FLAGS.wandb_project = FLAGS.wandb_project or 'feedback_prize'
  ic(FLAGS.wandb_project)
  mns_name = str(len(mns)) + '_' + '|'.join(mns)
  wandb_dir = f'{os.path.dirname(os.path.dirname(model_dir))}/ensemble'
  config = FLAGS.flag_values_dict().copy()
  config['models'] = mns
  gezi.try_mkdir(wandb_dir)
  try:
    run = wandb.init(project=FLAGS.wandb_project,
                    group=f'{v}/ensemble' if len(mns) > 1 else f'{v}/single',
                    dir=wandb_dir,
                    name=mns_name,
                    id=None,
                    config=config,
                    resume=False)
  except Exception:
    pass
  votes_folds = []
  for fold in range(folds):
    model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
    model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
    model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    
    if len(mns) == 1:
      d = pd.read_csv(f'{model_dirs[0]}/metrics.csv')
      for mn in mns:
        gezi.pprint_dict({'fold': fold, 'f1/Overall': d['f1/Overall'].values[-1], 'mn': mn})
        # assert(d['f1/Overall'].values[-1] > 0.5), mn
      #   scores.append({'Fold': fold, 'f1/Overall':  d['f1/Overall'].values[-1]})
      # continue
    ic(len(model_dirs))
    # ic(model_dirs, mns)
    xs = [gezi.load(f'{model_dir}/valid.pkl') for model_dir in model_dirs]

    model_dir = model_dirs[0]
    df_gt = pd.read_feather(f'{model_dir}/valid_gt.fea')
    if 'num_words' not in df_gt.columns:
      df_gt['num_words'] = df_gt.text.apply(lambda x: len(x.split()))
    df_gts.append(df_gt)
    ensembler = Ensembler(need_sort=True)
    l = []
    votes = []
    for i, x in tqdm(enumerate(xs), total=len(xs), desc='convert', leave=False, ascii=True):
    # for i, x in enumerate(xs):
      d = pd.read_csv(f'{model_dirs[i]}/metrics.csv')
      m = {'mn': mns[i], 'fold': fold}
      for col in d.columns:
        m[col] = d[col].values[-1]
      l.append(m)
      if (d['f1/Overall'].values[-1] < 0.5):
        ic(model_dirs[i], d['f1/Overall'].values[-1])
      if 'probs' not in x:
        convert_res(x)      
      # ic(get_metrics(df_gt, x))
      # max_len = None if '.len' in mns[i] else 400
      # # max_len = None
      # # ic(mns[i], max_len)
      # ensembler.add(x, weights[i], max_len)
      ensembler.add(x, weights=[weights[i], weights2[i], weights3[i]])
      # ensembler.add(x, weights[i])
      if FLAGS.votes:
        votes.append(decodes(x, folds=50))
      
    gezi.pprint_df(pd.DataFrame(l), keys=['mn'] + show_keys)
    x = ensembler.finalize()
    # ic(get_metrics(df_gt, x))
    if FLAGS.votes:
      votes.insert(0, decodes(x, folds=50))
    xs_.append(x.copy())
    res = {'fold': fold, 'f1/Overall': 0}
    res = get_metrics(df_gt, x, res, folds=50)
    ic(res['f1/Overall'])
    # ic(res)
    gezi.log_wandb(res)
    df_preds.append(gezi.get('df_pred'))
    
    # if FLAGS.votes:
    #   df_pred_ = get_preds(x, votes=votes)
    #   res = get_metrics(df_gt, df_pred_, df_input=True)
    #   ic(res['f1/Overall'])
        
    # df_preds2.append(gezi.get('df_pred2', None))
    # gezi.pprint_dict(res, keys=list(res.keys())[:FLAGS.max_metrics_show])
    # x = ensemble_res(xs)
    # res = {'fold': fold, 'f1/Overall': 0}
    # res = get_metrics(df_gt, x, res)
    # gezi.pprint_dict(res, keys=show_keys)
    # ic(res['f1_1024+/Overall'])
    scores.append(res.copy())
    if len(scores) > 1:
      # if len(scores) == folds:
      df = pd.DataFrame(scores)
      answer = dict(df.mean())
      answer['fold'] = len(scores)
      gezi.pprint_df(df, keys=show_keys)
      gezi.pprint_dict(answer, keys=show_keys)
    else:
      res['fold'] = 1
      gezi.pprint_dict(res, keys=show_keys)

    votes_folds.append(votes)
    if FLAGS.first_fold_only:
      break
    
    if fold + 1 == FLAGS.eval_folds:
      break
    
  num_folds = len(scores)
  if num_folds > 1:  
    df_gt = pd.concat(df_gts)
    df_pred = pd.concat(df_preds)
    df_pred2 = None
    res = get_metrics(df_gt, df_pred, res, df_input=True, df_pred2=df_pred2)
    res['fold'] = num_folds
    gezi.log_wandb(res)
  
  if mark == 'offline' and num_folds == folds:
    res['mns'] = mns_name
    writer = gezi.MetricsWriter(f'../working/{mark}/{v}/ensemble.csv')
    writer.write(res)
    d = pd.read_csv(writer.metric_file)[['f1/Overall', 'mns']].drop_duplicates()
    print('by time:')
    gezi.pprint_df(d[['f1/Overall', 'mns']].tail(10))
    d = d.sort_values('f1/Overall', ascending=False)
    print('by score:')
    gezi.pprint_df(d[['f1/Overall', 'mns']].head(5))
    pre_best = d['f1/Overall'].values[0]
  
    ic(res['f1/Overall'] >= pre_best)
    gezi.pprint_df(df, keys=show_keys)
    gezi.pprint_dict(res, keys=show_keys)
    ic(num_folds, zip(mns, weights), len(mns), pre_best, res['f1/Overall'])
    if SAVE_PRED or (res['f1/Overall'] >= pre_best) or FLAGS.save_pred or FLAGS.save_pred_name:
      df_gt.to_csv(f'{root}/valid_gt.csv', index=False)
      df_pred = pd.concat(df_preds)
      df_pred.to_csv(f'{root}/valid_pred.csv', index=False)
      x = gezi.merge_array_dicts(xs_)
      pred_name = FLAGS.save_pred_name or 'valid'
      gezi.save(x, f'{root}/{pred_name}.pkl')
      
  if votes:
    votes = votes_folds[0]
    for i in range(1, len(votes_folds)):
      for j in range(len(mns) + 1):
        votes[j].update(votes_folds[i][j])
    ic('save votes', f'{root}/votes.pkl')
    gezi.save(votes, f'{root}/votes.pkl')
          
if __name__ == '__main__':
  app.run(main)  