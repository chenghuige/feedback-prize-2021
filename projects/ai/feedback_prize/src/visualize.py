#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   visualize.py
#        \author   chenghuige
#          \date   2021-12-27 04:35:35.262521
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import pandas as pd
import spacy
from gezi import tqdm

from src.eval import calc_f1

def prepare(df, df_texts=None, fast=False):
  df['start'] = df.predictionstring.apply(lambda x: int(str(x).split()[0]))
  df.sort_values(['id', 'start'], inplace=True)
  df = df.reset_index(drop=True)
  if 'class' in df.columns:
    df.rename({'class': 'discourse_type'}, axis=1, inplace=True)
  if fast:
    return df
  
  if 'text_' in df.columns:
    df.drop(['text_'], axis=1, inplace=True)
  if df_texts is not None:
    if 'text' in df.columns:
      df.drop(['text'], axis=1, inplace=True)
    df = pd.merge(df, df_texts, how='left', on='id')
  df["essay_len"] = df["text"].apply(lambda x: len(x))
  df["essay_words"] = df["text"].apply(lambda x: len(x.split()))
  dis_starts, dis_ends = [], []
  text_list = []
  for row in tqdm(df.itertuples(), total=len(df), desc='vis_prepare1', leave=False):
    preds = row.predictionstring.split()
    s, e = int(preds[0]), int(preds[-1])
    texts = row.text.split()
    start = sum([len(x) + 1 for x in texts[:s]])
    end = sum([len(x) + 1 for x in texts[:e + 1]])
    dis_starts.append(start)
    dis_ends.append(end)
    text = ' '.join(texts)
    text_list.append(text)

  df['discourse_start'], df['discourse_end'] = dis_starts, dis_ends
  df['text_'] = df['text']
  df['text'] = text_list

  df['gap_before'] = False
  df['gap_length'] = np.nan

  #set the first one
  df.loc[0, 'gap_before'] = True
  df.loc[0, 'gap_length'] = 0
  
  df['gap_end_length'] = df.essay_len - df.discourse_end

  #loop over rest
  for i in tqdm(range(1, len(df)), desc='vis_prepare2', leave=False):
    #gap if difference is not 1 within an essay
    if ((df.loc[i, "id"] == df.loc[i-1, "id"])\
        and (df.loc[i, "discourse_start"] - df.loc[i-1, "discourse_end"] > 0)):
      df.loc[i, 'gap_before'] = True
      df.loc[i,
             'gap_length'] = df.loc[i,
                                    "discourse_start"] - df.loc[i - 1,
                                                                "discourse_end"]
    #gap if the first discourse of an new essay does not start at 0
    elif ((df.loc[i, "id"] != df.loc[i-1, "id"])\
        and (df.loc[i, "discourse_start"] != 0)):
      df.loc[i, 'gap_before'] = True
      df.loc[i, 'gap_length'] = df.loc[i, "discourse_start"]

  return df

def add_gap_rows(df, essay):
  cols_to_keep = [
      'discourse_start', 'discourse_end', 'discourse_type', 'gap_length',
      'gap_end_length', 'text'
  ]
  df_essay = df.query('id == @essay')[cols_to_keep].reset_index(drop=True)

  #index new row
  insert_row = len(df_essay)

  #add gaps in between
  for i in range(1, len(df_essay)):
    if df_essay.loc[i, "gap_length"] > 0:
      start = df_essay.loc[i - 1, "discourse_end"]
      end = df_essay.loc[i, 'discourse_start']
      text = df_essay.loc[i, 'text']
      disc_type = "Nothing"
      gap_end = np.nan
      gap = np.nan
      df_essay.loc[insert_row] = [start, end, disc_type, gap, gap_end, text]
      insert_row += 1

  df_essay = df_essay.sort_values(by="discourse_start").reset_index(drop=True)

  #add gap at end
  if df_essay.loc[(len(df_essay) - 1), 'gap_end_length'] > 0:
    start = df_essay.loc[(len(df_essay) - 1), "discourse_end"]
    end = start + df_essay.loc[(len(df_essay) - 1), 'gap_end_length']
    disc_type = "Nothing"
    gap_end = np.nan
    gap = np.nan
    text = df_essay.loc[(len(df_essay) - 1), 'text']
    df_essay.loc[insert_row] = [start, end, disc_type, gap, gap_end, text]

  return df_essay



def visulize_df(df, essay):
  df_essay = add_gap_rows(df, essay)

  ents = []
  for i, row in df_essay.iterrows():
    data = row['text']
    start = int(row['discourse_start'])
    end = int(row['discourse_end'])
    ents.append({
        'start': start,
        'end': end,
        'label': row['discourse_type']
    })

  doc2 = {
      "text": data,
      "ents": ents,
  }

  colors = {
      'Lead': '#EE11D0',
      'Position': '#AB4DE1',
      'Claim': '#1EDE71',
      'Evidence': '#33FAFA',
      'Counterclaim': '#4253C1',
      'Concluding Statement': 'yellow',
      'Rebuttal': 'red'
  }
  options = {
      "ents": df_essay.discourse_type.unique().tolist(),
      "colors": colors
  }
  spacy.displacy.render(doc2,
                        style="ent",
                        options=options,
                        manual=True,
                        jupyter=True)


def visualize(essay):
  f1 = calc_f1(
      df_gt[df_gt['id'] == essay],
      df_pred[df_pred['id'] == essay].rename({'discourse_type': 'class'},
                                             axis=1))
  f1_binary = calc_f1(
      df_gt_[df_gt_['id'] == essay],
      df_pred_[df_pred_['id'] == essay].rename({'discourse_type': 'class'},
                                               axis=1))
  print(essay, 'f1:', f1, 'f1_binary:', f1_binary)
  visulize_df(df_gt, essay)
  visulize_df(df_pred, essay)
