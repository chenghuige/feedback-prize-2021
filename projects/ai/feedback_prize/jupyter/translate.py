#!/usr/bin/env python
# coding: utf-8
import numpy as np
from googletrans import Translator
import pandas as pd
import time
import gezi
from gezi import tqdm
import sys, os
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
from src import util

lang = sys.argv[1]
fold = int(sys.argv[2])
MAX_LEN = 10240 if len(sys.argv) <= 3 else int(sys.argv[3])
folds = 10

from nltk.tokenize import sent_tokenize
from googletrans import Translator
trans = Translator()
result = trans.translate('照片')
print(result)

d = pd.read_feather('../input/feedback-prize-2021/train_en.fea')

mid_texts = {}
target_texts = {}

id_list = []
text_list = []
mid_text_list = []
target_text_list = []

mid_texts_file = f'../input/feedback-prize-2021/mid_texts_{lang}_{fold}.pkl'
target_texts_file = f'../input/feedback-prize-2021/target_texts_{lang}_{fold}.pkl'
if os.path.exists(mid_texts_file):
  mid_texts = gezi.read_pickle(mid_texts_file)
  target_texts = gezi.read_pickle(target_texts_file)

ic(len(mid_texts), len(target_texts))
    
def translate(text):
  # MAX_LEN = 300 # set it smaller in second pass
  num_words = len(text.split())
  if num_words <= MAX_LEN:
    mid_text = trans.translate(text, dest=lang).text
    target_text = trans.translate(mid_text, dest='en').text
    time.sleep(0.5)
  else:
    mid_texts = []
    target_texts = []
    sents = sent_tokenize(text)    
    num_sents = len(sents)
    parts = int(num_words / MAX_LEN) + 1
    if num_sents > parts:
      sents = np.array_split(sents, 3)
      for i in range(len(sents)):
        sents[i] = ''.join(sents[i])
    for sent in sents:
      mid_text = trans.translate(sent, dest=lang).text
      target_text = trans.translate(mid_text, dest='en').text
      time.sleep(0.5)
      mid_texts.append(mid_text)
      target_texts.append(target_text)
    mid_text = ''.join(mid_texts)
    target_text = ''.join(target_texts)
  return mid_text, target_text

d_ = d.copy()
for i, row in tqdm(enumerate(d.itertuples()), total=len(d)):
  if i % folds != fold:
    continue
  row = row._asdict()
  text = row['para']
  target_text = text
  
  if text in target_texts:
    mid_text, target_text = mid_texts[text], target_texts[text]
  else:
    try:
      mid_text, target_text = translate(text)
    except Exception as e:
      ic(i, len(target_texts), len(text.split()), e)
      # time.sleep(2)
 
  if target_text != text:
    mid_texts[text] = mid_text
    target_texts[text] = target_text
    id_list.append(i)
    text_list.append(text)
    mid_text_list.append(mid_text)
    target_text_list.append(target_text)
    if len(target_text_list) % 100 == 0:
      gezi.save_pickle(target_texts, f'../input/feedback-prize-2021/target_texts_{lang}_{fold}.pkl')
      gezi.save_pickle(mid_texts, f'../input/feedback-prize-2021/mid_texts_{lang}_{fold}.pkl')
      pd.DataFrame({
        'pid': id_list, 
        'text': text_list,
        'mid': mid_text_list,
        'target': target_text_list
      }).to_csv(f'../input/feedback-prize-2021/trans_{lang}_{fold}.csv', index=False)
  else:
    ic(i, len(target_texts))

gezi.save_pickle(target_texts, f'../input/feedback-prize-2021/target_texts_{lang}_{fold}.pkl')
gezi.save_pickle(mid_texts, f'../input/feedback-prize-2021/mid_texts_{lang}_{fold}.pkl')
pd.DataFrame({
  'pid': id_list, 
  'text': text_list,
  'mid': mid_text_list,
  'target': target_text_list
}).to_csv(f'../input/feedback-prize-2021/trans_{lang}_{fold}.csv', index=False)


