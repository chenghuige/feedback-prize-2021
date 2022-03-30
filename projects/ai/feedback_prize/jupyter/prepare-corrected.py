#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import glob
import spacy
from IPython.display import display_html, display
import sys, os
sys.path.append('../../../../utils')
import gezi
from gezi import tqdm
pd.set_option('display.float_format', lambda x: '%.0f' % x)


# In[2]:


#df = pd.read_csv('../input/feedback-prize-2021/train.csv')
df = pd.read_csv('../input/feedback-prize-2021/corrected_train.csv')
df['predictionstring'] = df['new_predictionstring']
folds = pd.read_csv('../input/feedback-prize-2021/folds.csv')


# In[3]:


df[df.id=='00066EA9880D']


# In[4]:


df[df.id=='BE01ACCDF251']


# In[5]:


train_names, train_texts = [], []
for file in tqdm(glob.glob('../input/feedback-prize-2021/train/*.txt')):
    train_names.append(os.path.basename(file).replace('.txt', ''))
    train_texts.append(open(file, 'r').read())
train_texts = pd.DataFrame({'id': train_names, 'text': train_texts})
train_texts.to_csv('../input/feedback-prize-2021/texts.csv')
train_texts.reset_index().to_feather('../input/feedback-prize-2021/texts.fea')


# In[6]:


train = pd.merge(df, train_texts, how = 'left', on = 'id')
train["essay_len"] = train["text"].apply(lambda x: len(x))
train["essay_words"] = train["text"].apply(lambda x: len(x.split()))


# In[7]:


dis_starts, dis_ends = [], []
starts, ends = [], []
text_list = []
for row in tqdm(train.itertuples(), total=len(train)):
  preds = row.predictionstring.split()
  s, e = int(preds[0]), int(preds[-1])
  texts = row.text.split()
  start = sum([len(x) + 1 for x in texts[:s]])
  end = sum([len(x) + 1 for x in texts[:e + 1]])
  dis_starts.append(start)
  dis_ends.append(end)
  starts.append(s)
  ends.append(e + 1)
  text = ' '.join(texts)
  text_list.append(text)
  
train['discourse_start'], train['discourse_end'] = dis_starts, dis_ends
train['start'], train['end'] = starts, ends
train['text_'] = train['text']
train['text'] = text_list


# In[8]:


train["discourse_words"] = train.end - train.start


# In[9]:


#initialize columns
train['gap_before'] = False
train['gap_length'] = np.nan

#set the first one
train.loc[0, 'gap_before'] = True
train.loc[0, 'gap_length'] = 8

#loop over rest
for i in tqdm(range(1, len(train))):
    #gap if difference is not 1 within an essay
    if ((train.loc[i, "id"] == train.loc[i-1, "id"])        and (train.loc[i, "discourse_start"] - train.loc[i-1, "discourse_end"] > 0)):
        train.loc[i, 'gap_before'] = True
        train.loc[i, 'gap_length'] = train.loc[i, "discourse_start"] - train.loc[i-1, "discourse_end"]
    #gap if the first discourse of an new essay does not start at 0
    elif ((train.loc[i, "id"] != train.loc[i-1, "id"])        and (train.loc[i, "discourse_start"] != 0)):
        train.loc[i, 'gap_before'] = True
        train.loc[i, 'gap_length'] = train.loc[i, "discourse_start"]

train['gap_end_length'] = train.essay_len - train.discourse_end


# In[10]:


train.describe()


# In[11]:


px.histogram(train, x="discourse_words", nbins=1000)


# In[12]:


def add_gap_rows(essay):
    cols_to_keep = ['discourse_start', 'discourse_end', 'discourse_type', 'gap_length', 'gap_end_length', 'text']
    df_essay = train.query('id == @essay')[cols_to_keep].reset_index(drop = True)

    #index new row
    insert_row = len(df_essay)

    #add gaps in between
    for i in range(len(df_essay)):
        if df_essay.loc[i,"gap_length"] >0:
            start = df_essay.loc[i-1, "discourse_end"] 
            end = df_essay.loc[i, 'discourse_start'] 
            text = df_essay.loc[i, 'text']
            disc_type = "Nothing"
            gap_end = np.nan
            gap = np.nan
            df_essay.loc[insert_row] = [start, end, disc_type, gap, gap_end, text]
            insert_row += 1

    df_essay = df_essay.sort_values(by = "discourse_start").reset_index(drop=True)

    #add gap at end
    if df_essay.loc[(len(df_essay)-1),'gap_end_length'] > 0:
        start = df_essay.loc[(len(df_essay)-1), "discourse_end"] 
        end = start + df_essay.loc[(len(df_essay)-1), 'gap_end_length']
        disc_type = "Nothing"
        gap_end = np.nan
        gap = np.nan
        text = df_essay.loc[(len(df_essay)-1), 'text']
        df_essay.loc[insert_row] = [start, end, disc_type, gap, gap_end, text]
        
    return(df_essay)


# In[13]:


def print_colored_essay(essay):
    df_essay = add_gap_rows(essay)

    ents = []
    for i, row in df_essay.iterrows():
        ents.append({
                        'start': int(row['discourse_start']), 
                        'end': int(row['discourse_end']), 
                        'label': row['discourse_type']
                    })

    data = row['text']

    doc2 = {
        "text": data,
        "ents": ents,
    }

    colors = {'Lead': '#EE11D0','Position': '#AB4DE1','Claim': '#1EDE71','Evidence': '#33FAFA','Counterclaim': '#4253C1','Concluding Statement': 'yellow','Rebuttal': 'red'}
    options = {"ents": df_essay.discourse_type.unique().tolist(), "colors": colors}
    spacy.displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True);


# In[14]:


add_gap_rows("129497C3E0FC")


# In[15]:


print_colored_essay("129497C3E0FC")


# In[16]:


print_colored_essay("7330313ED3F0")


# In[17]:


list(train[train.id=='7330313ED3F0'].discourse_text.values)


# In[18]:


train


# In[19]:


train['id'].value_counts()


# In[20]:


id2dis = dict(enumerate(['Nothing'] + list(train.discourse_type.value_counts().index)))
dis2id = {v: k for k,v in id2dis.items()}


# In[21]:


id2dis


# In[22]:


dis2id


# In[23]:


train['discourse_type_id'] = train['discourse_type'].apply(lambda x: dis2id[x])


# In[24]:


train = train.merge(folds, on=['id'], how='left')


# In[25]:


train.head(2)


# In[26]:


# dfx = pd.get_dummies(train, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()
# cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
# dfx = dfx[cols]
# mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# labels = [c for c in dfx.columns if c != "id"]
# dfx_labels = dfx[labels]
# dfx["kfold"] = -1

# for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
#     print(len(trn_), len(val_))
#     dfx.loc[val_, "kfold"] = fold

# train = train.merge(dfx[["id", "kfold"]], on="id", how="left")


# In[27]:


# grp = train.groupby(["kfold", 'discourse_type'], as_index=False).count()
# display(grp.pivot(index='discourse_type', columns='kfold', values='id').T)

# assert len(train.groupby(["id", "kfold"], as_index=False).count()) == train["id"].nunique()


# In[28]:


# # 1015 1018 1028 1035 1046
# np.random.seed(1046)
# dfx = train[['id']].drop_duplicates()
# dfx['kfold'] = [np.random.randint(5) for _ in range(len(dfx))]
# if 'kfold' in train.columns:
#   train.drop('kfold', axis=1, inplace=True)
# train = train.merge(dfx[["id", "kfold"]], on="id", how="left")
# grp = train.groupby(["kfold", 'discourse_type'], as_index=False).count()
# display(grp.pivot(index='discourse_type', columns='kfold', values='id').T)

# assert len(train.groupby(["id", "kfold"], as_index=False).count()) == train["id"].nunique()


# In[29]:


train.reset_index().to_feather('../input/feedback-prize-2021/train.corrected.fea')


# In[30]:



# In[31]:



# In[32]:

# In[33]:


train[train.discourse_words > 637]


# In[34]:


train[train.id=='BE01ACCDF251']

