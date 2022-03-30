#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   preprocess.py
#        \author   chenghuige  
#          \date   2022-01-24 22:32:10.988305
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import re
from absl import flags
FLAGS = flags.FLAGS

# https://www.kaggle.com/aliasgherman/vocabulary-txt-coverage-improvement-80-to-99/notebook

aam_misspell_dict = {
                'colour':'color',
                'centre':'center',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                "genericname": "someone",
                 "driveless" : "driverless",
                 "canidates" : "candidates",
                 "electorial" : "electoral",
                 "genericschool" : "school",
                 "polution" : "pollution",
                 "enviorment" : "environment",
                 "diffrent" : "different",
                 "benifit" : "benefit",
                 "schoolname" : "school",
                 "artical" : "article",
                 "elctoral" : "electoral",
                 "genericcity" : "city",
                 "recieves" : "receives",
                 "completly" : "completely",
                 "enviornment" : "environment",
                 "somthing" : "something",
                 "everyones" : "everyone",
                 "oppurtunity" : "opportunity",
                 "benifits" : "benefits",
                 "benificial" : "beneficial",
                 "tecnology" : "technology",
                 "paragragh" : "paragraph",
                 "differnt" : "different",
                 "reist" : "resist",
                 "probaly" : "probably",
                 "usuage" : "usage",
                 "activitys" : "activities",
                 "experince" : "experience",
                 "oppertunity" : "opportunity",
                 "collge" : "college",
                 "presedent" : "president",
                 "dosent" : "doesnt",
                 "propername" : "name",
                 "eletoral" : "electoral",
                 "diffcult" : "difficult",
                 "desicision" : "decision"
 }
aam_misspell_dict.update( {"shouldnt" : "shant" })
aam_misspell_dict.update( {"teacherdesigned" : "designed",
                      "studentname" : "myself",
                      "studentdesigned" : "designed",
                      "teachername" : "teacher",
                      "winnertakeall" : "winner-take-all"})

aam_misspell_dict2 = {}
for word, word2 in aam_misspell_dict.items():
  aam_misspell_dict2[word.capitalize()] = word2.capitalize()
aam_misspell_dict.update(aam_misspell_dict2)

# char codes: https://unicode-table.com/en/#basic-latin
accent_map = {
    u'\u00c0': u'A',
    u'\u00c1': u'A',
    u'\u00c2': u'A',
    u'\u00c3': u'A',
    u'\u00c4': u'A',
    u'\u00c5': u'A',
    u'\u00c6': u'A',
    u'\u00c7': u'C',
    u'\u00c8': u'E',
    u'\u00c9': u'E',
    u'\u00ca': u'E',
    u'\u00cb': u'E',
    u'\u00cc': u'I',
    u'\u00cd': u'I',
    u'\u00ce': u'I',
    u'\u00cf': u'I',
    u'\u00d0': u'D',
    u'\u00d1': u'N',
    u'\u00d2': u'O',
    u'\u00d3': u'O',
    u'\u00d4': u'O',
    u'\u00d5': u'O',
    u'\u00d6': u'O',
    u'\u00d7': u'x',
    u'\u00d8': u'0',
    u'\u00d9': u'U',
    u'\u00da': u'U',
    u'\u00db': u'U',
    u'\u00dc': u'U',
    u'\u00dd': u'Y',
    u'\u00df': u'B',
    u'\u00e0': u'a',
    u'\u00e1': u'a',
    u'\u00e2': u'a',
    u'\u00e3': u'a',
    u'\u00e4': u'a',
    u'\u00e5': u'a',
    u'\u00e6': u'a',
    u'\u00e7': u'c',
    u'\u00e8': u'e',
    u'\u00e9': u'e',
    u'\u00ea': u'e',
    u'\u00eb': u'e',
    u'\u00ec': u'i',
    u'\u00ed': u'i',
    u'\u00ee': u'i',
    u'\u00ef': u'i',
    u'\u00f1': u'n',
    u'\u00f2': u'o',
    u'\u00f3': u'o',
    u'\u00f4': u'o',
    u'\u00f5': u'o',
    u'\u00f6': u'o',
    u'\u00f8': u'0',
    u'\u00f9': u'u',
    u'\u00fa': u'u',
    u'\u00fb': u'u',
    u'\u00fc': u'u'
}

def accent_remove (m):
  return accent_map[m.group(0)]

def preprocess(x):
  if x == FLAGS.br:
    return x
  
  if x == '\n' or x == '[SEP]' or x == '[MASK]':
    return x
  
  if FLAGS.tolower or FLAGS.normalize:
    x = x.lower()
  
  if not FLAGS.normalize:
    return x
  
  # 影响不大
  x = x.replace("n't", "nt")

  if len(x.strip()) == 1:
    return x #special case if a punctuation was the only alphabet in the token.

  # 影响巨大 remove能提升整体效果但是在position和claim会变差特别是postion
  x_ = x
  # if FLAGS.remove_punct:
  #rv1 影响较大 似乎没有不好的影响 所有类别都ok
  for punct in "/-'&":
    x = x.replace(punct, '')
    
  #rv2 影响巨大 position claim变差 但是conclude和evidence提升巨大
  for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
    x = x.replace(punct, '')
      
  #rv3 影响很小 可以不加
  # x = re.sub('[0-9]{1,}', '#', x) #replace all numbers by #

  # if FLAGS.norm_punct:
  if len(x.strip()) < 1:
    x = '.' #if it was all punctuations like ------ or ..... or .;?!!. Then we return only a period to keep token consistent performance.
  # else:
  #   if len(x.strip()) < 1:
  #     x = x_[-1] #if it was all punctuations like ------ or ..... or .;?!!. Then we return only a period to keep token consistent performance.
  
  # 影响不大
  x = re.sub(u'([\u00C0-\u00FC])', accent_remove, x.encode().decode('utf-8'))
  
  # 影响不大 可以保留
  if FLAGS.fix_misspell:
    if x in aam_misspell_dict:
      x = aam_misspell_dict[x]
    
  return x  
