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

import tensorflow as tf
import tensorflow.keras.backend as K
        
def dice_coef(y_true, y_pred, smooth=1, ignore_background=False):
  loss = 0
  start = 0 if not ignore_background else 1
  for i in range(start, y_pred.shape[-1]):
    intersection = K.sum(y_true[:,:,i] * y_pred[:,:,i], axis=[1])
    union = K.sum(y_true[:,:,i], axis=[1]) + K.sum(y_pred[:,:,i], axis=[1])
    loss_ = (2. * intersection + smooth) / (union + smooth)
    if i == start:
      loss = loss_
    else:
      loss += loss_
  loss /= y_pred.shape[-1]
  return loss
  
def dice_coef_loss(y_true, y_pred, smooth=1, ignore_background=False):
  return 1 - dice_coef(y_true, y_pred, smooth=smooth, ignore_background=ignore_background)

def tversky_loss(y_true, y_pred):
  alpha = 0.5
  beta  = 0.5
  
  ones = K.ones(K.shape(y_true))
  p0 = y_pred      # proba that voxels are class i
  p1 = ones - y_pred # proba that voxels are not class i
  g0 = y_true
  g1 = ones - y_true
  
  num = K.sum(p0 * g0, (0,1,2))
  den = num + alpha * K.sum(p0 * g1, (0,1,2)) + beta * K.sum(p1 * g0, (0,1,2))
  
  T = K.sum(num / den) # when summing over classes, T has dynamic range [0 Ncl]
  
  Ncl = K.cast(K.shape(y_true)[-1], 'float32')
  return Ncl-T
      