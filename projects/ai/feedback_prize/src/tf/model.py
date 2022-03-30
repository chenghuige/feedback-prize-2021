from absl import app, flags
FLAGS = flags.FLAGS

import copy
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
import tensorflow_addons as tfa
from transformers import TFAutoModel, AutoConfig
import tensorflow.keras.backend as K

import gezi
import melt as mt
from src.config import *
from src import util
from src.util import *
from src.tf.loss import *


class Model(mt.Model):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)  
    
    backbone = FLAGS.backbone if not FLAGS.continue_pretrain else f'{FLAGS.idir}/pretrain/bert'
    
    try:
      self.backbone = TFAutoModel.from_pretrained(backbone)
    except Exception:
      self.backbone = TFAutoModel.from_pretrained(backbone, from_pt=True)
    
    tokenizer = util.get_tokenizer(FLAGS.backbone)
    self.backbone.resize_token_embeddings(len(tokenizer))
    # config = AutoConfig.from_pretrained(backbone)
    # self.backbone = TFAutoModel.from_config(config)
    
    # hidden_size = config.hidden_size 
    # vocab_size = config.vocab_size
    
    # self.emb = Embedding(vocab_size, hidden_size)
    # RNN = getattr(keras.layers, 'LSTM')
    # emb_dim = 768
    # emb_dim = int(emb_dim / 2)
    # self.seq_encoder = keras.layers.Bidirectional(RNN(emb_dim, return_sequences=True), name='seq_encoder')
  
    Dense = mt.layers.MultiDropout if FLAGS.mdrop else keras.layers.Dense
    
    self.dense = Dense(FLAGS.num_classes, name='token')
    if FLAGS.num_classes == NUM_CLASSES:
      self.start_dense = Dense(2, name='start')
    
    if FLAGS.cls_parts:
      self.parts_dense = Dense(1, activation='sigmoid', name='cls')
    
    if FLAGS.cls_para:
      self.para_dense = Dense(FLAGS.num_classes, name='para')
          
    # backbone_layers = [self.backbone]
    # base_layers = [layer for layer in self.layers if layer not in backbone_layers]
    # opt_layers = [base_layers, backbone_layers]
    # gezi.set('opt_layers', opt_layers)
    # ic(opt_layers[-1])
    
  def groupby(self, logits, word_ids):
    logits = mt.unsorted_segment_reduce(logits, word_ids, FLAGS.max_len + 1, combiner=FLAGS.seg_reduce_method)
    return logits[:,:-1]
   
  def call(self, inputs, training=None):    
    self.inputs = inputs
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    x = self.backbone([input_ids, attention_mask])[0]   

    pred = self.dense(x)   
    if FLAGS.records_type == 'word':
      pred = self.groupby(pred, inputs['word_ids'])
    self.logits_list = gezi.get('xs', [])
    
    if FLAGS.method == 2:
      self.start_logits = self.start_dense(x)
      if FLAGS.records_type == 'word':
        self.start_logits = self.groupby(self.start_logits, inputs['word_ids'])
      self.start_logits_list = gezi.get('xs', [])

    if 'id' in inputs:
      if FLAGS.cls_parts:
        x_cls = x[:,0]
        self.parts = self.parts_dense(x_cls) * FLAGS.max_parts
      elif hasattr(self, 'start_logits'):
        self.parts = tf.cast(tf.reduce_sum(
              tf.cast((self.start_logits[:,:,1] - self.start_logits[:,:,0]) > 0, tf.int32) * self.inputs['mask'], axis=1, keepdims=True
            ), tf.float32)
        
      if FLAGS.cls_para:
        if 'label_index' in inputs:
          x = mt.unsorted_segment_embs(x, inputs['label_index'], FLAGS.max_parts + 1, combiner=FLAGS.para_pooling)
          x = x[:, 1:, :]
          self.para_logits = self.para_dense(x) 

    return pred
      
  def get_loss_fn(self):

    def loss_fn(y_true, y_pred):
      y_true_ = tf.cast(tf.one_hot(y_true, y_pred.shape[-1]), tf.float32)
      y_true = tf.cast(y_true, tf.float32) 
      y_pred = tf.cast(y_pred, tf.float32) 

      loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True) 
      if not self.logits_list:
        loss = loss_obj(y_true, y_pred)
      else:
        loss_list = [loss_obj(y_true, logits) for logits in self.logits_list]
        loss = tf.reduce_mean(tf.stack(loss_list, 1), axis=1)
      loss = loss_obj(y_true, y_pred)
      loss *= FLAGS.token_loss_rate
       
      token_loss = mt.reduce_over(loss)
      self.scalar(token_loss, 'loss/token')
            
      if hasattr(self, 'start_logits'):
        start_loss = loss_obj(self.inputs['start'], self.start_logits)
        if not self.start_logits_list:
          start_loss = loss_obj(self.inputs['start'], self.start_logits)
        else:
          loss_list = [loss_obj(self.inputs['start'], logits) for logits in self.start_logits_list]
          start_loss = tf.reduce_mean(tf.stack(loss_list, 1), axis=1)
        #mask for word id 0
        start_loss *= tf.cast(self.inputs['start_mask'], tf.float32)
        start_loss *= FLAGS.start_loss_rate
        self.scalar(mt.reduce_over(start_loss), 'loss/start')
        loss += start_loss
          
      if FLAGS.mask_loss:
        mask = self.inputs['mask']
        loss = mt.mask_loss(loss, mask, reduction=FLAGS.loss_reduction, method=FLAGS.loss_method)
      else:
        if FLAGS.loss_reduction == 'sum':
          loss = tf.reduce_sum(loss, axis=-1)
        else:
          loss = tf.reduce_mean(loss, axis=-1)

      if FLAGS.parts_loss:
        loss_obj2 = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        pred_parts = self.parts / float(FLAGS.max_parts)
        true_parts = tf.cast(self.inputs['para_count'], tf.float32) / float(FLAGS.max_parts)
        # bs,1 bs,1 -> bs
        parts_loss = loss_obj2(true_parts, pred_parts)
        parts_loss *= FLAGS.parts_loss_rate
        self.scalar(mt.reduce_over(parts_loss), 'parts_loss')
        loss += parts_loss 
        
      if hasattr(self, 'para_logits'):
        para_logits = self.para_logits
        para_label = self.inputs['label_type']
        para_loss = loss_obj(para_label, para_logits)
        para_loss = mt.mask_loss(mt.reduce_over(para_loss), self.inputs['para_mask'], 
                                 reduction=FLAGS.loss_reduction, method=FLAGS.loss_method)
        para_loss *= FLAGS.para_loss_rate
        self.scalar(mt.reduce_over(para_loss), 'loss/para')
        loss += para_loss
        
      if FLAGS.dice_loss_rate > 0:
        # raise UnImplementedError('dice_loss_rate')
        dice_loss = dice_coef_loss(y_true_, y_pred, ignore_background=FLAGS.ignore_background)
        dice_loss *= FLAGS.dice_loss_rate
        self.scalar(mt.reduce_over(dice_loss), 'loss/dice')
        loss += dice_loss
        
      loss = mt.reduce_over(loss)
      loss *= FLAGS.loss_scale
           
      return loss
    
    return loss_fn
  
  def build_model(self):
    tokens = tf.keras.layers.Input(shape=(FLAGS.max_len,), name = 'input_ids', dtype=tf.int32)
    attention = tf.keras.layers.Input(shape=(FLAGS.max_len,), name = 'attention_mask', dtype=tf.int32)
    
    inputs = {
      'input_ids': tokens,
      'attention_mask': attention
    }
    x = self.call(inputs)
    outputs = {
      'pred': x
    }
    if FLAGS.method == 2:
      outputs['start_logits'] = self.start_logits
    # model_ = tf.keras.Model(inputs=[tokens,attention], outputs=outputs) 
    model_ = tf.keras.Model(inputs=inputs, outputs=outputs) 
    return model_
  
