import glob
import os
import gezi

v1 = 40
mns1 = [    
]

v2 = 47
mns2 = []

v2 = 58
mns2 = []
v3 = 60
# mns3 = [  
#   'deberta-v3.start', 
#   'deberta-v3.end',
#   'deberta-v3.se',
#   'deberta-v3.start.stride-256',
 
#   'deberta.start',
#   # 'deberta-xlarge.start',
#   'deberta.end',
#   # 'deberta-xlarge.end',
#   'deberta.start.stride-256',
#   # 'deberta-xlarge.start.stride-256',
  
#   'bart.start',
  
#   'deberta-v3.start.len1536',
  
#   'deberta-v3.se.len1024',
#   # 'deberta-v3.start.len1024',
#   # 'deberta-v3.end.len1024',
  
#   'deberta-v3.start.len1024.stride-512',
#   'deberta-v3.start.len1024.stride-512.seq_encoder-0',
  
#   'deberta-v3.start.len1024.stride-256',
#   'deberta-v3.start.len1024.stride-256.seq_encoder-0',
    
#   'longformer.start.len1536',
#   ## 'deberta-v3.start.mui-end-mid',
# ]
mns3 = [
          # 'deberta-v3.start.len1024.stride-512',
          # # 'funnel-xlarge.start.len1024.half_lr.rnn_layers-2',
          # 'deberta-v3.start.nwemb-0',
          # 'bart.start',
          # 'longformer.start.len1536',
          # 'roberta.start.nwemb-0',
          # 'bart.start.len768',
          # # 'funnel.start.len1536.bs-8',
          # 'deberta-v3.start.rnn_layers-2',
          # 'bart.end.run2',
          # 'deberta-v3.mid',
          # 'deberta-v3.end.len1280',
          # 'deberta.start.stride-256',
          # 'deberta-v3.end.len1536.rnn_layers-2',
          # 'albert.start.nwemb-0',
          # 'deberta-v3.end.len1024.seq_encoder-0',
          # 'deberta-v3.se2.len1024',
          # 'longformer.start.len1600',
          # 'deberta-v3.start.len1600',
          # # 'deberta-v3.start.len1536.weight_decay',
          # 'roberta.start',
          # # 'bart.start.weight_decay',
          # 'deberta-v3.end.len1024.rnn_type-GRU',
          # 'xlnet.start',
          # 'deberta-v3.start.len1024.stride-256.seq_encoder-0',
          # 'deberta.start.mui-end-mid',
          # 'deberta-xlarge.end',
          # 'longformer.start.len1280',
          # 'electra.start.nwemb-0',
          # 'deberta-v3.se2',
          # # 'deberta-v3-nli.start.len1024',
          # # 'bird.start.len1024',
          # 'deberta-v3.se',
          # 'deberta.se',
          # 'deberta-v3.start.len1024.stride-512.seq_encoder-0',
          # 'deberta-v3.start.stride-256',
          # 'deberta-v3.start.mui-end-mid',
          # 'deberta-v3.start.mark_end-0',
          # 'deberta-v3.start.len1024.rnn_bi',
          # 'deberta-v3.start.len1280.rnn_layers-2',
          # 'deberta-v3.start.len1280',
          # 'deberta-v3.start',
          # # 'funnel.start.len1536.rnn_layers-2.bs-8',
          # # 'deberta-v3.start.weight_decay',
          # 'deberta-v3.end.len1024',
          # 'deberta-xlarge.start',
          # 'deberta-v3.start.len1024',
          # 'deberta-v3.start.len1536.rnn_layers-2',
          # 'deberta-v3.start.len1024.rnn_type-GRU',
          # 'deberta-v3.start.len1280.rnn_type-GRU',
          # 'deberta-v3.end.len1280.rnn_type-GRU',
          # 'deberta-xlarge.start.stride-256',
          # 'deberta-v3.start.len1536.rnn_type-GRU',
          # 'deberta-v3.end.len1536.seq_encoder-0',
          # 'deberta.start',
          # 'deberta-v3.end.rnn_layers-2',
          # 'deberta-v3.start.len1024.stride-256',
          # 'deberta.mid',
          # 'deberta-v3.start.stride-256.seq_encoder-0',
          # 'bart.se2',
          # # 'funnel.end.len1024',
          # 'longformer.end.len1280',
          # 'tiny.start.len1536',
          # 'deberta-v3.end',
          # 'deberta-v3.se.rnn_layers-2',
          # 'deberta-v3.start.len1024.rnn_layers-2',
          # 'deberta-v3.start.len1536',
          # 'deberta-v3.se.len1024',
          # 'deberta.end',
          # 'bart.end',
          # # 'funnel.start.len1024',
          # 'deberta-v3.start.len1600.rnn_layers-2',
          # 'deberta-v3.end.len1536',
          # 'bart.start.run2',
          # 'tiny.start.len1024',
          # 'deberta-v3.start.nwemb-0.mark_end-0',
          # 'deberta-v3.mid.len1024',
          # 'deberta-v3.start.len1600.rnn_type-GRU',
          # # 'deberta-v3.start.len1024.ucm'
 ]

# mns3 = [os.path.basename(x) for x in glob.glob(f'../working/offline/{v3}/0/*') if os.path.exists(f'{x}/valid.pkl')]
# mns3 = [os.path.basename(x) for x in glob.glob(f'../working/offline/{v3}/1/*') if os.path.exists(f'{x}/valid.pkl')]
mns3 = set([os.path.basename(x) for x in glob.glob(f'../working/offline/{v3}/4/*') if os.path.exists(f'{x}/valid.pkl')]) &  \
       set([os.path.basename(x) for x in glob.glob(f'../working/offline/{v3}/3/*') if os.path.exists(f'{x}/valid.pkl')]) & \
       set([os.path.basename(x) for x in glob.glob(f'../working/offline/{v3}/2/*') if os.path.exists(f'{x}/valid.pkl')]) & \
       set([os.path.basename(x) for x in glob.glob(f'../working/offline/{v3}/1/*') if os.path.exists(f'{x}/valid.pkl')]) & \
       set([os.path.basename(x) for x in glob.glob(f'../working/offline/{v3}/0/*') if os.path.exists(f'{x}/valid.pkl')])
mns3 = list(mns3)
mns = mns1 + mns2 + mns3
ic(mns)

v = v3
# v= 47
weights_dict = {}

# offline 721 online 716 5h20min
weights_dict = { 
  'bart.start.run2': 9, #only bart model has online diff may try another run 
  'roberta.start.nwemb-0': 9,
  'deberta.start': 8,
 
  'deberta-xlarge.start': 9, 
  'deberta-xlarge.end': 9, 

  'deberta-v3.start.len1024.stride-256.seq_encoder-0': 10, 
  'deberta-v3.start.len1024.stride-256': 6,

  'deberta-v3.start.len1536': 7, 
  ## 'deberta-v3.start.len1536.rl': 7, 
  
  # # 'deberta-v3.start.len1536.rnn_type-GRU': 7, 
  # # 'deberta-v3.start.len1280.rnn_layers-2': 7, 
  # # 'deberta-v3.start.len1280.rnn_layers-2': 7, 
  
  'deberta-v3.start.len1024.rnn_bi': 8, 
  'deberta-v3.end.len1024.seq_encoder-0': 10,
  'deberta-v3.mid.len1024': 8,

  'deberta-v3.start.stride-256.seq_encoder-0': 7, 
  'deberta-v3.start.nwemb-0.mark_end-0': 10, 
  'deberta-v3.se': 10,
  'deberta-v3.se2': 10,

  'longformer.start.len1536': 6,
}

weights_dict0 = {'bart.start.run2': 6,
                        'deberta-v3.end.len1024.seq_encoder-0': 6,
                        'deberta-v3.mid.len1024': 4,
                        'deberta-v3.se': 6,
                        'deberta-v3.se2': 1,
                        'deberta-v3.start.len1024.rnn_bi': 5,
                        'deberta-v3.start.len1024.stride-256': 6,
                        'deberta-v3.start.len1024.stride-256.seq_encoder-0': 10,
                        'deberta-v3.start.len1536': 4,
                        # 'deberta-v3.start.len1536': 2,
                        # 'deberta-v3.start.len1280.rnn_layers-2': 2,
                        # 'deberta-v3.start.len1536.rnn_type-GRU': 4,
                        'deberta-v3.start.nwemb-0.mark_end-0': 8,
                        'deberta-v3.start.stride-256.seq_encoder-0': 9,
                        'deberta-xlarge.end': 0,
                        'deberta-xlarge.start': 6,
                        'deberta.start': 6,
                        'longformer.start.len1536': 9,
                        'roberta.start.nwemb-0': 6}
weights_dict1 = {'bart.start.run2': 7,
                        'deberta-v3.end.len1024.seq_encoder-0': 10,
                        'deberta-v3.mid.len1024': 6,
                        'deberta-v3.se': 2,
                        'deberta-v3.se2': 7,
                        'deberta-v3.start.len1024.rnn_bi': 8,
                        'deberta-v3.start.len1024.stride-256': 10,
                        'deberta-v3.start.len1024.stride-256.seq_encoder-0': 7,
                        'deberta-v3.start.len1536': 8,
                        # 'deberta-v3.start.len1536': 4,
                        # 'deberta-v3.start.len280.rnn_layers-2': 4,
                        # 'deberta-v3.start.len1536.rnn_type-GRU': 8,
                        'deberta-v3.start.nwemb-0.mark_end-0': 8,
                        'deberta-v3.start.stride-256.seq_encoder-0': 7,
                        'deberta-xlarge.end': 7,
                        'deberta-xlarge.start': 10,
                        'deberta.start': 6,
                        'longformer.start.len1536': 8,
                        'roberta.start.nwemb-0': 5}
weights_dict2 = {'bart.start.run2': 6,
                        'deberta-v3.end.len1024.seq_encoder-0': 3,
                        'deberta-v3.mid.len1024': 4,
                        'deberta-v3.se': 4,
                        'deberta-v3.se2': 7,
                        'deberta-v3.start.len1024.rnn_bi': 3,
                        'deberta-v3.start.len1024.stride-256': 7,
                        'deberta-v3.start.len1024.stride-256.seq_encoder-0': 2,
                        'deberta-v3.start.len1536': 3,
                        # 'deberta-v3.start.len1536': 1.5,
                        # 'deberta-v3.start.len1280.rnn_layers-2': 1.5,
                        # 'deberta-v3.start.len1536.rnn_type-GRU': 3,
                        'deberta-v3.start.nwemb-0.mark_end-0': 9,
                        'deberta-v3.start.stride-256.seq_encoder-0': 9,
                        'deberta-xlarge.end': 8,
                        'deberta-xlarge.start': 10,
                        'deberta.start': 6,
                        'longformer.start.len1536': 7,
                        'roberta.start.nwemb-0': 7}

# weights_dict = {
#   'bart.start.run2': 9,
#   'deberta-v3.end.len1024.seq_encoder-0': 2,
#   'deberta-v3.mid.len1024': 3,
#   'deberta-v3.se': 9,
#   'deberta-v3.se2': 6,
#   'deberta-v3.start.len1024.rnn_bi': 6,
#   'deberta-v3.start.len1024.stride-256': 9,
#   'deberta-v3.start.len1024.stride-256.seq_encoder-0': 5,
#   'deberta-v3.start.len1536': 0,
#   'deberta-v3.start.nwemb-0.mark_end-0': 0,
#   'deberta-v3.start.stride-256.seq_encoder-0': 10,
#   'deberta-xlarge.end': 9,
#   'deberta-xlarge.start': 5,
#   'deberta.start': 0,
#   'longformer.start.len1536': 10,
#   'roberta.start.nwemb-0': 3
# }

weights_dicts = [weights_dict0, weights_dict1, weights_dict2]

def get_weight(x, idx=0):
  weight = 1
  # return 1
  if x in weights_dict:
    try:
      return weights_dicts[idx][x]
    except Exception:
      return weights_dict[x]
  return max(weight, 1)

if weights_dict:
  mns = [x for x in mns if x in weights_dict]
  mns1 = [x for x in mns1 if x in weights_dict]
  mns2 = [x for x in mns2 if x in weights_dict]
  mns3 = [x for x in mns3 if x in weights_dict]
# mns = sorted(mns)

weights = [get_weight(x) for x in mns]
weights2 = [get_weight(x, 1) for x in mns]
weights3 = [get_weight(x, 2) for x in mns]
# weights2 = weights.copy()
# weights3 = weights.copy()
# weights.extend([1
# ] * 100)
ic(list(zip(mns, weights)), len(mns))
# ic(gezi.sort_byval(weights_dict))

SAVE_PRED = 0
# SAVE_PRED = 1
