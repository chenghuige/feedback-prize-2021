#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('../../../../utils')
import gezi
from transformers import AutoTokenizer

# In[3]:

backbone_name = sys.argv[1]
#backbone_name = 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(backbone_name, add_prefix_space=True)

ic("\n" in tokenizer.vocab)

gezi.save_huggingface(backbone_name, '/work/data/huggingface', tokenizer)


# In[ ]:


# # Load intial pretrained model
# model = AutoModel.from_pretrained(backbone_name)
# # size mismatch for embeddings.position_ids: copying a param with shape torch.Size([1, 514]) from checkpoint, the shape in current model is torch.Size([1, 1026]).
# # size mismatch for embeddings.position_embeddings.weight: copying a param with shape torch.Size([514, 1024]) from checkpoint, the shape in current model is torch.Size([1026, 1024]).
# # Reshape Axial Position Embeddings layer to match desired max seq length       
# model.reformer.embeddings.position_embeddings.weights[1] = torch.nn.Parameter(model.reformer.embeddings.position_embeddings.weights[1][0][:256])

# # Update the config file to match custom max seq length
# model.config.axial_pos_shape = 128, 256
# model.config.max_position_embeddings = 128*256 # 32768

# # Save model with custom max length
# output_model_path = "path/to/model"
# model.save_pretrained(output_model_path)

