import os
import time
import errno 
import torch
import random
import numpy as np
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs

def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)

reproducibility(4)

train_file = "pretraining_data.txt"

model_args = LanguageModelingArgs(output_dir = "models/CStyleBERT", 
                                  overwrite_output_dir = True,
                                  max_seq_length = 512,
                                  vocab_size = 50000, 
                                  train_batch_size = 64, 
                                  num_train_epochs = 7
                                  )
model_args.config = {"max_position_embeddings" : 1026}

start_time = time.time()
try:
    model = LanguageModelingModel("roberta", 
                                  None, 
                                  args = model_args, 
                                  train_files = train_file, 
                                  use_cuda = True, 
                                  cuda_device = 0
                                  )
    model.train_model(train_file)
except IOError as err: 
    if err.errno == errno.EPIPE: 
      pass

print(f"Pre-training time: {time.time() - start_time} sec")