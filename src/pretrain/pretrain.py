import torch
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs

class Pretrain:
	def pretrain(self):
		torch.manual_seed(60)
		train_file = None
		model_args = LanguageModelingArgs(output_dir = "models/cstyleBERTi-std",
									max_seq_length = 512, 
									overwrite_output_dir = True,
									learning_rate = 3e-05,
									train_batch_size = 32, 
									num_train_epochs = 10)
		model_args.config = {"num_hidden_layers": 12, 
					   "num_attention_heads" : 12, 
					   "max_position_embeddings" : 1026,
					   "vocab_size" : 10000,
					   "type_vocab_size" : 1
					   } 
		model = LanguageModelingModel("roberta", None, args = model_args, train_files = train_file, use_cuda = True, cuda_device = 1)
		model.train_model(train_file)