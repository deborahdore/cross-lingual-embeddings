import pandas as pd
from loguru import logger
from torchtext.vocab import Vocab

from train import train_autoencoder
from utils.utils import read_json


def ablation_study(corpus_4_model_training: pd.DataFrame(),
				   model_config_file: str,
				   vocab_fr: Vocab,
				   vocab_it: Vocab,
				   model_file: str,
				   plot_file: str,
				   study_result_dir: str):
	config = read_json(model_config_file)

	# ------- train without dropout ------- #
	logger.info("[ablation_study_eval] training without dropout")
	config_without_dropout = config
	config_without_dropout['enc_dropout'] = 0
	config_without_dropout['dec_dropout'] = 0
	train_autoencoder(config_without_dropout,
					  corpus_4_model_training,
					  model_config_file,
					  vocab_fr,
					  vocab_it,
					  model_file,
					  plot_file,
					  study_result_dir,
					  optimize=False,
					  ablation_study=True)

	# ------- train with just one layer of lstm ------- #
	logger.info("[ablation_study_eval] training with just one layer of lstm")
	config_one_layer = config
	config_one_layer['num_layers'] = 1
	train_autoencoder(config_one_layer,
					  corpus_4_model_training,
					  model_config_file,
					  vocab_fr,
					  vocab_it,
					  model_file,
					  plot_file,
					  study_result_dir,
					  optimize=False,
					  ablation_study=True)

	# ------- train without contrastive loss ------- #
	logger.info("[ablation_study_eval] training without contrastive loss")
	config_zero_alpha = config
	config_zero_alpha['alpha'] = 0
	train_autoencoder(config_zero_alpha,
					  corpus_4_model_training,
					  model_config_file,
					  vocab_fr,
					  vocab_it,
					  model_file,
					  plot_file,
					  study_result_dir,
					  optimize=False,
					  ablation_study=True)

	# ------- train without reconstruction loss ------- #
	logger.info("[ablation_study_eval] training without reconstruction loss")
	config_zero_beta = config
	config_zero_beta['beta'] = 0
	train_autoencoder(config_zero_beta,
					  corpus_4_model_training,
					  model_config_file,
					  vocab_fr,
					  vocab_it,
					  model_file,
					  plot_file,
					  study_result_dir,
					  optimize=False,
					  ablation_study=True)
