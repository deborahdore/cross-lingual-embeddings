from functools import partial

import pandas as pd
import ray
from loguru import logger
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torchtext.vocab import Vocab

from config.path import best_model_config_file, model_dir, model_file, plot_file
from train import train_autoencoder
from utils.utils import write_json


def optimization(corpus: pd.DataFrame, study_result_dir: str, vocab_fr: Vocab, vocab_it: Vocab):
	"""
	The optimization function is used to find the best hyperparameters for our model.
	It uses Ray Tune, a library that allows us to run multiple experiments in parallel.
	The function takes as input the corpus of data we want to train on, and two vocabularies (one for each language).
	It then creates a configuration dictionary with all possible combinations of hyperparameters we want to try out.
	We use this config dictionary when calling tune.run(). This function will create 25 different models using random
	combinations from our config dict and train them in parallel on GPUs (if available). We also specify an
	ASHAScheduler

	:param corpus: pd.DataFrame: Pass the corpus to the function
	:param study_result_dir: str: Store the results of the optimization in a folder
	:param vocab_fr: Vocab: Pass the vocabulary of the french corpus to train_model
	:param vocab_it: Vocab: Pass the vocabulary of the italian language to the function
	:return: A dictionary of the best parameters
	"""
	logger.info("[optimization] starting optimization")

	# create configuration to try out
	config = {
		"num_epochs"   : 50,
		"patience"     : 5,
		"batch_size"   : tune.choice([16, 32, 64, 128]),
		"lr"           : tune.loguniform(1e-4, 1e-1),
		"embedding_dim": tune.choice([50, 100, 150]),
		"hidden_dim"   : tune.choice([64, 128, 256]),
		"hidden_dim2"  : tune.choice([16, 32, 54]),
		"num_layers"   : tune.choice([1, 2, 3]),
		"enc_dropout"  : tune.loguniform(0.1, 0.3),
		"dec_dropout"  : tune.loguniform(0.1, 0.3),
		"alpha"        : tune.loguniform(1, 1.5),
		"beta"         : tune.loguniform(1, 1.5)}

	# scheduler to minimize loss
	scheduler = ASHAScheduler(metric="loss", mode="min", max_t=25, grace_period=1, reduction_factor=2)

	try:
		ray.init()
		corpus = ray.put(corpus)
		result = tune.run(partial(train_autoencoder,
								  corpus=corpus,
								  vocab_fr=vocab_fr,
								  vocab_it=vocab_it,
								  model_file=model_file,
								  plot_file=plot_file,
								  study_result_dir=study_result_dir,
								  optimize=True),
						  config=config,
						  num_samples=25,
						  resources_per_trial={"gpu": 1},
						  scheduler=scheduler,
						  local_dir=model_dir,
						  verbose=0)

		best_trial = result.get_best_trial("loss", "min", "last")
		logger.info(f"Best trial config: {best_trial.config}")
		write_json(best_trial.config, best_model_config_file)

		train_autoencoder(best_trial.config,
						  corpus,
						  vocab_fr,
						  vocab_it,
						  model_file,
						  plot_file,
						  study_result_dir,
						  optimize=False)

	finally:
		ray.shutdown()
