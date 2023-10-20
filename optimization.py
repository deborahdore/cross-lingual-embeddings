from functools import partial

import ray
from loguru import logger
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from config.path import best_model_config_file, model_dir, model_file, plot_file
from train import train_autoencoder
from utils.utils import write_json


def optimization(corpus, model_config_file, vocab_fr, vocab_it):
	logger.info("[optimization] starting optimization")

	# create configuration to try out
	config = {
		"len_vocab_it" : 30004,
		"len_vocab_fr" : 30004,
		"output_dim_it": 100,
		"output_dim_fr": 100,
		"num_epochs"   : 200,
		"patience"     : 10,
		"batch_size"   : tune.choice([16, 32, 64, 128]),
		"lr"           : tune.loguniform(1e-4, 1e-1),
		"embedding_dim": tune.choice([100, 150, 200]),
		"hidden_dim"   : tune.choice([16, 32, 64, 128]),
		"num_layers"   : tune.choice([1, 2, 3]),
		"enc_dropout"  : tune.loguniform(0.1, 0.3),
		"dec_dropout"  : tune.loguniform(0.1, 0.3),
		"alpha"        : tune.loguniform(1, 10),
		"beta"         : tune.loguniform(1, 10)}

	# scheduler to minimize loss
	scheduler = ASHAScheduler(metric="loss", mode="min", max_t=25, grace_period=1, reduction_factor=2)

	try:
		ray.init()
		result = tune.run(partial(train_autoencoder,
								  corpus=corpus,
								  model_config_file=model_config_file,
								  vocab_fr=vocab_fr,
								  vocab_it=vocab_it,
								  model_file=model_file,
								  plot_file=plot_file,
								  optimize=True),
						  config=config,
						  num_samples=10,
						  scheduler=scheduler,
						  local_dir=model_dir,
						  verbose=0)

		best_trial = result.get_best_trial("loss", "min", "last")
		logger.info(f"Best trial config: {best_trial.config}")
		write_json(best_trial.config, best_model_config_file)

		train_autoencoder(best_trial.config,
						  corpus,
						  model_config_file,
						  vocab_fr,
						  vocab_it,
						  model_file,
						  plot_file,
						  optimize=False)

	finally:
		ray.shutdown()
