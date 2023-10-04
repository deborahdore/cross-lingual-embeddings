from functools import partial

import ray
from loguru import logger
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from config.path import best_model_config_file, model_dir, model_file, plot_file
from test import test
from train import train_autoencoder
from utils.utils import write_json


def optimization(train_loader, val_loader, test_loader):
	logger.info("[optimization] starting optimization")

	optimize = True
	train_loader = ray.put(train_loader)
	val_loader = ray.put(val_loader)
	test_loader = ray.put(test_loader)

	# create configuration to try out
	config = {
		"len_vocab"        : 17504965,
		"output_dim_it"    : 114,
		"output_dim_fr"    : 128,
		"num_epochs"       : 25,
		"batch_size"       : 32,
		"patience"         : 3,
		"lr"               : tune.loguniform(1e-4, 1e-1),
		"enc_embedding_dim": tune.choice([25, 50, 100, 150]),
		"enc_hidden_dim"   : tune.choice([16, 32, 64]),
		"enc_num_layers"   : tune.choice([1, 2, 3]),
		"enc_dropout"      : tune.loguniform(0.2, 0.7),
		"proj_dim"         : tune.choice([8, 16, 32]),
		"dec_hidden_dim"   : tune.choice([16, 32, 64]),
		"dec_num_layers"   : tune.choice([1, 2, 3]),
		"dec_dropout"      : tune.loguniform(0.2, 0.7)}

	# scheduler to minimize loss
	scheduler = ASHAScheduler(metric="loss", mode="min", max_t=25, grace_period=1, reduction_factor=2)

	try:
		result = tune.run(partial(train_autoencoder,
								  train_loader_wrapper=train_loader,
								  val_loader_wrapper=val_loader,
								  model_file=model_file,
								  plot_file=plot_file,
								  optimize=optimize),
						  config=config,
						  num_samples=10,
						  scheduler=scheduler,
						  local_dir=model_dir,
						  verbose=0)

		best_trial = result.get_best_trial("loss", "min", "last")
		logger.info(f"Best trial config: {best_trial.config}")
		write_json(best_trial.config, best_model_config_file)
		test(best_trial.config, test_loader, model_file, optimize)
	finally:
		ray.shutdown()
