import ast
import sys
from functools import partial

import ray
from loguru import logger
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split

from config.path import (aligned_file, best_model_config_file, eng_lang_file, lang_file,
                         model_config_file, model_dir, model_file, plot_file, processed_file, vocab_file)
from test import test
from train import train
from utils.dataset import prepare_dataset
from utils.processing import align_dataset, process_dataset
from utils.utils import read_file, read_file_to_df, read_json, write_json


def parse_command_line():
	gd = False
	opt = False
	if "--generate" in sys.argv:
		gd = ast.literal_eval(sys.argv[sys.argv.index("--generate") + 1])
	if "--optimize" in sys.argv:
		opt = ast.literal_eval(sys.argv[sys.argv.index("--optimize") + 1])
	return gd, opt


if __name__ == '__main__':

	if len(sys.argv) < 5:
		logger.error("[main] missing arguments from command line")
		raise Exception("missing arguments from command line")

	generate, optimize = parse_command_line()

	if generate:

		# download_from_url(download_corpus.format(lang="it"), dataset_dir, "it")
		# download_from_url(download_corpus.format(lang="fr"), dataset_dir, "fr")

		# it_file = read_file(lang_file.format(lang = "it"))
		# eng_it_file = read_file(eng_lang_file.format(lang = "it"))
		#
		# fr_file = read_file(lang_file.format(lang = "fr"))
		# eng_fr_file = read_file(eng_lang_file.format(lang = "fr"))
		#
		# align_dataset(fr_file, eng_fr_file, it_file, eng_it_file, aligned_file)

		corpus, vocab_fr, vocab_it = process_dataset(
				aligned_file, processed_file, vocab_file, model_config_file, plot_file
				)

	else:
		corpus = read_file_to_df(processed_file)
		corpus['french'] = corpus['french'].apply(ast.literal_eval)
		corpus['italian'] = corpus['italian'].apply(ast.literal_eval)

		vocab_it = read_file_to_df(vocab_file.format(lang = "it"))['words'].tolist()
		vocab_fr = read_file_to_df(vocab_file.format(lang = "fr"))['words'].tolist()

	corpus_4_model_training, corpus_4_testing = train_test_split(corpus, test_size = 0.1, random_state = 42)
	corpus_4_model_training = corpus_4_model_training.reset_index(drop = True)
	corpus_4_testing = corpus_4_testing.reset_index(drop = True)

	config = read_json(model_config_file)
	train_dataset, val_dataset, test_dataset = prepare_dataset(corpus_4_model_training, config)

	if optimize:

		train_loader = ray.put(train_dataset)
		val_loader = ray.put(val_dataset)
		test_loader = ray.put(test_dataset)
		config = {"len_vocab_fr":    96807, "len_vocab_it": 114436, "output_dim": 100, "num_epochs": 25,
		          "batch_size":      32, "patience": 3, "embedding_dim": tune.choice([25, 50, 100, 150]),
		          "hidden_dim":      tune.choice([16, 32, 64]), "ls_dim": tune.choice([8, 16, 32]),
		          "hidden_lstm_dim": tune.choice([16, 32, 64]), "lr": tune.loguniform(1e-4, 1e-1),
		          "num_layers1":     tune.choice([1, 2, 3]), "dropout1": tune.loguniform(0.2, 0.7),
		          "num_layers2":     tune.choice([1, 2, 3]), "dropout2": tune.loguniform(0.2, 0.7)}

		scheduler = ASHAScheduler(metric = "loss", mode = "min", max_t = 25, grace_period = 1, reduction_factor = 2)

		try:
			result = tune.run(
					partial(
							train, train_loader_wrapper = train_loader, val_loader_wrapper = val_loader,
							model_file = model_file, plot_file = plot_file, optimize = optimize
							), config = config, num_samples = 10, scheduler = scheduler, local_dir = model_dir,
					verbose = 1
					)

			best_trial = result.get_best_trial("loss", "min", "last")
			logger.info(f"Best trial config: {best_trial.config}")
			write_json(best_trial.config, best_model_config_file)
			test(best_trial.config, test_loader, model_file, optimize)
		finally:
			ray.shutdown()

	else:
		train(config, train_dataset, val_dataset, model_file, plot_file, optimize)
		test(config, test_dataset, model_file, optimize)
