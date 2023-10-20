import ast
import sys

import pandas as pd
from ablation_study import ablation_study_eval
from loguru import logger
from sklearn.model_selection import train_test_split

from config.path import (aligned_file,
						 eng_lang_file,
						 lang_file,
						 model_config_file,
						 model_file,
						 plot_file,
						 processed_file,
						 study_result_dir, )
from optimization import optimization
from test import visualize_latent_space
from train import train_autoencoder
from utils.dataset import create_vocab, sequence2index
from utils.processing import align_dataset, process_dataset
from utils.utils import read_file, read_file_to_df, read_json


def parse_command_line():
	gd = False
	opt = False
	abl = False
	if "--generate" in sys.argv:
		gd = ast.literal_eval(sys.argv[sys.argv.index("--generate") + 1])
	if "--optimize" in sys.argv:
		opt = ast.literal_eval(sys.argv[sys.argv.index("--optimize") + 1])
	if "--ablation" in sys.argv:
		abl = ast.literal_eval(sys.argv[sys.argv.index("--ablation") + 1])
	return gd, opt, abl


if __name__ == '__main__':

	if len(sys.argv) < 5:
		logger.error("[main] missing arguments from command line")
		raise Exception("missing arguments from command line")

	generate, optimize, ablation_study = parse_command_line()

	if generate:

		# download_from_url(download_corpus.format(lang="it"), dataset_dir, "it")
		# download_from_url(download_corpus.format(lang="fr"), dataset_dir, "fr")

		it_file = read_file(lang_file.format(lang="it"))
		eng_it_file = read_file(eng_lang_file.format(lang="it"))

		fr_file = read_file(lang_file.format(lang="fr"))
		eng_fr_file = read_file(eng_lang_file.format(lang="fr"))

		# creating aligned dataset
		align_dataset(fr_file, eng_fr_file, it_file, eng_it_file, aligned_file)

		# run nlp pipeline
		corpus = process_dataset(aligned_file, processed_file, plot_file)

	else:
		corpus = read_file_to_df(processed_file)

	tokenized_dataset_fr, vocab_fr = create_vocab(corpus['french'], "fr_core_news_sm")
	tokenized_dataset_it, vocab_it = create_vocab(corpus['italian'], "it_core_news_sm")

	corpus_tokenized = pd.DataFrame({
		'french' : sequence2index(tokenized_dataset_fr, vocab_fr),
		'italian': sequence2index(tokenized_dataset_it, vocab_it)})

	corpus_4_model_training, corpus_4_testing = train_test_split(corpus_tokenized, test_size=0.1)
	corpus_4_model_training = corpus_4_model_training.reset_index(drop=True)
	corpus_4_testing = corpus_4_testing.reset_index(drop=True)

	if optimize:
		# find optimal hyperparameters
		optimization(corpus_4_model_training, model_config_file, study_result_dir, vocab_fr, vocab_it)
	elif ablation_study:
		ablation_study_eval(corpus_4_model_training,
							model_config_file,
							vocab_fr,
							vocab_it,
							model_file,
							plot_file,
							study_result_dir, )
	else:
		# normal training
		config = read_json(model_config_file)
		train_autoencoder(config,
						  corpus_4_model_training,
						  model_config_file,
						  vocab_fr,
						  vocab_it,
						  model_file,
						  plot_file,
						  study_result_dir,
						  optimize=False)
		visualize_latent_space(config, corpus_4_testing, model_file, plot_file, vocab_fr, vocab_it)
