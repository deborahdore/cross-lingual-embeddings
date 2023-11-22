import ast
import sys

from loguru import logger
from sklearn.model_selection import train_test_split

from config.path import (aligned_file,
						 dataset_dir,
						 download_corpus,
						 eng_lang_file,
						 lang_file,
						 model_config_file,
						 model_file,
						 plot_file,
						 processed_file,
						 study_result_dir, )
from optimization import optimization
from study import ablation_study
from test import visualize_latent_space
from train import train_autoencoder
from utils.dataset import create_vocab_and_dataset
from utils.processing import align_dataset, process_dataset
from utils.utils import download_from_url, read_file, read_file_to_df, read_json


def parse_command_line():
	"""
	The parse_command_line function parses the command line arguments and returns three boolean values:
		generate - whether or not to generate a new dataset
		optimize - whether or not to optimize the model parameters using Bayesian optimization
		ablation - whether or not to perform an ablation study on the model

	:return: A tuple of three boolean values
	"""
	generate = False
	optimize = False
	ablation = False
	if "--generate" in sys.argv:
		generate = ast.literal_eval(sys.argv[sys.argv.index("--generate") + 1])

	if "--optimize" in sys.argv:
		optimize = ast.literal_eval(sys.argv[sys.argv.index("--optimize") + 1])

	if "--ablation" in sys.argv:
		ablation = ast.literal_eval(sys.argv[sys.argv.index("--ablation") + 1])
		if ablation:
			assert not optimize

	return generate, optimize, ablation


def main():
	"""
	The main function of the script.
	"""
	if len(sys.argv) < 5:
		logger.error("[main] missing arguments from command line")
		raise Exception("missing arguments from command line")

	generate_param, optimize_param, ablation_study_param = parse_command_line()

	if generate_param:

		download_from_url(download_corpus.format(lang="it"), dataset_dir, "it")
		download_from_url(download_corpus.format(lang="fr"), dataset_dir, "fr")

		it_file = read_file(lang_file.format(lang="it"))
		eng_it_file = read_file(eng_lang_file.format(lang="it"))

		fr_file = read_file(lang_file.format(lang="fr"))
		eng_fr_file = read_file(eng_lang_file.format(lang="fr"))

		# creating aligned dataset
		align_dataset(fr_file, eng_fr_file, it_file, eng_it_file, aligned_file)

		# run nlp pipeline
		corpus = process_dataset(aligned_file, processed_file, plot_file)

	else:
		corpus = read_file_to_df(processed_file).sample(frac=0.05)

	corpus_tokenized, vocab_fr, vocab_it = create_vocab_and_dataset(corpus)

	corpus4training, corpus4testing = train_test_split(corpus_tokenized, test_size=0.05)

	corpus4training = corpus4training.reset_index(drop=True)
	corpus4testing = corpus4testing.reset_index(drop=True)

	if optimize_param:
		# find optimal hyperparameters
		optimization(corpus4training, study_result_dir, vocab_fr, vocab_it)
	elif ablation_study_param:
		# run ablation study
		ablation_study(corpus4training, model_config_file, vocab_fr, vocab_it, model_file, plot_file, study_result_dir)
	else:
		# normal training
		config = read_json(model_config_file)
		train_autoencoder(config,
						  corpus4training,
						  vocab_fr,
						  vocab_it,
						  model_file,
						  plot_file,
						  study_result_dir,
						  optimize=False)
		visualize_latent_space(config, corpus4testing, model_file, plot_file, vocab_fr, vocab_it)


if __name__ == '__main__':
	main()
