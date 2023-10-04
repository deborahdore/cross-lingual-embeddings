import ast
import sys

from loguru import logger
from sklearn.model_selection import train_test_split

from config.path import (aligned_file, eng_lang_file, lang_file, model_config_file, model_file, plot_file,
						 processed_file)
from optimization import optimization
from train import train
from utils.dataset import create_vocab, prepare_dataset
from utils.processing import align_dataset, process_dataset
from utils.utils import read_file, read_file_to_df, read_json


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

		it_file = read_file(lang_file.format(lang="it"))
		eng_it_file = read_file(eng_lang_file.format(lang="it"))

		fr_file = read_file(lang_file.format(lang="fr"))
		eng_fr_file = read_file(eng_lang_file.format(lang="fr"))

		align_dataset(fr_file, eng_fr_file, it_file, eng_it_file, aligned_file)

		corpus = process_dataset(aligned_file, processed_file, plot_file)

	else:
		corpus = read_file_to_df(processed_file)

	vocab = create_vocab(corpus['french'].tolist(), corpus['italian'].tolist())

	corpus_4_model_training, corpus_4_testing = train_test_split(corpus, test_size=0.1)
	train_loader, val_loader, test_loader = prepare_dataset(corpus_4_model_training, model_config_file, vocab)

	if optimize:
		optimization(train_loader, val_loader, test_loader)
	else:
		config = read_json(model_config_file)
		train(config, train_loader, val_loader, model_file, plot_file, optimize=False)
