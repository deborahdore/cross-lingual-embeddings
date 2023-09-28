import ast
import sys

from sklearn.model_selection import train_test_split

from config.path import aligned_file, processed_file, \
	vocab_file, model_config_file, plot_file, download_corpus, dataset_dir, lang_file, eng_lang_file, model_file, \
	embedding_model
from test import visualize_latent_space, test, translate
from train import train
from utils.processing import process_dataset, align_dataset
from utils.utils import read_file_to_df, download_from_url, read_file
from loguru import logger

if __name__ == '__main__':
	if len(sys.argv) < 2:
		logger.error("[main] missing arguments from command line")
		raise Exception("missing arguments from command line")

	create = ast.literal_eval(sys.argv[1])

	if create:

		# download_from_url(download_corpus.format(lang="it"), dataset_dir, "it")
		# download_from_url(download_corpus.format(lang="fr"), dataset_dir, "fr")

		it_file = read_file(lang_file.format(lang="it"))
		eng_it_file = read_file(eng_lang_file.format(lang="it"))

		fr_file = read_file(lang_file.format(lang="fr"))
		eng_fr_file = read_file(eng_lang_file.format(lang="fr"))

		align_dataset(fr_file, eng_fr_file, it_file, eng_it_file, aligned_file)

		corpus, vocab_fr, vocab_it = process_dataset(aligned_file,
													 processed_file,
													 vocab_file,
													 model_config_file,
													 plot_file,
													 embedding_model)

	else:
		corpus = read_file_to_df(processed_file)
		corpus['french'] = corpus['french'].apply(ast.literal_eval)
		corpus['italian'] = corpus['italian'].apply(ast.literal_eval)

		vocab_it = read_file_to_df(vocab_file.format(lang="it"))['words'].tolist()
		vocab_fr = read_file_to_df(vocab_file.format(lang="fr"))['words'].tolist()

	corpus_4_model_training, corpus_4_testing = train_test_split(corpus, test_size=0.1, random_state=42)
	corpus_4_model_training = corpus_4_model_training.reset_index(drop=True)
	corpus_4_testing = corpus_4_testing.reset_index(drop=True)

	test(train(corpus_4_model_training, model_config_file, model_file, plot_file), model_config_file, model_file)
	visualize_latent_space(corpus_4_testing, embedding_model, model_config_file, model_file, plot_file)
	translate(corpus_4_testing, vocab_it, vocab_fr, model_config_file, model_file)
