import os.path

import pandas as pd
import spacy
import torchtext
from loguru import logger
from torchtext.vocab import Vocab

from config.path import dataset_dir, download_corpus, eng_lang_file, lang_file
from train import train_autoencoder
from utils.dataset import sequence2index
from utils.processing import nlp_pipeline
from utils.utils import download_from_url, read_file, read_json

spacy.load('de_core_news_sm')
spacy.load('en_core_web_sm')


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
	study_result_dir = os.path.join(study_result_dir, "dropout")
	train_autoencoder(config_without_dropout,
					  corpus_4_model_training,
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
	study_result_dir = os.path.join(study_result_dir, "num_layers")
	train_autoencoder(config_one_layer,
					  corpus_4_model_training,
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
	study_result_dir = os.path.join(study_result_dir, "cl_loss")
	train_autoencoder(config_zero_alpha,
					  corpus_4_model_training,
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
	study_result_dir = os.path.join(study_result_dir, "reconstruction_loss")
	train_autoencoder(config_zero_beta,
					  corpus_4_model_training,
					  vocab_fr,
					  vocab_it,
					  model_file,
					  plot_file,
					  study_result_dir,
					  optimize=False,
					  ablation_study=True)

	# ------- test on english/german dataset ------- #
	logger.info("[ablation_study_eval] training with english/german")
	study_result_dir = os.path.join(study_result_dir, "english_german")
	test_english_german(config, model_file, plot_file, study_result_dir)

	# ------- test on english/french dataset ------- #
	logger.info("[ablation_study_eval] training with english/french")
	study_result_dir = os.path.join(study_result_dir, "english_french")
	test_english_french(config, model_file, plot_file, study_result_dir)


def test_english_french(config, model_file, plot_file, study_result_dir):
	fr_file = read_file(lang_file.format(lang="fr"))
	eng_fr_file = read_file(eng_lang_file.format(lang="fr"))

	corpus = pd.DataFrame(data={'french': fr_file.split("\n"), 'english': eng_fr_file.split("\n")})

	corpus['french'] = nlp_pipeline(corpus['french'])
	corpus['english'] = nlp_pipeline(corpus['english'])

	tokenizer_fr = torchtext.data.utils.get_tokenizer('spacy', language="fr_core_news_sm")
	tokenizer_en = torchtext.data.utils.get_tokenizer('spacy', language="en_core_web_sm")

	tokenized_dataset_fr = corpus['french'].apply(lambda x: tokenizer_fr(x))
	tokenized_dataset_en = corpus['english'].apply(lambda x: tokenizer_en(x))

	vocab_fr = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset_fr, max_tokens=30000)
	vocab_fr.insert_token('<pad>', 0)
	vocab_fr.insert_token('<eos>', 1)
	vocab_fr.insert_token('<unk>', 2)
	vocab_fr.set_default_index(vocab_fr['<unk>'])
	vocab_en = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset_en, max_tokens=30000)
	vocab_en.insert_token('<pad>', 0)
	vocab_en.insert_token('<eos>', 1)
	vocab_en.insert_token('<unk>', 2)
	vocab_en.set_default_index(vocab_en['<unk>'])

	tokenized_dataset_en = tokenized_dataset_en.apply(lambda x: x + ["<eos>"])
	tokenized_dataset_fr = tokenized_dataset_fr.apply(lambda x: x + ["<eos>"])

	corpus_tokenized = pd.DataFrame({
		'english': sequence2index(tokenized_dataset_en, vocab_en),
		'french' : sequence2index(tokenized_dataset_fr, vocab_fr)})

	corpus_tokenized = corpus_tokenized[
		corpus_tokenized.apply(lambda row: len(row['french']) <= 100 and len(row['english']) <= 100 and len(
			row['french']) > 10 and len(row['english']) > 10, axis=1)]

	corpus_tokenized['french'] = corpus_tokenized['french'].apply(lambda x: x[:100] + [0] * (100 - len(x)))
	corpus_tokenized['english'] = corpus_tokenized['english'].apply(lambda x: x[:100] + [0] * (100 - len(x)))

	config_en_fr = config
	train_autoencoder(config_en_fr, corpus_tokenized, vocab_fr,  # vocab fr
					  vocab_en,  # vocab it
					  model_file, plot_file, study_result_dir, optimize=False, ablation_study=True)


def test_english_german(config, model_file, plot_file, study_result_dir):
	if not os.path.exists(download_corpus.format(lang="de")):
		download_from_url(download_corpus.format(lang="de"), dataset_dir, "de")

	de_file = read_file(lang_file.format(lang="de"))
	eng_de_file = read_file(eng_lang_file.format(lang="de"))

	corpus = pd.DataFrame(data={'german': de_file.split("\n"), 'english': eng_de_file.split("\n")})

	corpus['german'] = nlp_pipeline(corpus['german'])
	corpus['english'] = nlp_pipeline(corpus['english'])

	tokenizer_de = torchtext.data.utils.get_tokenizer('spacy', language="de_core_news_sm")
	tokenizer_en = torchtext.data.utils.get_tokenizer('spacy', language="en_core_web_sm")

	tokenized_dataset_de = corpus['german'].apply(lambda x: tokenizer_de(x))
	tokenized_dataset_en = corpus['english'].apply(lambda x: tokenizer_en(x))

	vocab_de = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset_de, max_tokens=30000)
	vocab_de.insert_token('<pad>', 0)
	vocab_de.insert_token('<eos>', 1)
	vocab_de.insert_token('<unk>', 2)
	vocab_de.set_default_index(vocab_de['<unk>'])

	vocab_en = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset_en, max_tokens=30000)
	vocab_en.insert_token('<pad>', 0)
	vocab_en.insert_token('<eos>', 1)
	vocab_en.insert_token('<unk>', 2)
	vocab_en.set_default_index(vocab_en['<unk>'])

	tokenized_dataset_en = tokenized_dataset_en.apply(lambda x: x + ["<eos>"])
	tokenized_dataset_de = tokenized_dataset_de.apply(lambda x: x + ["<eos>"])

	corpus_tokenized = pd.DataFrame({
		'english': sequence2index(tokenized_dataset_en, vocab_en),
		'german' : sequence2index(tokenized_dataset_de, vocab_de)})

	corpus_tokenized = corpus_tokenized[
		corpus_tokenized.apply(lambda row: len(row['german']) <= 100 and len(row['english']) <= 100 and len(
			row['german']) > 10 and len(row['english']) > 10, axis=1)]

	corpus_tokenized['german'] = corpus_tokenized['german'].apply(lambda x: x[:100] + [0] * (100 - len(x)))
	corpus_tokenized['english'] = corpus_tokenized['english'].apply(lambda x: x[:100] + [0] * (100 - len(x)))

	config_en_de = config
	train_autoencoder(config_en_de, corpus_tokenized, vocab_de,  # vocab fr
					  vocab_en,  # vocab it
					  model_file, plot_file, study_result_dir, optimize=False, ablation_study=True)
