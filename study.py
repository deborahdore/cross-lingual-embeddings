import pandas as pd
import torchtext
from loguru import logger
from torchtext.vocab import Vocab

from config.path import eng_lang_file, lang_file
from train import train_autoencoder
from utils.dataset import sequence2index
from utils.processing import nlp_pipeline
from utils.utils import read_file, read_json


def ablation_study(corpus_4_model_training: pd.DataFrame(),
				   model_config_file: str,
				   vocab: Vocab,
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
					  vocab,
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
					  vocab,
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
					  vocab,
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
					  vocab,
					  model_file,
					  plot_file,
					  study_result_dir,
					  optimize=False,
					  ablation_study=True)

	# ------- test on english/german dataset ------- #
	logger.info("[ablation_study_eval] training with english/german")
	# test_english_german(config, model_file, plot_file, study_result_dir)

	# ------- test on english/french dataset ------- #
	logger.info("[ablation_study_eval] training with english/french")


# test_english_french(config, model_file, plot_file, study_result_dir)


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

	vocab_fr = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset_fr, max_tokens=40000)
	vocab_fr.insert_token('<pad>', 0)
	vocab_fr.insert_token('<eos>', 1)
	vocab_fr.insert_token('<unk>', 2)
	vocab_fr.set_default_index(vocab_fr['<unk>'])
	vocab_en = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset_en, max_tokens=40000)
	vocab_en.insert_token('<pad>', 0)
	vocab_en.insert_token('<eos>', 1)
	vocab_en.insert_token('<unk>', 2)
	vocab_en.set_default_index(vocab_en['<unk>'])

	corpus_tokenized = pd.DataFrame({
		'french' : sequence2index(tokenized_dataset_fr.apply(lambda x: x + ["<eos>"]), vocab_fr),
		'english': sequence2index(tokenized_dataset_en.apply(lambda x: x + ["<eos>"]), vocab_en)})

	corpus_tokenized = corpus_tokenized.dropna().drop_duplicates().reset_index(drop=True)

	config_en_fr = config
	train_autoencoder(config_en_fr,
					  corpus_tokenized,
					  vocab_fr,
					  vocab_en,
					  model_file,
					  plot_file,
					  study_result_dir,
					  optimize=False,
					  ablation_study=True)


def test_english_german(config, model_file, plot_file, study_result_dir):
	de_file = read_file(lang_file.format(lang="de"))
	eng_de_file = read_file(eng_lang_file.format(lang="de"))

	corpus = pd.DataFrame(data={'german': de_file.split("\n"), 'english': eng_de_file.split("\n")})

	corpus['german'] = nlp_pipeline(corpus['german'])
	corpus['english'] = nlp_pipeline(corpus['english'])

	tokenizer_de = torchtext.data.utils.get_tokenizer('spacy', language="de_core_news_sm")
	tokenizer_en = torchtext.data.utils.get_tokenizer('spacy', language="en_core_web_sm")

	tokenized_dataset_de = corpus['german'].apply(lambda x: tokenizer_de(x))
	tokenized_dataset_en = corpus['english'].apply(lambda x: tokenizer_en(x))

	vocab_de = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset_de, max_tokens=40000)
	vocab_de.insert_token('<pad>', 0)
	vocab_de.insert_token('<eos>', 1)
	vocab_de.insert_token('<unk>', 2)
	vocab_de.set_default_index(vocab_de['<unk>'])
	vocab_en = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset_en, max_tokens=40000)
	vocab_en.insert_token('<pad>', 0)
	vocab_en.insert_token('<eos>', 1)
	vocab_en.insert_token('<unk>', 2)
	vocab_en.set_default_index(vocab_en['<unk>'])

	corpus_tokenized = pd.DataFrame({
		'german' : sequence2index(tokenized_dataset_de.apply(lambda x: x + ["<eos>"]), vocab_de),
		'english': sequence2index(tokenized_dataset_en.apply(lambda x: x + ["<eos>"]), vocab_en)})

	corpus_tokenized = corpus_tokenized.dropna().drop_duplicates().reset_index(drop=True)

	config_en_de = config

	train_autoencoder(config_en_de,
					  corpus_tokenized,
					  vocab_de,
					  vocab_en,
					  model_file,
					  plot_file,
					  study_result_dir,
					  optimize=False,
					  ablation_study=True)
