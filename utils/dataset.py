from typing import Tuple

import pandas as pd
import spacy
import torchtext
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import Vocab

from dao.Dataset import LSTMDataset
from utils.utils import write_json

spacy.load('fr_core_news_sm')
spacy.load('it_core_news_sm')


def collate_fn(batch):
	# Sort the batch in descending order of the length of French sentences
	batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

	# Separate the French sentences, Italian sentences, and labels
	french_sentences, italian_sentences, labels = zip(*batch)

	# Pad French and Italian sentences with zeros to the length of the longest sentence in the batch
	padded_french_sentences = pad_sequence(french_sentences, batch_first=True, padding_value=0)
	padded_italian_sentences = pad_sequence(italian_sentences, batch_first=True, padding_value=0)

	return padded_french_sentences, padded_italian_sentences, labels


def split_dataset(dataset: LSTMDataset, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
	train_size = int(0.7 * len(dataset))
	val_size = int(0.10 * len(dataset))
	test_size = len(dataset) - train_size - val_size

	logger.info(f"[split_dataset] len train: {train_size} | len validation: {val_size} | len test: {test_size}")

	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
							  collate_fn=collate_fn)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False, collate_fn=collate_fn)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, collate_fn=collate_fn)

	return train_loader, val_loader, test_loader


def prepare_dataset(corpus: pd.DataFrame, model_config_file: str, vocab_fr: Vocab, vocab_it: Vocab, config: dict):
	logger.info("[prepare_dataset] preparing dataset")
	# creating dataset model
	dataset = LSTMDataset(corpus_fr=corpus['french'], corpus_it=corpus['italian'])

	# modify configuration

	train_loader, val_loader, test_loader = split_dataset(dataset, config.get("batch_size"))

	config['len_vocab_it'] = len(vocab_it)
	config['len_vocab_fr'] = len(vocab_fr)

	# todo: remove?
	config['output_dim_it'] = dataset.get_max_len_it()
	config['output_dim_fr'] = dataset.get_max_len_fr()

	write_json(config, model_config_file)

	return train_loader, val_loader, test_loader


def create_vocab(corpus: pd.DataFrame, language: str):
	# generating vocab from text file
	logger.info("[create_vocab] creating vocabulary")
	tokenizer = torchtext.data.utils.get_tokenizer('spacy', language=language)

	tokenized_dataset = corpus.apply(lambda x: tokenizer(x))

	vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset, min_freq=3, max_tokens=30000)
	vocab.insert_token('<pad>', 0)
	vocab.insert_token('<sos>', 1)
	vocab.insert_token('<eos>', 2)
	vocab.insert_token('<unk>', 3)

	vocab.set_default_index(vocab['<unk>'])

	tokenized_dataset = tokenized_dataset.apply(lambda x: x + ["<eos>"])

	return tokenized_dataset, vocab


def sequence2index(corpus: pd.DataFrame, vocab: Vocab):  # end of sentence
	encoded = []
	for sentence in corpus:
		encoded.append([vocab[token] for token in sentence])
	return encoded
