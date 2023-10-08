import io
from typing import Tuple

import nltk
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import Vocab, build_vocab_from_iterator

from dao.Dataset import LSTMDataset
from utils.utils import read_json, write_json


def split_dataset(dataset: LSTMDataset, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
	train_size = int(0.7 * len(dataset))
	val_size = int(0.10 * len(dataset))
	test_size = len(dataset) - train_size - val_size

	logger.info(f"[split_dataset] len train: {train_size} | len validation: {val_size} | len test: {test_size}")

	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

	return train_loader, val_loader, test_loader


def prepare_dataset(corpus: pd.DataFrame, model_config_file: str, vocab: Vocab):
	logger.info("[prepare_dataset] preparing dataset")
	# creating dataset model
	dataset = LSTMDataset(corpus_fr=corpus['french'], corpus_it=corpus['italian'])

	# modify configuration
	config = read_json(model_config_file)

	train_loader, val_loader, test_loader = split_dataset(dataset, config.get("batch_size"))

	config['len_vocab'] = len(vocab)
	config['output_dim_it'] = dataset.get_max_len_it()
	config['output_dim_fr'] = dataset.get_max_len_fr()

	write_json(config, model_config_file)

	return train_loader, val_loader, test_loader


def create_vocab(corpus):
	# generating vocab from text file
	def yield_tokens(corpus):
			for sentence in corpus:
				yield sentence.split()

	# create same vocabulary for both languages
	logger.info("[create_vocab] creating vocabulary shared between languages")

	corpus = pd.concat([corpus['french'], corpus['italian']], axis=0)
	vocab = build_vocab_from_iterator(yield_tokens(corpus), specials=['<PAD>', '<UNK>'], max_tokens=20000)
	vocab.set_default_index(vocab['<UNK>'])
	return vocab

def sequence2index(corpus: pd.DataFrame, vocab: Vocab):
	fr_tokenized = [torch.Tensor(vocab.lookup_indices(sentence.split())) for sentence in corpus['french'].values]
	it_tokenized = [torch.Tensor(vocab.lookup_indices(sentence.split())) for sentence in corpus['italian'].values]

	return pd.DataFrame({'french': fr_tokenized, 'italian': it_tokenized})
