from typing import Tuple

import pandas as pd
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import Vocab, build_vocab_from_iterator

from dao.Dataset import LSTMDataset
from utils.utils import read_json, write_json


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


def prepare_dataset(corpus: pd.DataFrame, model_config_file: str, vocab_fr: Vocab, vocab_it: Vocab):
	logger.info("[prepare_dataset] preparing dataset")
	# creating dataset model
	dataset = LSTMDataset(corpus_fr=corpus['french'], corpus_it=corpus['italian'])

	# modify configuration
	config = read_json(model_config_file)

	train_loader, val_loader, test_loader = split_dataset(dataset, config.get("batch_size"))

	config['len_vocab_it'] = len(vocab_it)
	config['len_vocab_fr'] = len(vocab_fr)

	# todo: remove?
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
	logger.info("[create_vocab] creating vocabulary")

	vocab = build_vocab_from_iterator(yield_tokens(corpus),
									  specials=['<PAD>', '<UNK>', '<SOS>', '<EOS>'],
									  max_tokens=20000)
	vocab.set_default_index(vocab['<UNK>'])
	return vocab


def sequence2index(corpus: pd.DataFrame, vocab: Vocab):
	SOS = vocab.lookup_indices(['<SOS>'])  # start of sentence
	EOS = vocab.lookup_indices(['<EOS>'])  # end of sentence
	tokenized = [vocab.lookup_indices(sentence.split()) for sentence in corpus.values]
	complete = [SOS + sentence + EOS for sentence in tokenized]
	return complete
