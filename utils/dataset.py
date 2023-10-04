from typing import Tuple

import pandas as pd
from loguru import logger
from torch.utils.data import DataLoader, random_split

from dao.Dataset import LSTMDataset
from dao.Vocab import Vocab
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
	dataset = LSTMDataset(corpus_fr=corpus['french'], corpus_it=corpus['italian'], vocab=vocab)

	config = read_json(model_config_file)

	train_loader, val_loader, test_loader = split_dataset(dataset, config.get("batch_size"))

	config['len_vocab'] = vocab.get_vocab_len()
	config['output_dim_it'] = dataset.get_max_len_it()
	config['output_dim_fr'] = dataset.get_max_len_fr()

	write_json(config, model_config_file)

	return train_loader, val_loader, test_loader


def create_vocab(french: [], italian: []):
	# Same vocabulary for both languages
	logger.info("[create_vocab] creating vocabulary shared between languages")
	vocab = Vocab()
	for sentence in french:
		vocab.add_sentence(sentence, "french")
	for sentence in italian:
		vocab.add_sentence(sentence, "italian")
	return vocab
