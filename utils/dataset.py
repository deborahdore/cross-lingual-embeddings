from typing import Tuple

import pandas as pd
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from dao.Dataset import LSTMDataset
from dao.Vocab import Vocab


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


def collate_fn(batch):
	batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
	fr, it, label = zip(*batch)
	# Pad sequences to the length of the longest sequence in the batch
	padded_sequences_fr = pad_sequence(fr, batch_first=True, padding_value=0)
	padded_sequences_it = pad_sequence(it, batch_first=True, padding_value=0)

	return torch.Tensor(padded_sequences_fr), torch.Tensor(padded_sequences_it), torch.Tensor(label)


def prepare_dataset(corpus_4_model_training: pd.DataFrame, config: {}, vocab_it: Vocab, vocab_fr: Vocab):
	dataset = LSTMDataset(corpus_fr=corpus_4_model_training['french'].tolist(),
						  corpus_it=corpus_4_model_training['italian'].tolist(),
						  vocab_it=vocab_it,
						  vocab_fr=vocab_fr)

	train_loader, val_loader, test_loader = split_dataset(dataset, config.get("batch_size"))
	return train_loader, val_loader, test_loader
