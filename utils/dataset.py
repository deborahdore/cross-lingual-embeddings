from typing import Tuple

import pandas as pd
from loguru import logger
from torch.utils.data import DataLoader, random_split

from dao.AEDataset import AEDataset


def split_dataset(dataset: AEDataset, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
	train_size = int(0.7 * len(dataset))
	val_size = int(0.10 * len(dataset))
	test_size = len(dataset) - train_size - val_size

	logger.info(f"[split_dataset] len train: {train_size} | len validation: {val_size} | len test: {test_size}")

	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True, shuffle = True)
	val_loader = DataLoader(val_dataset, batch_size = batch_size, drop_last = True, shuffle = False)
	test_loader = DataLoader(test_dataset, batch_size = batch_size, drop_last = True, shuffle = False)

	return train_loader, val_loader, test_loader


def prepare_dataset(corpus_4_model_training: pd.DataFrame, config: {}):
	dataset = AEDataset(
			corpus_fr = corpus_4_model_training['french'].tolist(),
			corpus_it = corpus_4_model_training['italian'].tolist()
			)

	train_loader, val_loader, test_loader = split_dataset(dataset, config.get("batch_size"))
	return train_loader, val_loader, test_loader
