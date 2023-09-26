import random

import torch
from torch.utils.data import Dataset


class AEDataset(Dataset):
	def __init__(self, corpus_fr, corpus_it, negative_sampling: bool = True):
		self.french = corpus_fr
		self.italian = corpus_it

		self.negative_sampling = negative_sampling

	def __getitem__(self, index):
		sample_fr = torch.Tensor(self.french[index])

		if self.negative_sampling and random.random() < 0.5:
			return sample_fr, torch.Tensor(self.italian[random.randint(0, len(self.french) - 1)]), 0  # Negative
		else:
			return sample_fr, torch.Tensor(self.italian[index]), 1  # Positive

	def __len__(self):
		return len(self.french)
