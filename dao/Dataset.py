import random

import torch
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
	def __init__(self, corpus_fr, corpus_it, negative_sampling: bool = True):
		self.french = corpus_fr
		self.italian = corpus_it
		self.negative_sampling = negative_sampling

	def __getitem__(self, index):
		sample_fr = self.french[index]

		if self.negative_sampling and random.random() < 0.5:
			new_index = random.randint(0, len(self.italian) - 1)
			return torch.Tensor(sample_fr), torch.Tensor(self.italian[new_index]), 0  # Negative

		return torch.Tensor(sample_fr), torch.Tensor(self.italian[index]), 1  # Positive

	def __len__(self):
		return len(self.french)

	def get_max_len_it(self):
		return len(max(self.italian, key=len))

	def get_max_len_fr(self):
		return len(max(self.french, key=len))
