import random

import torch
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
	def __init__(self, corpus_fr, corpus_it, negative_sampling: bool = True):
		self.french = corpus_fr
		self.italian = corpus_it

		self.negative_sampling = negative_sampling

	def __getitem__(self, index):
		sample_fr = torch.tensor(self.french[index])

		if self.negative_sampling and random.random() < 0.5:
			new_index = random.randint(0, len(self.italian) - 1)
			# Negative
			return sample_fr, torch.tensor(self.italian[new_index]), torch.tensor(0)

		# Positive
		else:
			return sample_fr, torch.tensor(self.italian[index]), torch.tensor(1)

	def __len__(self):
		return len(self.french)
