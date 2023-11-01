import random

import torch


class LSTMDataset(torch.utils.data.Dataset):
	def __init__(self, corpus_fr, corpus_it, negative_sampling: bool = True, max_len: int = 100):
		self.french = corpus_fr
		self.italian = corpus_it
		self.negative_sampling = negative_sampling
		self.max_len = max_len

	def __getitem__(self, index):
		sample_fr = self.french[index]

		if self.negative_sampling and random.random() < 0.5:
			new_index = random.randint(0, len(self.italian) - 1)
			while new_index == index:
				new_index = random.randint(0, len(self.italian) - 1)
			return self.pad_and_to_tensor_elem(sample_fr), self.pad_and_to_tensor_elem(self.italian[new_index]), 0

		return self.pad_and_to_tensor_elem(sample_fr), self.pad_and_to_tensor_elem(self.italian[index]), 1  # Positive

	def __len__(self):
		return len(self.french)

	def pad_and_to_tensor_elem(self, x):
		x = x + [0] * (self.max_len - len(x))
		return torch.Tensor(x)
