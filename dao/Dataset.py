import random

import torch
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
	def __init__(self, corpus_fr, corpus_it, vocab_fr, vocab_it, negative_sampling: bool = True):
		self.french = corpus_fr
		self.italian = corpus_it
		self.vocab_it = vocab_it
		self.vocab_fr = vocab_fr
		self.negative_sampling = negative_sampling

	def __getitem__(self, index):
		sample_fr = [self.vocab_fr.word2idx.get(word) for word in self.french[index].split()]

		if self.negative_sampling and random.random() < 0.5:
			new_index = random.randint(0, len(self.italian) - 1)
			sample_it = [self.vocab_it.word2idx.get(word) for word in self.italian[new_index].split()]
			return torch.Tensor(sample_fr), torch.Tensor(sample_it), 0  # Negative
		else:
			sample_it = [self.vocab_it.word2idx.get(word) for word in self.italian[index].split()]
			return torch.Tensor(sample_fr), torch.Tensor(sample_it), 1  # Positive

	def __len__(self):
		return len(self.french)
