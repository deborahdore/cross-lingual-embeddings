import random

import nltk
import torch
from torch.nn.utils import rnn
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
	def __init__(self, corpus_fr, corpus_it, vocab, negative_sampling: bool = True):
		self.french = []
		self.italian = []
		self.vocab = vocab
		self.negative_sampling = negative_sampling

		self.init(corpus_fr, corpus_it)

	def __getitem__(self, index):
		sample_fr = self.french[index]

		if self.negative_sampling and random.random() < 0.5:
			new_index = random.randint(0, len(self.italian) - 1)
			return sample_fr, self.italian[new_index], 0  # Negative

		return sample_fr, self.italian[index], 1  # Positive

	def __len__(self):
		return len(self.french)

	def get_max_len_it(self):
		return len(max(self.italian, key=len))

	def get_max_len_fr(self):
		return len(max(self.french, key=len))

	def tokenize_and_convert(self, sentence):
		tokens = nltk.word_tokenize(sentence)

		indices = ([self.vocab.to_index('<SOS>')] + [self.vocab.to_index(word) for word in tokens] + [
			self.vocab.to_index('<EOS>')])
		return torch.Tensor(indices)

	def init(self, corpus_fr, corpus_it):
		fr_tokenized = corpus_fr.apply(self.tokenize_and_convert).tolist()
		it_tokenized = corpus_it.apply(self.tokenize_and_convert).tolist()

		self.french = rnn.pad_sequence(fr_tokenized, batch_first=True, padding_value=self.vocab.to_index('<PAD>'))
		self.italian = rnn.pad_sequence(it_tokenized, batch_first=True, padding_value=self.vocab.to_index('<PAD>'))
