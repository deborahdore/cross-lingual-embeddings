import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.vocab import Vocab


class Encoder(nn.Module):
	def __init__(self,
				 vocab_dim: int,
				 embedding_dim: int,
				 hidden_dim: int,
				 num_layers: int = 1,
				 dropout: float = 0.0,
				 tune_embeddings: bool = True):
		super(Encoder, self).__init__()

		self.embedder = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim, padding_idx=0)
		self.embedder.weight.requires_grad = tune_embeddings

		self.encoder = nn.LSTM(embedding_dim,
							   hidden_dim,
							   num_layers=num_layers,
							   dropout=dropout,
							   bidirectional=False,
							   batch_first=True)

		self.num_layers = num_layers
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

	# self.init_weights()

	def init_weights(self):
		init_range_emb = 0.1
		init_range_other = 1 / math.sqrt(self.hidden_dim)
		self.embedder.weight.data.uniform_(-init_range_emb, init_range_emb)
		for i in range(self.num_layers):
			self.encoder.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
															   self.hidden_dim).uniform_(-init_range_other,
																						 init_range_other)
			self.encoder.all_weights[i][1] = torch.FloatTensor(self.hidden_dim,
															   self.hidden_dim).uniform_(-init_range_other,
																						 init_range_other)

	def init_hidden(self, batch_size, device):
		hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
		cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
		return hidden, cell

	def forward(self, x, hidden):
		lengths = torch.sum(x != 0, dim=1).cpu()
		embeds = self.embedder(x.long())
		packed_data = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
		output, hidden = self.encoder(packed_data, hidden)
		return hidden


class Decoder(nn.Module):
	def __init__(self,
				 vocab_dim: int,
				 embedding_dim: int,
				 hidden_dim: int,
				 hidden_dim2: int,
				 vocab: Vocab,
				 num_layers: int = 1,
				 dropout: float = 0.0):
		super(Decoder, self).__init__()

		self.embedder = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim, padding_idx=0)
		self.decoder = nn.LSTM(embedding_dim,
							   hidden_dim,
							   num_layers=num_layers,
							   batch_first=True,
							   dropout=dropout,
							   bidirectional=False)
		self.norm = nn.LayerNorm(hidden_dim)
		self.fc1 = nn.Linear(hidden_dim, hidden_dim2)
		self.fc2 = nn.Linear(hidden_dim2, vocab_dim)

		self.num_layers = num_layers
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

		# self.init_weights()
		self.vocab = vocab

	def detach_hidden(self, hidden):
		hidden, cell = hidden
		hidden = hidden.detach()
		cell = cell.detach()
		return hidden, cell

	def init_weights(self):
		init_range_emb = 0.1
		init_range_other = 1 / math.sqrt(self.hidden_dim)
		self.embedder.weight.data.uniform_(-init_range_emb, init_range_emb)
		self.fc.weight.data.uniform_(-init_range_other, init_range_other)
		self.fc.bias.data.zero_()
		for i in range(self.num_layers):
			self.decoder.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
															   self.hidden_dim).uniform_(-init_range_other,
																						 init_range_other)
			self.decoder.all_weights[i][1] = torch.FloatTensor(self.hidden_dim,
															   self.hidden_dim).uniform_(-init_range_other,
																						 init_range_other)

	def init_hidden(self, batch_size, device):
		hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
		cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
		return hidden, cell

	def forward(self, x, hidden, teacher_forcing: bool = True):
		if teacher_forcing:
			output, hidden = self.one_step_decoder(x, hidden)
			return output, hidden
		else:  # autoregression
			seq_length = x.shape[1]
			input = torch.Tensor([self.vocab['<sos>']]).expand(x.shape[0], 1).to(x.device)
			result = torch.Tensor([]).to(x.device)
			for i in range(seq_length):
				zero_rows_indices = np.where(input.cpu() == 0)[0]
				if zero_rows_indices.shape[0] == x.shape[0]:
					# everything is zero
					return torch.nn.functional.pad(result, (0, 0, x.shape[1] - result.shape[1], 0, 0, 0)), hidden
				input = np.delete(input.cpu().detach(), zero_rows_indices, axis=0).to(x.device)
				output, hidden = self.one_step_decoder(input, hidden)
				for j in range(zero_rows_indices.shape[0]):
					output = np.insert(output.cpu().detach(),
									   zero_rows_indices[j],
									   torch.zeros((1, 1, output.shape[-1]), device='cpu'),
									   axis=0).to(x.device)

				input = torch.multinomial(F.softmax(output, dim=-1).squeeze(),
										  num_samples=1,
										  replacement=True).to(x.device)
				result = torch.cat((result, output), dim=1).to(x.device)

			return result, hidden

	def one_step_decoder(self, x, hidden):
		lengths = torch.sum(x != 0, dim=1).cpu()
		embeds = self.embedder(x.long())
		packed_data = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
		output_packed, hidden = self.decoder(packed_data, hidden)
		output, lens = pad_packed_sequence(output_packed, batch_first=True)
		output = self.norm(output)
		output = self.fc1(output)
		output = self.fc2(output)
		return output, hidden
