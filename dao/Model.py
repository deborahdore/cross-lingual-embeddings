import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
	def __init__(self,
				 vocab_dim: int,
				 embedding_dim: int,
				 hidden_dim: int,
				 hidden_dim2: int,
				 num_layers: int = 1,
				 dropout: float = 0.0,
				 bidirectional: bool = True):
		super(Encoder, self).__init__()
		MAX_LEN = 100

		self.embedder = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim, padding_idx=0)
		self.encoder = nn.LSTM(embedding_dim,
							   hidden_dim,
							   num_layers=num_layers,
							   dropout=dropout,
							   bidirectional=True,
							   batch_first=True)
		self.fc = nn.Linear(MAX_LEN * (hidden_dim * 2), hidden_dim2)

		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.bidirectional = bidirectional

	def init_hidden(self, batch_size, device):
		bidirectional = 2 if self.bidirectional == True else 1
		hidden = torch.randn(self.num_layers * bidirectional, batch_size, self.hidden_dim).to(device)
		cell = torch.randn(self.num_layers * bidirectional, batch_size, self.hidden_dim).to(device)

		return hidden, cell

	def detach_hidden(self, hidden):
		return hidden[0].detach(), hidden[1].detach()

	def forward(self, x, hidden):
		embeds = self.embedder(x.long())
		output, hidden = self.encoder(embeds, hidden)
		output = self.fc(output.reshape(x.size(0), -1))
		output = F.relu(output)
		return output, hidden


class Decoder(nn.Module):
	def __init__(self,
				 vocab_dim: int,
				 embedding_dim: int,
				 hidden_dim: int,
				 hidden_dim2: int,
				 num_layers: int = 1,
				 dropout: float = 0.0,
				 bidirectional: bool = True):
		super(Decoder, self).__init__()

		self.decoder = nn.LSTM(hidden_dim2,
							   hidden_dim,
							   num_layers=num_layers,
							   batch_first=True,
							   dropout=dropout,
							   bidirectional=True)
		self.fc = nn.Linear(hidden_dim * 2, vocab_dim)

		self.bidirectional = bidirectional
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.hidden_dim2 = hidden_dim2
		self.max_len = 100
		self.vocab_dim = vocab_dim

	def init_hidden(self, batch_size, device):
		bidirectional = 2 if self.bidirectional == True else 1
		hidden = torch.randn(self.num_layers * bidirectional, batch_size, self.hidden_dim).to(device)
		cell = torch.randn(self.num_layers * bidirectional, batch_size, self.hidden_dim).to(device)

		return hidden, cell

	def forward(self, x, hidden):
		decoded_sentence = []

		decoder_hidden = hidden

		for _ in range(self.max_len):
			decoder_output, decoder_hidden = self.decoder(x.unsqueeze(1), decoder_hidden)
			decoded_sentence.append(decoder_output)

		output = torch.stack(decoded_sentence, dim=-1).squeeze(1)
		output = self.fc(output.permute(0, 2, 1))
		return F.log_softmax(output, dim=-1)

	def one_step_decoder(self, x, hidden):
		embeds = self.embedder(x.long())
		output, hidden = self.decoder(embeds, hidden)
		output = self.fc(output)
		output = F.log_softmax(output, dim=-1)
		return output, hidden
