import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
	def __init__(self, vocab_dim: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
		super(Encoder, self).__init__()

		self.embedder = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim, padding_idx=0)
		self.embedder.weight.requires_grad = True

		self.encoder = nn.LSTM(embedding_dim,
							   hidden_dim,
							   num_layers=num_layers,
							   dropout=dropout,
							   bidirectional=False,
							   batch_first=True)

		self.num_layers = num_layers
		self.hidden_dim = hidden_dim

	def init_hidden(self, batch_size, device):
		hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
		cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

		return hidden, cell

	def forward(self, x, hidden):
		lengths = torch.sum(x != 0, dim=1).cpu()
		embeds = self.embedder(x.long())
		packed_data = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
		output, (hidden, cell) = self.encoder(packed_data, hidden)
		return (hidden, cell)


class Decoder(nn.Module):
	def __init__(self,
				 vocab_dim: int,
				 embedding_dim: int,
				 hidden_dim: int,
				 hidden_dim2: int,
				 num_layers: int = 1,
				 dropout: float = 0.0):
		super(Decoder, self).__init__()

		self.embedder = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim, padding_idx=0)
		self.embedder.weight.requires_grad = True

		self.decoder = nn.LSTM(embedding_dim,
							   hidden_dim,
							   num_layers=num_layers,
							   batch_first=True,
							   dropout=dropout,
							   bidirectional=False)
		self.fc1 = nn.Linear(hidden_dim, hidden_dim2)
		self.fc2 = nn.Linear(hidden_dim2, vocab_dim)

		self.num_layers = num_layers
		self.hidden_dim = hidden_dim

	def detach_hidden(self, hidden):
		hidden, cell = hidden[0], hidden[1]
		return hidden.detach(), cell.detach()

	def forward(self, x, hidden):
		lengths = torch.sum(x != 0, dim=1).cpu()
		embeds = self.embedder(x.long())
		packed_data = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
		output_packed, hidden = self.decoder(packed_data, hidden)
		output, lens = pad_packed_sequence(output_packed, batch_first=True)
		output = self.fc1(output)
		output = F.relu(output)
		output = self.fc2(output)
		output = F.log_softmax(output, dim=-1)
		return output
