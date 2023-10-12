import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


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

		self.encoder = nn.GRU(embedding_dim,
							  hidden_dim,
							  num_layers=num_layers,
							  dropout=dropout,
							  bidirectional=False,
							  batch_first=True)

	def forward(self, x):
		lengths = torch.sum(x != 0, dim=1)
		embeds = self.embedder(x.long())
		packed_data = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
		output, hidden = self.encoder(packed_data)
		return hidden


class Decoder(nn.Module):
	def __init__(self,
				 vocab_dim: int,
				 embedding_dim: int,
				 hidden_dim: int,
				 output_dim: int,
				 num_layers: int = 1,
				 dropout: float = 0.0):
		super(Decoder, self).__init__()

		self.embedder = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim, padding_idx=0)
		self.decoder = nn.GRU(embedding_dim,
							  hidden_dim,
							  num_layers=num_layers,
							  batch_first=True,
							  dropout=dropout,
							  bidirectional=False)
		self.norm = nn.LayerNorm(hidden_dim)
		self.fc = nn.Linear(hidden_dim, vocab_dim)
		self.output_dim = output_dim

	def forward(self, x, hidden):
		embeds = self.embedder(x.long())
		output, _ = self.decoder(embeds, hidden)
		output = self.norm(output)
		output = self.fc(output)
		return output
