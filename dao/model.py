import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LatentSpace(nn.Module):
	def __init__(self, hidden_dim: int, ls_dim: int):
		super(LatentSpace, self).__init__()

		self.latent_space = nn.Linear(hidden_dim, ls_dim)

	def forward(self, x):
		x = self.latent_space(x)
		return F.leaky_relu(x)


class Encoder(nn.Module):
	def __init__(self, vocab_dim: int, embedding_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
		super(Encoder, self).__init__()
		self.embedder = nn.Embedding(num_embeddings = vocab_dim, embedding_dim = embedding_dim)
		self.lstm = nn.LSTM(
				embedding_dim, hidden_dim, num_layers = num_layers, batch_first = True, dropout = dropout
				)

	def forward(self, x):
		seq_lengths = torch.Tensor([sum(1 for num in sublist if num != 0) for sublist in x])
		x = self.embedder(x.long())
		packed_x = pack_padded_sequence(x, seq_lengths, batch_first = True, enforce_sorted = False)
		x, (hn, cn) = self.lstm(packed_x)
		return hn[-1]


class Decoder(nn.Module):
	def __init__(self, ls_dim: int, hidden_lstm_dim: int, output_dim: int, num_layers: int = 1, dropout: float = 0.2):
		super(Decoder, self).__init__()
		self.lstm = nn.LSTM(ls_dim, hidden_lstm_dim, num_layers = num_layers, dropout = dropout, batch_first = True)
		self.norm = nn.LayerNorm(hidden_lstm_dim)
		self.fc = nn.Linear(hidden_lstm_dim, ls_dim)

		self.max_length = output_dim

	def forward(self, x, target_input):
		# x.shape (batch_size, 16)
		hidden = None
		output_sequence = []

		for t in range(self.max_length):
			x, hidden = self.lstm(x, hidden)
			x = self.norm(x)
			x = self.fc(x)

			# Append the last output of the LSTM output cell
			output_sequence.append(x[:, -1])

			# Use the generated character as input for the next time step

		# Stack the generated outputs to create the final output sequence
		final_output = torch.stack(output_sequence, dim = 1)

		return final_output


