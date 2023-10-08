import random

import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self,
				 vocab_dim: int,
				 embedding_dim: int,
				 hidden_dim: int,
				 proj_dim: int,
				 num_layers: int = 1,
				 dropout: float = 0.0):
		super(Encoder, self).__init__()

		self.embedder = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim, padding_idx=0)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
		self.fc = nn.Linear(100 * 64, proj_dim)

		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

		init.xavier_uniform_(self.lstm.weight_ih_l0)  # Initialize LSTM input weights
		init.xavier_uniform_(self.lstm.weight_hh_l0)  # Initialize LSTM hidden weights

	def forward(self, x):
		lengths = torch.sum(x != 0, dim=1)
		embedding = self.embedder(x.long())
		packed_data = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
		out_packed, hidden = self.lstm(packed_data)
		out, lens = pad_packed_sequence(out_packed, batch_first=True)
		desidered_pad = 100 - out.shape[1]
		out = F.pad(out, (0, 0, 0, desidered_pad, 0, 0), value=0)
		out = self.fc(out.reshape(out.size(0), -1))
		return out, hidden


class Decoder(nn.Module):
	def __init__(self, proj_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1, dropout: float = 0.0):
		super(Decoder, self).__init__()

		self.lstm = nn.LSTM(1, proj_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
		self.fc1 = nn.Linear(proj_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)

		self.proj_dim = proj_dim
		self.output_dim = output_dim
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim

		init.xavier_uniform_(self.lstm.weight_ih_l0)  # Initialize LSTM input weights
		init.xavier_uniform_(self.lstm.weight_hh_l0)  # Initialize LSTM hidden weights

	def forward(self, x, hidden, trg, teacher_forcing_ratio=0.5):
		batch_size = x.size(0)
		trg_len = x.size(1)

		outputs = torch.zeros(batch_size, trg_len, self.output_dim)  # Swap batch_size and trg_len dimensions
		hidden = None
		input = trg[:, 0]
		for t in range(1, trg_len):
			output, hidden = self.lstm(input.view(batch_size, 1, -1).float(), hidden)  # Add view to reshape input
			output = self.fc1(output)
			output = F.relu(output)
			output = self.fc2(output)
			outputs[:, t] = output.squeeze(1)  # Squeeze the output to match the dimensions

			# Determine the next input based on teacher forcing ratio
			teacher_force = random.random() < teacher_forcing_ratio
			input = trg[:, t] if teacher_force else output.squeeze(1).argmax(1)  # Change input based on teacher forcing

		return outputs.permute(1, 0, 2)[-1, : , :]  # Swap dimensions back to original order
# def forward(self, x, hidden):
	# 	seq_len = x.size(1)
	# 	outputs = []
	# 	hidden=None
	# 	input = x
	# 	for t in range(seq_len):
	# 		# Pass the input through the LSTM
	# 		out, hidden = self.lstm(input, hidden)
	# 		outputs.append(out[:, -1])
	# 		input = out
	#
	# 	# Stack the outputs along the sequence dimension
	# 	outputs = torch.stack(outputs, dim=1)
	# 	outputs = self.fc1(outputs)
	# 	outputs = F.relu(outputs)
	# 	outputs = self.fc2(outputs)
	# 	return outputs

