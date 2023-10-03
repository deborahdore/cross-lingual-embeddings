import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence




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
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
		self.fc = nn.Linear(hidden_dim, proj_dim)

	def forward(self, x):
		# input shape (batch_size, seq_len)
		x = self.embedder(x.long()) # (batch_size, seq_len, emb_dim)
		x, (hn, cn) = self.lstm(x)
		out = self.fc(hn[-1])
		return out, (hn, cn)


class Decoder(nn.Module):
	def __init__(self,
				 vocab_size: int,
				 embedding_dim: int,
				 hidden_dim: int,
				 output_dim: int,
				 num_layers: int = 1,
				 dropout: float = 0.0):
		super(Decoder, self).__init__()

		self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.output_dim = output_dim

	def forward(self, x, hidden, target_tensor, MAX_LENGTH):
		batch_size = x.size(0)
		SOS_token = 1
		EOS_token = 2
		decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_token)
		decoder_hidden = hidden
		decoder_outputs = []

		for i in range(MAX_LENGTH):
			decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
			decoder_outputs.append(decoder_output)

			if target_tensor is not None:
				# Teacher forcing: Feed the target as the next input
				decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
			else:
				# Without teacher forcing: use its own predictions as the next input
				_, topi = decoder_output.topk(1)
				decoder_input = topi.squeeze(-1).detach()  # detach from history as input

		decoder_outputs = torch.stack(decoder_outputs, dim=1)
		decoder_outputs = F.log_softmax(decoder_outputs.squeeze(-1), dim=-1)
		return decoder_outputs

	def forward_step(self, x, hidden):
		output = self.embedding(x.long())
		output = F.relu(output)
		output, hidden = self.lstm(output, hidden)
		output = self.fc(output.squeeze(1))
		return output, hidden
