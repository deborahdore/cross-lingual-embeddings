import torch
import torch.nn.functional as F
from torch import nn


class SharedSpace(nn.Module):
	def __init__(self, hidden_dim: int, hidden_dim2: int, max_len: int = 100):
		super(SharedSpace, self).__init__()
		self.fc = nn.Linear(max_len * (hidden_dim * 2), hidden_dim2)

	def forward(self, x):
		x = self.fc(x.reshape(x.size(0), -1))
		return F.leaky_relu(x)


class Encoder(nn.Module):
	def __init__(self,
				 vocab_dim: int,
				 embedding_dim: int,
				 hidden_dim: int,
				 hidden_dim2: int,
				 num_layers: int = 1,
				 dropout: float = 0.0,
				 shared_space=nn.Module,
				 bidirectional: bool = True):
		super(Encoder, self).__init__()

		self.embedder = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim, padding_idx=0)
		self.encoder = nn.LSTM(embedding_dim,
							   hidden_dim,
							   dropout=dropout,
							   num_layers=num_layers,
							   bidirectional=bidirectional,
							   batch_first=True)
		self.shared_space = shared_space
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.bidirectional = bidirectional

	def init_hidden(self, batch_size, device):
		bidirectional = 2 if self.bidirectional == True else 1
		hidden = torch.zeros(self.num_layers * bidirectional, batch_size, self.hidden_dim, device=device)
		cell = torch.zeros(self.num_layers * bidirectional, batch_size, self.hidden_dim, device=device)

		return hidden, cell

	def extract_last_hidden(self, hidden, batch_size):
		bidirectional = 2 if self.bidirectional == True else 1
		final_state = hidden[0].view(self.num_layers, bidirectional, batch_size, self.hidden_dim)[-1]

		# Handle directions
		if bidirectional == 1:
			final_hidden_state = final_state.squeeze(0)
		else:
			h_1, h_2 = final_state[0], final_state[1]
			final_hidden_state = torch.cat((h_1, h_2), 1)

		return final_hidden_state

	def forward(self, x):
		embedded = self.embedder(x)
		outputs, hidden = self.encoder(embedded)
		outputs = self.shared_space(outputs)
		return outputs, hidden


class Decoder(nn.Module):
	def __init__(self,
				 vocab_dim: int,
				 embedding_dim: int,
				 hidden_dim: int,
				 hidden_dim2: int,
				 num_layers: int = 1,
				 dropout: float = 0.0,
				 bidirectional: bool = True,
				 max_len: int = 100):
		super(Decoder, self).__init__()

		self.decoder = nn.LSTM(hidden_dim2,
							   hidden_dim,
							   dropout=dropout,
							   num_layers=num_layers,
							   bidirectional=bidirectional,
							   batch_first=True)
		self.fc = nn.Linear(hidden_dim * 2, vocab_dim)
		self.max_len = max_len

	def forward(self, x, hidden):
		decoded_sentence = []
		for _ in range(self.max_len):
			output, hidden = self.decoder(x.unsqueeze(1), hidden)
			decoded_sentence.append(output)

		decoded_sentence = torch.stack(decoded_sentence, dim=-1).squeeze(1).permute(0, 2, 1)
		decoded_sentence = self.fc(decoded_sentence)
		decoded_sentence = F.log_softmax(decoded_sentence, dim=-1)
		return decoded_sentence
