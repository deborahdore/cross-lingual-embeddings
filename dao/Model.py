from torch import nn


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
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(hidden_dim, proj_dim)

		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

	def forward(self, x):
		x = x.long()
		embedding = self.embedder(x)
		out, hidden = self.lstm(embedding)
		out = nn.Tanh(out)
		out = self.dropout(out)
		out = self.fc(out[:, -1, :])
		return out, hidden


class Decoder(nn.Module):
	def __init__(self,
				 proj_dim: int,
				 hidden_dim: int,
				 output_dim: int,
				 num_layers: int = 1,
				 dropout: float = 0.0):
		super(Decoder, self).__init__()

		self.lstm = nn.LSTM(proj_dim, hidden_dim, num_layers=num_layers, batch_first=True)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(hidden_dim, output_dim)

		self.output_dim = output_dim
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim

	def forward(self, x, hidden):
		out, _ = self.lstm(x[:, None, :], hidden)
		out = nn.Tanh(out)
		out = self.dropout(out)
		out = self.fc(out[:, -1, :])
		return out
