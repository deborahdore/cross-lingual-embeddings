from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchtext.data import bleu_score
from torchtext.vocab import Vocab

from dao.Dataset import LSTMDataset
from dao.Model import Decoder, Encoder
from utils.utils import load_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import torch.nn.functional as F


def generate(config, test_loader: Any, model_file: str, vocab_fr: Vocab, vocab_it: Vocab):
	test_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)

	logger.info(f"[generate] device: {device}")

	# model parameters
	batch_size = 1

	len_vocab_it = config['len_vocab_it']
	len_vocab_fr = config['len_vocab_fr']

	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	num_layers = config['num_layers']
	enc_dropout = config['enc_dropout']
	dec_dropout = config['dec_dropout']

	# model loading
	encoder_fr = Encoder(len_vocab_fr, embedding_dim, hidden_dim, num_layers, enc_dropout).to(device)
	encoder_fr.load_state_dict(load_model(model_file.format(type='encoder_fr')))
	encoder_fr.to(device)

	encoder_it = Encoder(len_vocab_it, embedding_dim, hidden_dim, num_layers, enc_dropout).to(device)
	encoder_it.load_state_dict(load_model(model_file.format(type='encoder_it')))
	encoder_it.to(device)

	decoder_fr = Decoder(len_vocab_fr, embedding_dim, hidden_dim, vocab_fr, num_layers, dec_dropout).to(device)
	decoder_fr.load_state_dict(load_model(model_file.format(type='decoder_fr')))
	decoder_fr.to(device)

	decoder_it = Decoder(len_vocab_it, embedding_dim, hidden_dim, vocab_it, num_layers, dec_dropout).to(device)
	decoder_it.load_state_dict(load_model(model_file.format(type='decoder_it')))
	decoder_it.to(device)

	encoder_fr.eval()
	encoder_it.eval()
	decoder_fr.eval()
	decoder_it.eval()

	italian_sentences = []
	real_italian_sentences = []

	french_sentences = []
	real_french_sentences = []

	itos_it = vocab_it.get_itos()
	itos_fr = vocab_fr.get_itos()

	for batch_idx, (input_fr, input_it, label) in enumerate(test_loader):
		hidden_fr = encoder_fr.init_hidden(batch_size, device)
		hidden_it = encoder_it.init_hidden(batch_size, device)

		with torch.no_grad():
			input_fr = input_fr.to(device)
			input_it = input_it.to(device)

			hidden_fr = encoder_fr(input_fr, hidden_fr)
			hidden_it = encoder_it(input_it, hidden_it)

			output_it, _ = decoder_it(input_it, hidden_fr, teacher_forcing=False)
			output_fr, _ = decoder_fr(input_fr, hidden_it, teacher_forcing=False)

			# get indexes
			output_it = torch.argmax(F.softmax(output_it, dim=-1), dim=-1).squeeze().to(device)
			output_fr = torch.argmax(F.softmax(output_fr, dim=-1), dim=-1).squeeze().to(device)

			italian_sentences.append([itos_it[int(i)] for i in output_it])
			real_italian_sentences.append([itos_it[int(i)] for i in input_it[-1]])

			french_sentences.append([itos_fr[int(i)] for i in output_fr])
			real_french_sentences.append([itos_fr[int(i)] for i in input_fr[-1]])

	bleu_score_it = bleu_score(candidate_corpus=french_sentences, references_corpus=real_french_sentences)
	bleu_score_fr = bleu_score(candidate_corpus=italian_sentences, references_corpus=real_italian_sentences)

	logger.info(f"[generate] Bleu score italian corpus {bleu_score_it}")
	logger.info(f"[generate] Bleu score french corpus {bleu_score_fr}")


def visualize_latent_space(config: {},
						   dataset: pd.DataFrame,
						   model_file: str,
						   plot_file: str,
						   vocab_fr: Vocab,
						   vocab_it: Vocab):
	logger.info(f"[visualize_latent_space] {config}")

	# model parameters
	len_vocab_it = config['len_vocab_it']
	len_vocab_fr = config['len_vocab_fr']
	batch_size = config['batch_size']

	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	num_layers = config['num_layers']
	enc_dropout = config['enc_dropout']
	dec_dropout = config['dec_dropout']

	# load dataset
	dataset = LSTMDataset(corpus_fr=dataset['french'], corpus_it=dataset['italian'], negative_sampling=False)

	# model loading
	encoder_fr = Encoder(len_vocab_fr, embedding_dim, hidden_dim, num_layers, enc_dropout).to(device)
	encoder_fr.load_state_dict(load_model(model_file.format(type='encoder_fr')))
	encoder_fr.to(device)

	encoder_it = Encoder(len_vocab_it, embedding_dim, hidden_dim, num_layers, enc_dropout).to(device)
	encoder_it.load_state_dict(load_model(model_file.format(type='encoder_it')))
	encoder_it.to(device)

	decoder_fr = Decoder(len_vocab_fr, embedding_dim, hidden_dim, vocab_fr, num_layers, dec_dropout).to(device)
	decoder_fr.load_state_dict(load_model(model_file.format(type='decoder_fr')))
	decoder_fr.to(device)

	decoder_it = Decoder(len_vocab_it, embedding_dim, hidden_dim, vocab_it, num_layers, dec_dropout).to(device)
	decoder_it.load_state_dict(load_model(model_file.format(type='decoder_it')))
	decoder_it.to(device)

	encoder_fr.eval()
	encoder_it.eval()
	decoder_fr.eval()
	decoder_it.eval()

	points = []
	text = []
	colors = []

	loader = DataLoader(dataset, batch_size=1)

	itos_fr = vocab_fr.get_itos()
	itos_it = vocab_it.get_itos()

	for i, (input_fr, input_it, _) in enumerate(loader):
		input_fr = input_fr.to(device)
		input_it = input_it.to(device)

		hidden_fr = encoder_fr.init_hidden(batch_size, device)
		hidden_it = encoder_it.init_hidden(batch_size, device)

		hidden_fr = encoder_fr(input_fr, hidden_fr)
		hidden_it = encoder_it(input_it, hidden_it)

		points.extend(hidden_fr[0][-1].detach().numpy())
		text.append([itos_fr[int(i)] for i in input_fr])

		points.extend(hidden_it[0][-1].detach().numpy())
		text.append([itos_it[int(i)] for i in input_it])

		colors.extend([i] * 2)

	points = np.array(points)
	pca = PCA(n_components=2, svd_solver='full')
	new_points = pca.fit_transform(points)

	plt.figure(figsize=(10, 10))
	plt.scatter(new_points[:, 0], new_points[:, 1], s=20.0, c=colors, cmap='tab10', alpha=0.9)

	for i, label in enumerate(text):
		plt.text(new_points[i, 0], new_points[i, 1], label, fontsize=8, ha='center', va='bottom')

	plt.title('Latent Space Projection')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.savefig(plot_file.format(file_name="latent_space_projection"))
	# plt.show()
	plt.close()
