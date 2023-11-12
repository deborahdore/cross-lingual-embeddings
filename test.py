from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from nltk.translate.meteor_score import meteor_score
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchtext.data import bleu_score
from torchtext.vocab import Vocab

from dao.Dataset import LSTMDataset
from dao.Model import Decoder, Encoder, SharedSpace
from utils.processing import get_until_eos
from utils.utils import load_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def generate(config, test_loader: Any, model_file: str, vocab_fr: Vocab, vocab_it: Vocab):
	logger.info(f"[generate] device: {device}")

	# model parameters
	batch_size = 1

	len_vocab_fr = len(vocab_fr)
	len_vocab_it = len(vocab_it)

	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	hidden_dim2 = config['hidden_dim2']

	num_layers = config['num_layers']
	enc_dropout = config['enc_dropout']
	dec_dropout = config['dec_dropout']

	# model loading
	shared_space = SharedSpace(hidden_dim, hidden_dim2).to(device)
	shared_space.load_state_dict(load_model(model_file.format(type='shared_space')))

	encoder_fr = Encoder(len_vocab_fr,
						 embedding_dim,
						 hidden_dim,
						 hidden_dim2,
						 num_layers,
						 enc_dropout,
						 shared_space).to(device)
	encoder_it = Encoder(len_vocab_it,
						 embedding_dim,
						 hidden_dim,
						 hidden_dim2,
						 num_layers,
						 enc_dropout,
						 shared_space).to(device)
	decoder_fr = Decoder(len_vocab_fr, embedding_dim, hidden_dim, hidden_dim2, num_layers, dec_dropout).to(device)
	decoder_it = Decoder(len_vocab_it, embedding_dim, hidden_dim, hidden_dim2, num_layers, dec_dropout).to(device)

	encoder_fr.load_state_dict(load_model(model_file.format(type='encoder_fr')))
	encoder_it.load_state_dict(load_model(model_file.format(type='encoder_it')))
	decoder_fr.load_state_dict(load_model(model_file.format(type='decoder_fr')))
	decoder_it.load_state_dict(load_model(model_file.format(type='decoder_it')))

	encoder_fr.eval()
	encoder_it.eval()
	decoder_fr.eval()
	decoder_it.eval()
	shared_space.eval()

	bleu_score_it = 0
	bleu_score_fr = 0
	meteor_score_it = 0
	meteor_score_fr = 0

	itos_fr = vocab_fr.get_itos()
	itos_it = vocab_it.get_itos()

	with torch.no_grad():
		for batch_idx, (input_fr, input_it, _) in enumerate(test_loader):
			input_fr = input_fr.to(device)
			input_it = input_it.to(device)

			# computing embeddings from encoders
			embedding_fr, hidden_fr = encoder_fr(input_fr)
			embedding_it, hidden_it = encoder_it(input_it)

			output_it = decoder_it(embedding_fr, hidden_fr)
			output_fr = decoder_fr(embedding_it, hidden_it)  # swap

			# get indexes
			output_it = output_it.argmax(dim=-1).squeeze()
			output_fr = output_fr.argmax(dim=-1).squeeze()

			candidate_it_corpus = [itos_it[int(i)] for i in get_until_eos(output_it.tolist(), vocab_it)]
			candidate_fr_corpus = [itos_fr[int(i)] for i in get_until_eos(output_fr.tolist(), vocab_fr)]

			reference_fr_corpus = [itos_fr[int(i)] for i in get_until_eos(input_fr.squeeze().tolist(), vocab_fr)]
			reference_it_corpus = [itos_it[int(i)] for i in get_until_eos(input_it.squeeze().tolist(), vocab_it)]

			meteor_score_it += meteor_score([reference_it_corpus], candidate_it_corpus)
			meteor_score_fr += meteor_score([reference_fr_corpus], candidate_fr_corpus)

			bleu_score_it += bleu_score([candidate_it_corpus], [reference_it_corpus])
			bleu_score_fr += bleu_score([candidate_fr_corpus], [reference_fr_corpus])

	bleu_score_it = bleu_score_it / len(test_loader)
	bleu_score_fr = bleu_score_fr / len(test_loader)

	logger.info(f"[generate] Bleu score italian corpus {bleu_score_it}")
	logger.info(f"[generate] Bleu score french corpus {bleu_score_fr}")
	logger.info(f"[generate] Meteor score italian corpus {meteor_score_it}")
	logger.info(f"[generate] Meteor score french corpus {meteor_score_fr}")

	return bleu_score_fr, bleu_score_it, meteor_score_it, meteor_score_fr


def visualize_embeddings(config: {},
						 dataset: pd.DataFrame,
						 model_file: str,
						 plot_file: str,
						 vocab_fr: Vocab,
						 vocab_it: Vocab):
	logger.info(f"[visualize_embeddings] {config}")

	batch_size = 1

	len_vocab_fr = len(vocab_fr)
	len_vocab_it = len(vocab_it)

	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	hidden_dim2 = config['hidden_dim2']

	num_layers = config['num_layers']
	enc_dropout = config['enc_dropout']
	dec_dropout = config['dec_dropout']

	# model loading
	shared_space = SharedSpace(hidden_dim, hidden_dim2).to(device)
	shared_space.load_state_dict(load_model(model_file.format(type='shared_space')))

	encoder_fr = Encoder(len_vocab_fr,
						 embedding_dim,
						 hidden_dim,
						 hidden_dim2,
						 num_layers,
						 enc_dropout,
						 shared_space).to(device)
	encoder_it = Encoder(len_vocab_it,
						 embedding_dim,
						 hidden_dim,
						 hidden_dim2,
						 num_layers,
						 enc_dropout,
						 shared_space).to(device)
	decoder_fr = Decoder(len_vocab_fr, embedding_dim, hidden_dim, hidden_dim2, num_layers, dec_dropout).to(device)
	decoder_it = Decoder(len_vocab_it, embedding_dim, hidden_dim, hidden_dim2, num_layers, dec_dropout).to(device)

	encoder_fr.load_state_dict(load_model(model_file.format(type='encoder_fr')))
	encoder_it.load_state_dict(load_model(model_file.format(type='encoder_it')))
	decoder_fr.load_state_dict(load_model(model_file.format(type='decoder_fr')))
	decoder_it.load_state_dict(load_model(model_file.format(type='decoder_it')))

	encoder_fr.eval()
	encoder_it.eval()
	decoder_fr.eval()
	decoder_it.eval()
	shared_space.eval()

	dataset = dataset.sample(n=8).reset_index(drop=True)  # sample 8 for plot
	dataset = LSTMDataset(corpus_fr=dataset['french'], corpus_it=dataset['italian'], negative_sampling=False)
	loader = DataLoader(dataset, batch_size=batch_size)

	points = []
	text = []
	colors = []

	itos_fr = vocab_fr.get_itos()
	itos_it = vocab_it.get_itos()

	with torch.no_grad():
		for i, (input_fr, input_it, _) in enumerate(loader):
			input_fr = input_fr.to(device)
			input_it = input_it.to(device)

			# computing embeddings from encoders
			embedding_fr, hidden_fr = encoder_fr(input_fr)
			embedding_it, hidden_it = encoder_it(input_it)

			points.append(embedding_fr.squeeze().cpu().detach().numpy())
			text.append(" ".join([itos_fr[int(i)] for i in get_until_eos(input_fr.squeeze().tolist(), vocab_fr)]))

			points.append(embedding_it.squeeze().cpu().detach().numpy())
			text.append(" ".join([itos_it[int(i)] for i in get_until_eos(input_it.squeeze().tolist(), vocab_it)]))

			colors.extend([i] * 2)

	points = np.array(points)
	pca = PCA(n_components=2, svd_solver='full')
	new_points = pca.fit_transform(points)

	plt.figure(figsize=(10, 10))
	plt.scatter(new_points[:, 0], new_points[:, 1], s=20.0, c=colors, cmap='tab10', alpha=0.9)

	for i, label in enumerate(text):
		plt.text(new_points[i, 0], new_points[i, 1], label, fontsize=8, ha='center', va='bottom')

	plt.title('Embeddings Projection')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.tight_layout()
	plt.savefig(plot_file.format(file_name="embeddings_projection"))
	plt.close()


def visualize_latent_space(config: {},
						   dataset: pd.DataFrame,
						   model_file: str,
						   plot_file: str,
						   vocab_fr: Vocab,
						   vocab_it: Vocab):
	logger.info(f"[visualize_latent_space] {config}")

	batch_size = 1

	len_vocab_fr = len(vocab_fr)
	len_vocab_it = len(vocab_it)

	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	hidden_dim2 = config['hidden_dim2']

	num_layers = config['num_layers']
	enc_dropout = config['enc_dropout']
	dec_dropout = config['dec_dropout']

	# model loading
	shared_space = SharedSpace(hidden_dim, hidden_dim2).to(device)
	shared_space.load_state_dict(load_model(model_file.format(type='shared_space')))

	encoder_fr = Encoder(len_vocab_fr,
						 embedding_dim,
						 hidden_dim,
						 hidden_dim2,
						 num_layers,
						 enc_dropout,
						 shared_space).to(device)
	encoder_it = Encoder(len_vocab_it,
						 embedding_dim,
						 hidden_dim,
						 hidden_dim2,
						 num_layers,
						 enc_dropout,
						 shared_space).to(device)
	decoder_fr = Decoder(len_vocab_fr, embedding_dim, hidden_dim, hidden_dim2, num_layers, dec_dropout).to(device)
	decoder_it = Decoder(len_vocab_it, embedding_dim, hidden_dim, hidden_dim2, num_layers, dec_dropout).to(device)

	encoder_fr.load_state_dict(load_model(model_file.format(type='encoder_fr')))
	encoder_it.load_state_dict(load_model(model_file.format(type='encoder_it')))
	decoder_fr.load_state_dict(load_model(model_file.format(type='decoder_fr')))
	decoder_it.load_state_dict(load_model(model_file.format(type='decoder_it')))

	encoder_fr.eval()
	encoder_it.eval()
	decoder_fr.eval()
	decoder_it.eval()
	shared_space.eval()

	dataset = LSTMDataset(corpus_fr=dataset['french'], corpus_it=dataset['italian'], negative_sampling=False)
	loader = DataLoader(dataset, batch_size=batch_size)

	points_fr = []
	points_it = []

	with torch.no_grad():
		for i, (input_fr, input_it, label) in enumerate(loader):
			input_fr = input_fr.to(device)
			input_it = input_it.to(device)
			assert label.item() == 1
			# computing embeddings from encoders
			embedding_fr, hidden_fr = encoder_fr(input_fr)
			embedding_it, hidden_it = encoder_it(input_it)

			points_fr.append(embedding_fr.squeeze().cpu().detach().numpy())
			points_it.append(embedding_it.squeeze().cpu().detach().numpy())

	points_fr = np.array(points_fr)
	points_it = np.array(points_it)

	pca = PCA(n_components=2, svd_solver='full')
	points_fr = pca.fit_transform(points_fr)
	points_it = pca.fit_transform(points_it)

	plt.figure(figsize=(10, 10))
	points = np.concatenate([points_fr, points_it], axis=0)
	colors = np.concatenate([[0] * len(points_fr), [1] * len(points_it)], axis=0)
	plt.scatter(points[:, 0], points[:, 1], s=20.0, c=colors, alpha=0.9)

	plt.title('Latent Space Projection')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.tight_layout()
	plt.savefig(plot_file.format(file_name="latent_space_projection"))
	plt.close()
