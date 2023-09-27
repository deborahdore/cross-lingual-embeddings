import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchtext.data import bleu_score
from tqdm import tqdm

from dao.AEDataset import AEDataset
from dao.model import Encoder, LatentSpace, Decoder
from train import contrastive_loss
from utils.processing import get_sentence_in_natural_language
from utils.utils import read_json, load_model

from loguru import logger

def test(test_loader: DataLoader, model_config_file: str, model_file: str) -> None:
	config = read_json(model_config_file)
	logger.info(f"[test] {config}")

	# model parameters
	len_vocab_fr = config['len_vocab_fr']
	len_vocab_it = config['len_vocab_it']
	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	ls_dim = config['ls_dim']
	hidden_lstm_dim = config['hidden_lstm_dim']
	output_dim = config['output_dim']

	# model loading
	encoder_fr = Encoder(len_vocab_fr, embedding_dim, hidden_dim)
	encoder_fr.load_state_dict(load_model(model_file.format(type='encoder_fr')))

	encoder_it = Encoder(len_vocab_it, embedding_dim, hidden_dim)
	encoder_it.load_state_dict(load_model(model_file.format(type='encoder_it')))

	latent_space = LatentSpace(hidden_dim, ls_dim)
	latent_space.load_state_dict(load_model(model_file.format(type='latent_space')))

	decoder_fr = Decoder(ls_dim, hidden_lstm_dim, output_dim)
	decoder_fr.load_state_dict(load_model(model_file.format(type='decoder_fr')))

	decoder_it = Decoder(ls_dim, hidden_lstm_dim, output_dim)
	decoder_it.load_state_dict(load_model(model_file.format(type='decoder_it')))

	encoder_fr.eval()
	encoder_it.eval()
	decoder_fr.eval()
	decoder_it.eval()
	latent_space.eval()

	total_test_loss = 0.0
	losses = []

	mse_loss = MSELoss()

	with tqdm(total=len(test_loader), desc="Testing", unit="batch") as pbar:
		for batch_idx, (input_fr, input_it, label) in enumerate(test_loader):
			with torch.no_grad():
				embedding_fr = latent_space(encoder_fr(input_fr))
				embedding_it = latent_space(encoder_it(input_it))

				loss = contrastive_loss(embedding_fr, embedding_it, label=label)

				output_fr = decoder_fr(embedding_fr)
				output_it = decoder_it(embedding_it)

				loss += mse_loss(input_it, output_it) + mse_loss(input_fr, output_fr)

				total_test_loss += loss.item()
				losses.append(loss.item())

			pbar.update(1)

	avg_test_loss = total_test_loss / len(test_loader)
	logger.info(f"[test] Average Test Loss: {avg_test_loss:.20f}")


def visualize_latent_space(dataset: pd.DataFrame, vocab_it: [], vocab_fr: [], model_config_file: str, model_file: str,
						   plot_file: str) -> None:
	config = read_json(model_config_file)
	logger.info(f"[visualize_latent_space] {config}")

	# model parameters
	len_vocab_fr = config['len_vocab_fr']
	len_vocab_it = config['len_vocab_it']
	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	ls_dim = config['ls_dim']

	# load dataset
	dataset = dataset.sample(n=10).reset_index(drop=True)
	dataset = AEDataset(corpus_fr=dataset['french'], corpus_it=dataset['italian'], negative_sampling=False)

	# load saved models
	encoder_fr = Encoder(len_vocab_fr, embedding_dim, hidden_dim)
	encoder_fr.load_state_dict(load_model(model_file.format(type='encoder_fr')))

	encoder_it = Encoder(len_vocab_it, embedding_dim, hidden_dim)
	encoder_it.load_state_dict(load_model(model_file.format(type='encoder_it')))

	latent_space = LatentSpace(hidden_dim, ls_dim)
	latent_space.load_state_dict(load_model(model_file.format(type='latent_space')))

	encoder_fr.eval()
	encoder_it.eval()
	latent_space.eval()

	points = []
	text = []
	colors = []

	loader = DataLoader(dataset, batch_size=1)

	for i, (sent_fr, sent_it, _) in enumerate(loader):
		# Extract the embeddings from the latent space
		embedding_fr = latent_space(encoder_fr(sent_fr))
		embedding_it = latent_space(encoder_it(sent_it))

		points.extend(embedding_fr.detach().numpy())
		points.extend(embedding_it.detach().numpy())

		text.append(get_sentence_in_natural_language(sent_fr, vocab_fr))
		text.append(get_sentence_in_natural_language(sent_it, vocab_it))

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


def translate(dataset: pd.DataFrame, vocab_it: [], vocab_fr: [], model_config_file: str, model_file: str):
	config = read_json(model_config_file)
	logger.info(f"[translate] {config}")

	# model parameters
	len_vocab_fr = config['len_vocab_fr']
	len_vocab_it = config['len_vocab_it']
	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	ls_dim = config['ls_dim']
	hidden_lstm_dim = config['hidden_lstm_dim']
	output_dim = config['output_dim']

	# model loading
	encoder_fr = Encoder(len_vocab_fr, embedding_dim, hidden_dim)
	encoder_fr.load_state_dict(load_model(model_file.format(type='encoder_fr')))

	encoder_it = Encoder(len_vocab_it, embedding_dim, hidden_dim)
	encoder_it.load_state_dict(load_model(model_file.format(type='encoder_it')))

	latent_space = LatentSpace(hidden_dim, ls_dim)
	latent_space.load_state_dict(load_model(model_file.format(type='latent_space')))

	decoder_fr = Decoder(ls_dim, hidden_lstm_dim, output_dim)
	decoder_fr.load_state_dict(load_model(model_file.format(type='decoder_fr')))

	decoder_it = Decoder(ls_dim, hidden_lstm_dim, output_dim)
	decoder_it.load_state_dict(load_model(model_file.format(type='decoder_it')))

	encoder_fr.eval()
	encoder_it.eval()
	decoder_fr.eval()
	decoder_it.eval()
	latent_space.eval()
	candidate_corpus = []
	references_corpus = []

	dataset = AEDataset(corpus_fr=dataset['french'], corpus_it=dataset['italian'], negative_sampling=False)
	loader = DataLoader(dataset, batch_size=1)

	for batch_idx, (sentence_fr, sentence_it, _) in enumerate(loader):
		with torch.no_grad():
			output_it = decoder_it(latent_space(encoder_fr(sentence_fr)))

			candidate_corpus.append(get_sentence_in_natural_language(output_it, vocab_it))
			references_corpus.append(get_sentence_in_natural_language(sentence_it, vocab_it))

	logger.info(f'Bleu score: {bleu_score(candidate_corpus=candidate_corpus, references_corpus=references_corpus)}')
