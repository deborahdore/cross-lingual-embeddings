import os.path
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import ray
import torch.utils.data
import wandb
from loguru import logger
from ray import train
from torch.nn import NLLLoss
from torchtext.vocab import Vocab
from tqdm import tqdm

from config.path import base_dir
from dao.Model import Decoder, Encoder, SharedSpace
from test import generate
from utils.dataset import prepare_dataset
from utils.loss import contrastive_loss
from utils.processing import get_until_eos
from utils.utils import save_model, save_plot, write_json

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_an_example(encoder_it, encoder_fr, decoder_it, decoder_fr, vocab_fr, vocab_it, val_loader, batch_size):
	with torch.no_grad():
		itos_fr = vocab_fr.get_itos()
		itos_it = vocab_it.get_itos()

		num = random.randint(0, batch_size - 1)
		input_fr, input_it, label = next(iter(val_loader))

		input_fr = input_fr.to(device)
		input_it = input_it.to(device)

		input_it = input_it[num].unsqueeze(0)
		input_fr = input_fr[num].unsqueeze(0)

		italian_real_phrase = " ".join([itos_it[int(i)] for i in get_until_eos(input_it.squeeze().tolist(), vocab_it)])
		french_real_phrase = " ".join([itos_fr[int(i)] for i in get_until_eos(input_fr.squeeze().tolist(), vocab_fr)])

		logger.info(f"Input sentence (fr): {french_real_phrase}")
		logger.info(f"Goal (it): {italian_real_phrase}")

		embedding_fr, hidden_fr = encoder_fr(input_fr)
		embedding_it, hidden_it = encoder_it(input_fr)

		reconstructed_french_phrase = " ".join([itos_fr[int(i)] for i in
												decoder_fr(embedding_fr, hidden_fr).argmax(-1).squeeze().tolist()])
		logger.info(f"Reconstructed sentence (fr): {reconstructed_french_phrase}")

		translated_italian_phrase = " ".join([itos_it[int(i)] for i in
											  decoder_it(embedding_fr, hidden_fr).argmax(-1).squeeze().tolist()])
		logger.info(f"Translated sentence (fr->it) with encoder fr: {translated_italian_phrase}")

		translated_italian_phrase = " ".join([itos_it[int(i)] for i in
											  decoder_it(embedding_it, hidden_it).argmax(-1).squeeze().tolist()])
		logger.info(f"Translated sentence (fr->it) with encoder it: {translated_italian_phrase}")


def train_autoencoder(config: dict,
					  corpus: Any,
					  vocab_fr: Vocab,
					  vocab_it: Vocab,
					  model_file: str,
					  plot_file: str,
					  study_result_dir: str,
					  optimize: bool = False,
					  ablation_study: bool = False):
	if optimize:
		corpus = ray.get(corpus)

	# init weight&biases feature
	# name of the run
	name = base_dir.split("/")[-1] + datetime.now()
	wandb.init(project="cross-lingual-embeddings", name=name, config=config)

	train_loader, val_loader, test_loader = prepare_dataset(corpus, config)

	logger.info(f"[train] device: {device}")

	# training parameters
	batch_size = config['batch_size']

	patience = config['patience']
	early_stop_counter = 0
	best_val_loss = float('inf')
	num_epochs = config['num_epochs']
	lr = config['lr']

	# model parameters
	len_vocab_fr = len(vocab_fr)
	len_vocab_it = len(vocab_it)

	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	hidden_dim2 = config['hidden_dim2']
	num_layers = config['num_layers']
	enc_dropout = config['enc_dropout']
	dec_dropout = config['dec_dropout']

	alpha = config['alpha']
	beta = config['beta']

	# model instantiation
	shared_space = SharedSpace(hidden_dim, hidden_dim2).to(device)

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

	wandb.watch(encoder_fr)
	wandb.watch(encoder_it)
	wandb.watch(decoder_it)
	wandb.watch(decoder_fr)
	wandb.watch(shared_space)

	params = list(encoder_it.parameters()) + list(encoder_fr.parameters()) + list(decoder_it.parameters()) + list(
		decoder_fr.parameters())

	optimizer = torch.optim.AdamW(params=params, lr=lr)
	scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

	train_losses = []
	val_losses = []
	learning_rates = []

	avg_val_loss = 0.0

	loss_fn = NLLLoss(ignore_index=0)

	for epoch in range(num_epochs):

		encoder_fr.train()
		encoder_it.train()
		decoder_fr.train()
		decoder_it.train()
		shared_space.train()

		total_train_loss = 0.0

		with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
			# train each encoder-decoder on their language
			for batch_idx, (input_fr, input_it, label) in enumerate(train_loader):
				input_fr = input_fr.to(device)
				input_it = input_it.to(device)
				label = label.to(device)

				optimizer.zero_grad()

				# computing embeddings from encoders
				embedding_fr, hidden_fr = encoder_fr(input_fr)
				embedding_it, hidden_it = encoder_it(input_it)

				output_fr = decoder_fr(embedding_fr, hidden_fr)
				output_it = decoder_it(embedding_it, hidden_it)

				# constrastive loss with label
				cl_loss = contrastive_loss(embedding_fr, embedding_it, label=label)

				reconstruction_loss = loss_fn(output_it.reshape(-1, len_vocab_it), input_it.reshape(-1).long())
				reconstruction_loss += loss_fn(output_fr.reshape(-1, len_vocab_fr), input_fr.reshape(-1).long())

				loss = cl_loss * alpha + reconstruction_loss * beta
				loss.backward()
				optimizer.step()

				total_train_loss += loss.item()

				pbar.set_postfix({"Train Loss": loss.item(), "C-Loss": cl_loss.item()})
				pbar.update(1)

		learning_rates.append(optimizer.param_groups[0]["lr"])
		scheduler.step()

		avg_train_loss = total_train_loss / len(train_loader)
		train_losses.append(avg_train_loss)
		logger.info(f"[train] Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_train_loss:.20f}")
		wandb.log({"train/loss": avg_train_loss, "epoch": epoch + 1})

		encoder_fr.eval()
		encoder_it.eval()
		decoder_fr.eval()
		decoder_it.eval()
		shared_space.eval()

		total_val_loss = 0.0
		total_cl_loss = 0.0
		total_reconstruction_loss = 0.0

		with torch.no_grad():
			for (input_fr, input_it, label) in val_loader:
				input_fr = input_fr.to(device)
				input_it = input_it.to(device)
				label = label.to(device)

				optimizer.zero_grad()

				# computing embeddings from encoders
				embedding_fr, hidden_fr = encoder_fr(input_fr)
				embedding_it, hidden_it = encoder_it(input_it)

				output_fr = decoder_fr(embedding_fr, hidden_fr)
				output_it = decoder_it(embedding_it, hidden_it)

				# constrastive loss with label
				cl_loss = contrastive_loss(embedding_fr, embedding_it, label=label)

				reconstruction_loss = loss_fn(output_it.reshape(-1, len_vocab_it), input_it.reshape(-1).long())
				reconstruction_loss += loss_fn(output_fr.reshape(-1, len_vocab_fr), input_fr.reshape(-1).long())

				total_cl_loss += cl_loss
				total_reconstruction_loss += reconstruction_loss
				loss = cl_loss * alpha + reconstruction_loss * beta

				total_val_loss += loss.item()

			avg_val_loss = total_val_loss / len(val_loader)
			avg_cl_loss = total_cl_loss / len(val_loader)
			avg_rec_loss = total_reconstruction_loss / len(val_loader)

			val_losses.append(avg_val_loss)

			wandb.log({"val/loss": avg_val_loss, "epoch": epoch + 1})
			wandb.log({"contrastive/loss": avg_cl_loss, "epoch": epoch + 1})
			wandb.log({"reconstruction/loss": avg_rec_loss, "epoch": epoch + 1})

			logger.info(f"[train] Validation Loss: {avg_val_loss:.20f}")
			logger.info(f"[train] Contrastive Loss: {avg_cl_loss:.20f}")
			logger.info(f"[train] Reconstruction Loss: {avg_rec_loss:.20f}")

			# --------------- GET AN EXAMPLE --------------- #
			get_an_example(encoder_it, encoder_fr, decoder_it, decoder_fr, vocab_fr, vocab_it, val_loader, batch_size)

		if optimize:
			train.report({'loss': avg_val_loss, 'train/loss': avg_train_loss})

		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss

			wandb.run.summary["best_loss"] = best_val_loss

			early_stop_counter = 0

			save_model(encoder_fr, file=model_file.format(type="encoder_fr"))
			save_model(decoder_fr, file=model_file.format(type="decoder_fr"))
			save_model(encoder_it, file=model_file.format(type="encoder_it"))
			save_model(decoder_it, file=model_file.format(type="decoder_it"))
			save_model(shared_space, file=model_file.format(type="shared_space"))

		else:
			early_stop_counter += 1
			if early_stop_counter >= patience:
				logger.info(f"[train] Early stopping at epoch {epoch + 1}")
				break

	logger.info("[train] Training complete.")

	bleu_score_fr, bleu_score_it, meteor_score_it, meteor_score_fr = generate(config,
																			  test_loader,
																			  model_file,
																			  vocab_fr,
																			  vocab_it)

	if ablation_study:
		trial_datetime = datetime.now()
		final_abl_dir = os.path.join(study_result_dir, str(trial_datetime))
		Path(final_abl_dir).mkdir(parents=True, exist_ok=True)

		train_vs_val_loss_file = os.path.join(final_abl_dir, "train_vs_val_loss.svg")
		lr_epochs_file = os.path.join(final_abl_dir, "learning_rate.svg")

		# save status + loss
		models_config = {str(trial_datetime): config}
		models_config.update({"loss": avg_val_loss})
		models_config.update({"bleu_it": bleu_score_it})
		models_config.update({"bleu_fr": bleu_score_fr})
		models_config.update({"meteor_it": meteor_score_it})
		models_config.update({"meteor_fr": meteor_score_fr})
		write_json(models_config, final_abl_dir)

	else:
		train_vs_val_loss_file = plot_file.format(file_name="train_vs_val_loss")
		lr_epochs_file = plot_file.format(file_name="learning_rate")

	# save losses
	plt_elems = [("train loss", train_losses), ("validation loss", val_losses)]
	save_plot(plt_elems, "epochs", "losses", "train vs val loss", train_vs_val_loss_file)

	# save learning rates
	save_plot([("learning rates", learning_rates)],
			  "epochs",
			  "learning rates",
			  "learning rates through epochs",
			  lr_epochs_file)

	wandb.finish()
