import os.path
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import ray
import torch.utils.data
from loguru import logger
from ray import train
from torch.nn import NLLLoss
from torchtext.vocab import Vocab
from tqdm import tqdm

from dao.Model import Decoder, Encoder
from test import generate
from utils.dataset import prepare_dataset
from utils.loss import contrastive_loss
from utils.processing import get_until_eos
from utils.utils import save_model, save_plot, write_json

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_autoencoder(config: dict,
					  corpus: Any,
					  vocab: Vocab,
					  model_file: str,
					  plot_file: str,
					  study_result_dir: str,
					  optimize: bool = False,
					  ablation_study: bool = False):
	if optimize:
		corpus = ray.get(corpus)

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
	len_vocab = len(vocab)  # config['len_vocab']

	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	hidden_dim2 = config['hidden_dim2']
	num_layers = config['num_layers']
	enc_dropout = config['enc_dropout']
	dec_dropout = config['dec_dropout']

	alpha = config['alpha']
	beta = config['beta']

	# model instantiation
	encoder_fr = Encoder(len_vocab, embedding_dim, hidden_dim, num_layers, enc_dropout).to(device)

	encoder_it = Encoder(len_vocab, embedding_dim, hidden_dim, num_layers, enc_dropout).to(device)

	decoder_fr = Decoder(len_vocab, embedding_dim, hidden_dim, hidden_dim2, num_layers, dec_dropout).to(device)

	decoder_it = Decoder(len_vocab, embedding_dim, hidden_dim, hidden_dim2, num_layers, dec_dropout).to(device)

	# optimizer
	parameters = list(encoder_it.parameters()) + list(encoder_fr.parameters()) + list(decoder_it.parameters()) + list(
		decoder_fr.parameters())

	# optimizer = torch.optim.SGD(params=parameters, lr=lr, momentum=0.9)
	optimizer = torch.optim.Adam(params=parameters, lr=lr)
	scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

	train_losses = []
	val_losses = []
	learning_rates = []

	avg_val_loss = 0.0

	loss_fn = NLLLoss(ignore_index=0)

	# init weight&biases feature
	# wandb.init(project="cross-lingual-it-fr-embeddings-3-loss",
	# 		   config=config}

	itos = vocab.get_itos()

	for epoch in range(num_epochs):

		encoder_fr.train()
		encoder_it.train()
		decoder_fr.train()
		decoder_it.train()

		total_train_loss = 0.0

		hidden_fr = encoder_fr.init_hidden(batch_size, device)
		hidden_it = encoder_it.init_hidden(batch_size, device)

		with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
			# train each encoder-decoder on their language

			for batch_idx, (input_fr, input_it, label) in enumerate(train_loader):
				input_fr = input_fr.to(device)
				input_it = input_it.to(device)

				# computing embeddings from encoders
				hidden_fr = encoder_fr(input_fr, hidden_fr)
				hidden_it = encoder_it(input_it, hidden_it)

				# constrastive loss with label
				cl_loss = contrastive_loss(hidden_fr[0][-1], hidden_it[0][-1], label=label, device=device)

				output_it = decoder_it(input_it, hidden_it)
				output_fr = decoder_fr(input_fr, hidden_fr)

				reconstruction_loss = loss_fn(output_it.reshape(-1, len_vocab), input_it.reshape(-1).long())
				reconstruction_loss += loss_fn(output_fr.reshape(-1, len_vocab), input_fr.reshape(-1).long())

				loss = cl_loss * alpha + reconstruction_loss * beta

				optimizer.zero_grad()

				loss.backward()

				torch.nn.utils.clip_grad_norm_(parameters, 0.25)

				optimizer.step()

				total_train_loss += loss.item()

				hidden_it = decoder_it.detach_hidden(hidden_it)
				hidden_fr = decoder_fr.detach_hidden(hidden_fr)

				pbar.set_postfix({"Train Loss": loss.item()})
				pbar.update(1)

		# optimize learning rate
		learning_rates.append(optimizer.param_groups[0]["lr"])
		scheduler.step()

		avg_train_loss = total_train_loss / len(train_loader)
		logger.info(f"[train] Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_train_loss:.20f}")
		logger.info(f"[train] Epoch [{epoch + 1}/{num_epochs}], Last Contrastive Loss: {cl_loss.item():.20f}")
		logger.info(f"[train] Epoch [{epoch + 1}/{num_epochs}], Last Reconstruction Loss: "
					f"{reconstruction_loss.item():.20f}")

		train_losses.append(avg_train_loss)
		# wandb.log({"train/loss": avg_train_loss, "epoch": epoch + 1})

		encoder_fr.eval()
		encoder_it.eval()
		decoder_fr.eval()
		decoder_it.eval()

		total_val_loss = 0.0

		hidden_fr = encoder_fr.init_hidden(batch_size, device)
		hidden_it = encoder_it.init_hidden(batch_size, device)

		with torch.no_grad():
			for (input_fr, input_it, label) in val_loader:
				input_fr = input_fr.to(device)
				input_it = input_it.to(device)

				# computing embeddings from encoders
				hidden_fr = encoder_fr(input_fr, hidden_fr)
				hidden_it = encoder_it(input_it, hidden_it)

				# constrastive loss with label
				cl_loss = contrastive_loss(hidden_fr[0][-1], hidden_it[0][-1], label=label, device=device)

				output_it = decoder_it(input_fr, hidden_it)
				output_fr = decoder_fr(input_it, hidden_fr)

				if output_it.shape[1] > input_it.shape[1]:
					output_it = output_it[:, :input_it.shape[1], :]
				else:
					output_it = torch.nn.functional.pad(output_it,
														(0, 0, 0, input_it.shape[1] - output_it.shape[1], 0, 0))

				if output_fr.shape[1] > input_fr.shape[1]:
					output_fr = output_fr[:, :input_fr.shape[1], :]
				else:
					output_fr = torch.nn.functional.pad(output_fr,
														(0, 0, 0, input_fr.shape[1] - output_fr.shape[1], 0, 0))

				reconstruction_loss = loss_fn(output_it.reshape(-1, len_vocab), input_it.reshape(-1).long())
				reconstruction_loss += loss_fn(output_fr.reshape(-1, len_vocab), input_fr.reshape(-1).long())

				loss = cl_loss * alpha + reconstruction_loss * beta

				total_val_loss += loss.item()

			# --------------- GET AN EXAMPLE --------------- #
			num = random.randint(0, batch_size - 1)

			input_fr, input_it, label = next(iter(val_loader))
			input_fr = input_fr.to(device)
			input_it = input_it.to(device)

			# computing embeddings from encoders

			output_it = decoder_it(input_fr, encoder_it(input_it, None))
			output_fr = decoder_fr(input_it, encoder_fr(input_fr, None))

			input_fr = input_fr[num].squeeze().tolist()
			input_it = input_it[num].squeeze().tolist()

			input_it = get_until_eos(input_it, vocab)
			input_fr = get_until_eos(input_fr, vocab)

			french_real_phrase = " ".join([itos[int(i)] for i in input_fr[1:]])
			italian_real_phrase = " ".join([itos[int(i)] for i in input_it[1:]])

			output_fr = output_fr.argmax(dim=-1)[num].squeeze().tolist()
			output_it = output_it.argmax(dim=-1)[num].squeeze().tolist()

			output_it = get_until_eos(output_it, vocab)
			output_fr = get_until_eos(output_fr, vocab)

			french_fake_phrase = " ".join([itos[int(i)] for i in output_fr[1:]])
			italian_fake_phrase = " ".join([itos[int(i)] for i in output_it[1:]])

			logger.info("--------------- ITALIAN: ")
			logger.info(f"Generated: {italian_fake_phrase} \n")
			logger.info(f"Real: {italian_real_phrase} \n")
			logger.info("--------------- FRENCH: ")
			logger.info(f"Generated: {french_fake_phrase} \n")
			logger.info(f"Real: {french_real_phrase} \n")

		# --------------- --------------- --------------- #

		avg_val_loss = total_val_loss / len(val_loader)
		val_losses.append(avg_val_loss)
		logger.info(f"[train] Validation Loss: {avg_val_loss:.20f}")

		if optimize:
			train.report({'loss': avg_val_loss, 'train/loss': avg_train_loss})

		# wandb.log({"val/loss": avg_val_loss, "epoch": epoch + 1})

		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss

			# wandb.run.summary["best_loss"] = best_val_loss

			early_stop_counter = 0

			# save best model
			save_model(encoder_fr, file=model_file.format(type="encoder_fr"))
			save_model(decoder_fr, file=model_file.format(type="decoder_fr"))
			save_model(encoder_it, file=model_file.format(type="encoder_it"))
			save_model(decoder_it, file=model_file.format(type="decoder_it"))

		else:
			early_stop_counter += 1
			if early_stop_counter >= patience:
				logger.info(f"[train] Early stopping at epoch {epoch + 1}")
				break

	logger.info("[train] Training complete.")

	bleu_score_fr, bleu_score_it = generate(config, test_loader, model_file, vocab)

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

# wandb.finish()
