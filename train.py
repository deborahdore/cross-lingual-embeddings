import os.path
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch.utils.data
from loguru import logger
from ray import train
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from dao.Model import Decoder, Encoder
from test import generate
from utils.dataset import prepare_dataset
from utils.loss import contrastive_loss
from utils.utils import save_model, save_plot, write_json

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(123)


def train_autoencoder(config,
					  corpus: pd.DataFrame,
					  model_config_file: str,
					  vocab_fr: Any,
					  vocab_it: Any,
					  model_file: str,
					  plot_file: str,
					  study_result_dir: str,
					  optimize: bool = False,
					  ablation_study: bool = False):
	train_loader, val_loader, test_loader = prepare_dataset(corpus, model_config_file, vocab_fr, vocab_it, config)

	logger.info(f"[train] device: {device}")

	# training parameters
	batch_size = config['batch_size']
	patience = config['patience']
	early_stop_counter = 0
	best_val_loss = float('inf')
	num_epochs = config['num_epochs']
	lr = config['lr']

	# model parameters
	len_vocab_it = config['len_vocab_it']
	len_vocab_fr = config['len_vocab_fr']

	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	num_layers = config['num_layers']
	enc_dropout = config['enc_dropout']
	dec_dropout = config['dec_dropout']

	alpha = config['alpha']
	beta = config['beta']

	# model instantiation
	encoder_fr = Encoder(len_vocab_fr, embedding_dim, hidden_dim, num_layers, enc_dropout).to(device)

	encoder_it = Encoder(len_vocab_it, embedding_dim, hidden_dim, num_layers, enc_dropout).to(device)

	decoder_fr = Decoder(len_vocab_fr, embedding_dim, hidden_dim, vocab_fr, num_layers, dec_dropout).to(device)

	decoder_it = Decoder(len_vocab_it, embedding_dim, hidden_dim, vocab_it, num_layers, dec_dropout).to(device)

	# optimizer
	parameters = list(encoder_it.parameters()) + list(encoder_fr.parameters()) + list(decoder_it.parameters()) + list(
		decoder_fr.parameters())

	optimizer = torch.optim.SGD(params=parameters, lr=lr, momentum=0.9)
	scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

	train_losses = []
	val_losses = []
	learning_rates = []

	avg_val_loss = 0.0

	loss_fn = CrossEntropyLoss()

	# init weight&biases feature
	# wandb.init(project="cross-lingual-it-fr-embeddings-3-loss",
	# 		   config=config}

	for epoch in range(num_epochs):

		encoder_fr.train()
		encoder_it.train()
		decoder_fr.train()
		decoder_it.train()

		total_train_loss = 0.0

		with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
			for batch_idx, (input_fr, input_it, label) in enumerate(train_loader):
				input_fr = input_fr.to(device)
				input_it = input_it.to(device)

				hidden_fr = encoder_fr.detach_hidden(encoder_fr.init_hidden(batch_size, device))
				hidden_it = encoder_it.detach_hidden(encoder_it.init_hidden(batch_size, device))

				# computing embeddings from encoders
				hidden_fr = encoder_fr(input_fr, hidden_fr)
				hidden_it = encoder_it(input_it, hidden_it)

				# constrastive loss with label
				cl_loss = contrastive_loss(hidden_fr[0], hidden_it[0], label=label, device=device)
				cl_loss += contrastive_loss(hidden_fr[1], hidden_it[1], label=label, device=device)

				output_it, _ = decoder_it(input_it, hidden_it)
				output_fr, _ = decoder_fr(input_fr, hidden_fr)

				reconstruction_loss = loss_fn(output_it.reshape(batch_size * input_it.shape[1], -1),
											  input_it.reshape(-1).long())
				reconstruction_loss += loss_fn(output_fr.reshape(batch_size * input_fr.shape[1], -1),
											   input_fr.reshape(-1).long())

				loss = cl_loss * alpha + reconstruction_loss * beta

				optimizer.zero_grad()

				loss.backward()

				torch.nn.utils.clip_grad_norm_(encoder_fr.parameters(), 0.25)
				torch.nn.utils.clip_grad_norm_(encoder_it.parameters(), 0.25)
				torch.nn.utils.clip_grad_norm_(decoder_fr.parameters(), 0.25)
				torch.nn.utils.clip_grad_norm_(decoder_it.parameters(), 0.25)

				optimizer.step()

				total_train_loss += loss.item()

				pbar.set_postfix({"Train Loss": loss.item()})

				pbar.update(1)

		learning_rates.append(optimizer.param_groups[0]["lr"])
		scheduler.step()
		avg_train_loss = total_train_loss / len(train_loader)
		logger.info(f"[train] Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_train_loss:.20f}")

		train_losses.append(avg_train_loss)
		# wandb.log({"train/loss": avg_train_loss, "epoch": epoch + 1})

		encoder_fr.eval()
		encoder_it.eval()
		decoder_fr.eval()
		decoder_it.eval()

		total_val_loss = 0.0

		with torch.no_grad():
			for (input_fr, input_it, label) in val_loader:
				input_fr = input_fr.to(device)
				input_it = input_it.to(device)

				hidden_fr = encoder_fr.detach_hidden(encoder_fr.init_hidden(batch_size, device))
				hidden_it = encoder_it.detach_hidden(encoder_it.init_hidden(batch_size, device))

				# computing embeddings from encoders
				hidden_fr = encoder_fr(input_fr, hidden_fr)
				hidden_it = encoder_it(input_it, hidden_it)

				# constrastive loss with label
				cl_loss = contrastive_loss(hidden_fr[0], hidden_it[0], label=label, device=device)
				cl_loss += contrastive_loss(hidden_fr[1], hidden_it[1], label=label, device=device)

				output_it, _ = decoder_it(input_it, hidden_it)
				output_fr, _ = decoder_fr(input_fr, hidden_fr)

				reconstruction_loss = loss_fn(output_it.reshape(batch_size * input_it.shape[1], -1),
											  input_it.reshape(-1).long())
				reconstruction_loss += loss_fn(output_fr.reshape(batch_size * input_fr.shape[1], -1),
											   input_fr.reshape(-1).long())

				loss = cl_loss * alpha + reconstruction_loss * beta

				total_val_loss += loss.item()

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

	bleu_score_fr, bleu_score_it = generate(config, test_loader, model_file, vocab_fr, vocab_it)

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
