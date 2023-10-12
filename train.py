from typing import Any

import ray
import torch.utils.data
from loguru import logger
from matplotlib import pyplot as plt
from ray import train
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm

from dao.Model import Decoder, Encoder
from utils.utils import save_model


def contrastive_loss(x1: torch.Tensor, x2: torch.Tensor, label: int, margin: int = 1.0) -> torch.Tensor:
	"""
	The contrastive_loss function takes in two tensors, x_i and x_j, which are the embeddings of two images. It also
	takes in a label y (either 0 or 1) that indicates whether the images are from the same class(1) or not (0). The
	function then computes a distance between these embeddings: If y is 1 (the images are from the same class),
	we want this distance to be small; if it's 0 (the images aren't from the same class), we want this distance to be
	over the margin

	:param x1: Represent the first image in the pair
	:param x2: Calculate the distance between x2 and x3
	:param label: Determine whether the two images are similar(1) or not(0)
	:param margin: Define the threshold for when a pair of images is considered similar
	:return: The mean of the loss for each pair
	"""
	label = torch.Tensor(label)
	dist = torch.nn.functional.pairwise_distance(x1, x2)
	loss = label * torch.pow(dist, 2) + (1 - label) * torch.pow(torch.clamp(margin - dist, min=0.0, max=None), 2)
	return torch.mean(loss)


def train_autoencoder(config,
					  train_loader: Any,
					  val_loader: Any,
					  model_file: str,
					  plot_file: str,
					  optimize: bool = False) -> None:
	if optimize:
		train_loader = ray.get(train_loader)
		val_loader = ray.get(val_loader)

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	logger.info(f"[train] device: {device}")

	# training parameters
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
	output_dim_it = config['output_dim_it']
	output_dim_fr = config['output_dim_fr']
	dec_dropout = config['dec_dropout']

	# model instantiation
	encoder_fr = Encoder(len_vocab_fr, embedding_dim, hidden_dim, num_layers, enc_dropout).to(device)

	encoder_it = Encoder(len_vocab_it, embedding_dim, hidden_dim, num_layers, enc_dropout).to(device)

	decoder_fr = Decoder(len_vocab_fr, embedding_dim, hidden_dim, output_dim_fr, num_layers, dec_dropout).to(device)

	decoder_it = Decoder(len_vocab_it, embedding_dim, hidden_dim, output_dim_it, num_layers, dec_dropout).to(device)

	# optimizer
	optimizer = torch.optim.Adam(list(encoder_fr.parameters()) + list(encoder_it.parameters()) + list(
		decoder_fr.parameters()) + list(
		decoder_it.parameters()), lr=lr, weight_decay=5e-4)

	train_losses = []
	val_losses = []

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
				break
				input_fr = input_fr.to(device)
				input_it = input_it.to(device)

				optimizer.zero_grad()

				hidden_fr = encoder_fr(input_fr)
				hidden_it = encoder_it(input_it)

				cl_loss = contrastive_loss(hidden_fr[-1, :, :], hidden_it[-1, :, :], label=label)

				output_fr = decoder_fr(input_fr, hidden_fr)
				output_it = decoder_it(input_it, hidden_it)

				output_it = output_it.view(-1, len_vocab_it)
				output_fr = output_fr.view(-1, len_vocab_fr)
				input_it = input_it.view(-1).long()
				input_fr = input_fr.view(-1).long()

				reconstruction_loss = loss_fn(output_it, input_it)
				reconstruction_loss += loss_fn(output_fr, input_fr)

				loss = cl_loss + reconstruction_loss

				loss.backward()

				optimizer.step()

				total_train_loss += loss.item()

				pbar.set_postfix({"Train Loss": loss.item()})

				pbar.update(1)

			avg_train_loss = total_train_loss / len(train_loader)
			logger.info(f"[train] Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_train_loss:.20f}")

			train_losses.append(avg_train_loss)
			# wandb.log({"train/loss": avg_train_loss, "epoch": epoch + 1})

			encoder_fr.eval()
			encoder_it.eval()
			decoder_fr.eval()
			decoder_it.eval()

			total_val_loss = 0.0

			test_loss = MSELoss()

			with torch.no_grad():
				for (input_fr, input_it, label) in val_loader:
					input_fr = input_fr.to(device)
					input_it = input_it.to(device)

					hidden_fr = encoder_fr(input_fr)
					hidden_it = encoder_it(input_it)

					cl_loss = contrastive_loss(hidden_fr[-1, :, :], hidden_it[-1, :, :], label=label)

					output_fr = decoder_fr(input_fr, hidden_fr)
					output_fr = torch.argmax(torch.softmax(output_fr, dim=2), dim=2)

					output_it = decoder_it(input_it, hidden_it)
					output_it = torch.argmax(torch.softmax(output_it, dim=2), dim=2)

					reconstruction_loss = test_loss(output_it, input_it)
					reconstruction_loss += test_loss(output_fr, input_fr)

					loss = cl_loss + reconstruction_loss

					total_val_loss += loss.item()

			avg_val_loss = total_val_loss / len(val_loader)
			logger.info(f"[train] Validation Loss: {avg_val_loss:.20f}")

			if optimize:
				train.report({'loss': avg_val_loss, 'train/loss': avg_train_loss})

			# wandb.log({"val/loss": avg_val_loss, "epoch": epoch + 1})

			val_losses.append(avg_val_loss)

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

			plt.figure(figsize=(10, 6))
			plt.plot(train_losses, label="Train Loss")
			plt.plot(val_losses, label="Validation Loss")
			plt.xlabel("Epoch")
			plt.ylabel("Loss")
			plt.title("Training and Validation Loss")
			plt.legend()
			plt.grid(True)
			plt.tight_layout()
			plt.savefig(plot_file.format(file_name="train_val_loss"))
			plt.close()

	# wandb.finish()
