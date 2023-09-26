import torch.utils.data
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config.logger import logger
from dao import AEDataset
from dao.model import Encoder, LatentSpace, Decoder
from utils.utils import read_json, save_model


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

	dist = torch.nn.functional.pairwise_distance(x1, x2)
	loss = label * torch.pow(dist, 2) + (1 - label) * torch.pow(torch.clamp(margin - dist, min=0.0, max=None), 2)
	return torch.mean(loss)


def split_dataset(dataset: AEDataset, batch_size: int) -> [DataLoader, DataLoader, DataLoader]:
	train_size = int(0.7 * len(dataset))
	val_size = int(0.10 * len(dataset))
	test_size = len(dataset) - train_size - val_size

	logger.info(f"[split_dataset] len train: {train_size} | len validation: {val_size} | len test: {test_size}")

	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

	return train_loader, val_loader, test_loader


def train(dataset: AEDataset, model_config_file: str, model_file: str, plot_file: str) -> DataLoader:
	config = read_json(model_config_file)
	logger.info(f"[train] {config}")

	# training parameters
	patience = config['patience']
	early_stop_counter = config['early_stop_counter']
	best_val_loss = float('inf')
	batch_size = config['batch_size']
	num_epochs = config['num_epochs']
	lr = config['lr']
	best_val_loss = float('inf')

	# model parameters
	len_vocab_fr = config['len_vocab_fr']
	len_vocab_it = config['len_vocab_it']
	embedding_dim = config['embedding_dim']
	hidden_dim = config['hidden_dim']
	ls_dim = config['ls_dim']
	hidden_lstm_dim = config['hidden_lstm_dim']
	output_dim = config['output_dim']

	train_loader, val_loader, test_loader = split_dataset(batch_size, dataset)

	# model instantiation
	encoder_fr = Encoder(len_vocab_fr, embedding_dim, hidden_dim)
	encoder_it = Encoder(len_vocab_it, embedding_dim, hidden_dim)

	latent_space = LatentSpace(hidden_dim, ls_dim)

	decoder_fr = Decoder(ls_dim, hidden_lstm_dim, output_dim)
	decoder_it = Decoder(ls_dim, hidden_lstm_dim, output_dim)

	optimizer = torch.optim.Adam(
		list(encoder_fr.parameters()) + list(encoder_it.parameters()) + list(latent_space.parameters()) + list(
			decoder_fr.parameters()) + list(decoder_it.parameters()), lr=lr, weight_decay=1e-8)

	train_losses = []
	val_losses = []

	mse_loss = MSELoss()

	# init weight&biases feature
	# wandb.init(project="cross-lingual-it-fr-embeddings-3-loss",
	# 		   config=config}

	for epoch in range(num_epochs):

		encoder_fr.train()
		encoder_it.train()
		decoder_fr.train()
		decoder_it.train()
		latent_space.train()

		total_train_loss = 0.0

		with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
			for batch_idx, (input_fr, input_it, label) in enumerate(train_loader):
				optimizer.zero_grad()

				embedding_fr = latent_space(encoder_fr(input_fr))
				embedding_it = latent_space(encoder_it(input_it))

				loss = contrastive_loss(embedding_fr, embedding_it, label=label)

				output_fr = decoder_fr(embedding_fr)
				output_it = decoder_it(embedding_it)

				loss += mse_loss(output_it, input_it) + mse_loss(output_fr, input_fr)

				loss.backward()

				max_grad_norm = 1.0
				torch.nn.utils.clip_grad_norm_(list(encoder_fr.parameters()) + list(encoder_it.parameters()) + list(
					decoder_fr.parameters()) + list(decoder_it.parameters()), max_grad_norm)

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
		latent_space.eval()

		total_val_loss = 0.0

		with torch.no_grad():
			for (input_fr, input_it, label) in val_loader:
				embedding_fr = latent_space(encoder_fr(input_fr))
				embedding_it = latent_space(encoder_it(input_it))

				loss = contrastive_loss(embedding_fr, embedding_it, label=label)

				output_fr = decoder_fr(embedding_fr.detach())
				output_it = decoder_it(embedding_it.detach())

				loss += mse_loss(output_it, input_it) + mse_loss(output_fr, input_fr)

				total_val_loss += loss.item()

		avg_val_loss = total_val_loss / len(val_loader)
		logger.info(f"[train] Validation Loss: {avg_val_loss:.20f}")
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
			save_model(latent_space, file=model_file.format(type="latent_space"))

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
	plt.savefig(plot_file.format("train_val_loss"))
	plt.close()

	# wandb.finish()

	return test_loader