import torch.nn


def contrastive_loss(x1: torch.Tensor, x2: torch.Tensor, label: torch.Tensor, margin: int = 1.0) -> torch.Tensor:
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
	:param device: device
	"""
	label = label.unsqueeze(1)
	dist = torch.nn.functional.pairwise_distance(x1, x2)
	loss = label * torch.pow(dist, 2) + (1 - label) * torch.pow(torch.clamp(margin - dist, min=0.0, max=None), 2)
	return torch.mean(loss)
