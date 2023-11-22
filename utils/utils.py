import json
import os
import tarfile
import urllib.request
from typing import Any, Union

import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt

from dao.Model import Decoder, Encoder, SharedSpace


def read_json(json_file_name: str) -> json:
	"""
	The read_json function reads a json file and returns the data as a json object.


	:param json_file_name: str: Specify the name of the file that is being read
	:return: A json object
	"""
	logger.info("[read_json] loading json file")
	with open(json_file_name, 'r') as file:
		data = json.load(file)
	file.close()
	return data


def write_json(json_obj: json, json_file_name: str) -> None:
	"""
	The write_json function takes a json object and writes it to a file.

	:param json_obj: json: Specify the type of data that is being passed into the function
	:param json_file_name: str: Specify the name of the json file to be written
	"""
	logger.info("[write_json] writing json file")
	with open(json_file_name, 'w') as file:
		file.write(json.dumps(json_obj, indent=4))
	file.close()


def read_file(file: str) -> str:
	"""
	The read_file function reads a file and returns the contents as a string.


	:param file: str: Pass in the file name
	:return: A string
	"""
	logger.info(f"[read_file] reading {file}")
	with open(file, "r") as f:
		corpus = f.read()
	f.close()
	return corpus


def read_file_to_df(file: str) -> pd.DataFrame:
	"""
	The read_file_to_df function reads a file into a pandas dataframe.

	:param file: str: Specify the file to be read
	:return: A pandas dataframe
	"""
	logger.info(f"[read_file_to_df] reading {file}")
	df = pd.read_csv(file)
	df = df.dropna().drop_duplicates().reset_index(drop=True)
	return df


def write_df_to_file(df: Union[pd.DataFrame, pd.Series], file: str) -> None:
	"""
	The write_df_to_file function takes a pandas DataFrame or Series and writes it to the specified file.

	:param df: Specify the type of dataframe that is being passed into the function
	:param file: str: Specify the file name and location to write the dataframe to
	"""
	df.to_csv(file, index=False)


def download_from_url(url: str, to_save: str, lang: str) -> None:
	"""
	The download_from_url function downloads a file from the given url and saves it to the specified location.
		Args:
			url (str): The URL of the file to download.
			to_save (str): The path where we want to save our downloaded file.


	:param url: str: Specify the url of the file to download
	:param to_save: str: Specify the directory to save the downloaded file
	:param lang: str: Specify the language of the data to download
	"""
	logger.info(f"[download_from_url] downloading from {url}")
	# download
	to_save_with_name = os.path.join(to_save, url.split("/")[-1])
	urllib.request.urlretrieve(url, to_save_with_name)
	# unzip
	with tarfile.open(to_save_with_name, "r:gz") as file:
		file.extractall(os.path.join(to_save, lang))

	os.remove(to_save_with_name)
	logger.info(f"[download_from_url] file saved to {to_save}")


def save_model(model: Union[Encoder, Decoder, SharedSpace], file: str) -> None:
	"""
	The save_model function saves the state of a model to a file.

	:param model: Union[Encoder, Decoder, SharedSpace]: Specify the type of model that is being saved
	:param file: str: Specify the name of the file to save the model to

	"""
	torch.save(model.state_dict(), file)


def load_model(file: str) -> Any:
	"""
	The load_model function loads a model from the specified file.

	:param file: str: Specify the path to the file containing the model
	:return: A model
	"""
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	return torch.load(file, map_location=device)


def save_plot(plt_elems, x_label, y_label, title, filename):
	"""
	The save_plot function takes in a list of plot elements, an x-axis label,
	a y-axis label, a title for the plot and a filename to save the figure as.
	It then creates a new figure with the given dimensions and plots each element
	in plt_elems using matplotlib's pyplot module. It then adds labels to both axes
	and sets the title of the plot before adding it to legend. Finally it saves
	the figure as an image file with name filename.

	:param plt_elems: Pass in a list of tuples
	:param x_label: Label the x-axis of the graph
	:param y_label: Label the y-axis of the plot
	:param title: Set the title of the plot
	:param filename: Save the plot to a file
	:return: The filename of the saved plot
	"""
	plt.figure(figsize=(10, 6))
	for elem in plt_elems:
		plt.plot(elem[1], label=elem[0])
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()
