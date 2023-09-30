import json
import os
import tarfile
import urllib.request
from typing import Any, Tuple, Union

import pandas as pd
import torch

from dao.model import Decoder, Encoder, LatentSpace
from loguru import logger


def read_json(json_file_name: str) -> json:
	logger.info("[read_json] loading json file")
	with open(json_file_name, 'r') as file:
		data = json.load(file)
	file.close()
	return data


def write_json(json_obj: json, json_file_name: str) -> None:
	logger.info("[write_json] writing json file")
	with open(json_file_name, 'w') as file:
		file.write(json.dumps(json_obj, indent=4))
	file.close()


def read_file(file: str) -> str:
	logger.info(f"[read_file] reading {file}")
	with open(file, "r") as f:
		corpus = f.read()
	f.close()
	return corpus


def read_file_to_df(file: str) -> pd.DataFrame:
	logger.info(f"[read_file_to_df] reading {file}")
	df = pd.read_csv(file)
	df = df.dropna().drop_duplicates().reset_index(drop=True)
	return df


def write_df_to_file(df: Union[pd.DataFrame, pd.Series], file: str) -> None:
	df.to_csv(file, index=False)


def download_from_url(url: str, to_save: str, lang: str) -> None:
	logger.info(f"[download_from_url] downloading from {url}")
	# download
	to_save_with_name = os.path.join(to_save, url.split("/")[-1])
	urllib.request.urlretrieve(url, to_save_with_name)
	# unzip
	with tarfile.open(to_save_with_name, "r:gz") as file:
		file.extractall(os.path.join(to_save, lang))

	os.remove(to_save_with_name)
	logger.info(f"[download_from_url] file saved to {to_save}")


def save_model(model: Union[Encoder, Decoder, LatentSpace], file: str) -> None:
	torch.save(model.state_dict(), file)


def load_model(file: str) -> Any:
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	return torch.load(file, map_location=device)
