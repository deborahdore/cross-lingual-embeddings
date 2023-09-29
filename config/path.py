import os
from pathlib import Path

# ------- download ------- #
download_corpus = "https://www.statmt.org/europarl/v7/{lang}-en.tgz"

# ------- directories ------- #
base_dir = os.path.abspath(".")
dataset_dir = os.path.join(base_dir, "dataset")
processed_dataset_dir = os.path.join(dataset_dir, "processed")
model_dir = os.path.join(base_dir, "models")
plot_dir = os.path.join(base_dir, "plot")
config_dir = os.path.join(base_dir, "config")
tokenizer_dir = os.path.join(base_dir, "word2vec")

# ------- files ------- #
lang_file = os.path.join(dataset_dir, "{lang}/europarl-v7.{lang}-en.{lang}")
eng_lang_file = os.path.join(dataset_dir, "{lang}/europarl-v7.{lang}-en.en")
aligned_file = os.path.join(processed_dataset_dir, "dataset_aligned.csv")
processed_file = os.path.join(processed_dataset_dir, "dataset_processed.csv")
vocab_file = os.path.join(processed_dataset_dir, "vocab_{lang}.txt")
model_file = os.path.join(model_dir, "{type}.pt")
plot_file = os.path.join(plot_dir, "{file_name}.svg")
model_config_file = os.path.join(config_dir, "model_config.json")
embedding_model = os.path.join(tokenizer_dir, "{lang}-tokenizer.model")

# ------- create missing directories ------- #
Path(dataset_dir).mkdir(parents=True, exist_ok=True)
Path(processed_dataset_dir).mkdir(parents=True, exist_ok=True)
Path(model_dir).mkdir(parents=True, exist_ok=True)
Path(plot_dir).mkdir(parents=True, exist_ok=True)
Path(config_dir).mkdir(parents=True, exist_ok=True)
Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
