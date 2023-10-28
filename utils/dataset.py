import pandas as pd
import spacy
import torchtext
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from dao.Dataset import LSTMDataset

spacy.load('fr_core_news_sm')
spacy.load('it_core_news_sm')


def collate_fn(batch):
	# Sort the batch in descending order of the length of French sentences
	batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

	# Separate the French sentences, Italian sentences, and labels
	french_sentences, italian_sentences, labels = zip(*batch)

	# Pad French and Italian sentences with zeros to the length of the longest sentence in the batch
	padded_french_sentences = pad_sequence(french_sentences, batch_first=True, padding_value=0)
	padded_italian_sentences = pad_sequence(italian_sentences, batch_first=True, padding_value=0)

	return padded_french_sentences, padded_italian_sentences, labels


def split_dataset(corpus: pd.DataFrame, batch_size: int):
	train, test = train_test_split(corpus, test_size=0.2, random_state=42)
	train, val = train_test_split(train, test_size=0.1, random_state=42)

	logger.info(f"[split_dataset] len train: {0.7 * len(corpus)} | len validation: {0.1 * len(corpus)} | len test: "
				f"{0.2 * len(corpus)}")

	train = train.reset_index(drop=True)
	test = test.reset_index(drop=True)
	val = val.reset_index(drop=True)

	train_dataset = LSTMDataset(corpus_fr=train[0], corpus_it=train[1])
	train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
							  collate_fn=collate_fn)

	val_dataset = LSTMDataset(corpus_fr=val[0], corpus_it=val[1], negative_sampling=False)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn)

	test_dataset = LSTMDataset(corpus_fr=test[0], corpus_it=test[1], negative_sampling=False)
	test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True, shuffle=True, collate_fn=collate_fn)

	return train_loader, val_loader, test_loader


def prepare_dataset(corpus: pd.DataFrame, config: dict):
	logger.info("[prepare_dataset] preparing dataset")
	# modify configuration

	train, test = train_test_split(corpus, test_size=0.2, random_state=42)
	train, val = train_test_split(train, test_size=0.1, random_state=42)

	logger.info(f"[split_dataset] len train: {0.7 * len(corpus)} | len validation: {0.1 * len(corpus)} | len test: "
				f"{0.2 * len(corpus)}")

	train = train.reset_index(drop=True)
	test = test.reset_index(drop=True)
	val = val.reset_index(drop=True)

	col = train.columns.values
	train_dataset = LSTMDataset(corpus_fr=train[col[0]], corpus_it=train[col[1]])
	train_loader = DataLoader(train_dataset,
							  batch_size=config.get('batch_size'),
							  drop_last=True,
							  shuffle=True,
							  collate_fn=collate_fn)

	val_dataset = LSTMDataset(corpus_fr=val[col[0]], corpus_it=val[col[1]], negative_sampling=False)
	val_loader = DataLoader(val_dataset,
							batch_size=config.get('batch_size'),
							drop_last=True,
							shuffle=True,
							collate_fn=collate_fn)

	test_dataset = LSTMDataset(corpus_fr=test[col[0]], corpus_it=test[col[1]], negative_sampling=False)
	test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True, shuffle=True, collate_fn=collate_fn)

	return train_loader, val_loader, test_loader


def create_vocab(corpus: pd.DataFrame):
	# generating vocab from text file
	logger.info("[create_vocab] creating vocabulary")
	tokenizer_fr = torchtext.data.utils.get_tokenizer('spacy', language="fr_core_news_sm")
	tokenizer_it = torchtext.data.utils.get_tokenizer('spacy', language="it_core_news_sm")

	tokenized_dataset_fr = corpus['french'].apply(lambda x: tokenizer_fr(x))
	tokenized_dataset_it = corpus['italian'].apply(lambda x: tokenizer_it(x))

	tokenized_dataset = pd.concat([tokenized_dataset_it, tokenized_dataset_fr], axis=0).sample(frac=1)  # shuffle

	vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset, max_tokens=60000)
	vocab.insert_token('<pad>', 0)
	vocab.insert_token('<eos>', 1)
	vocab.insert_token('<unk>', 2)

	vocab.set_default_index(vocab['<unk>'])

	tokenized_dataset_it = tokenized_dataset_it.apply(lambda x: x + ["<eos>"])
	tokenized_dataset_fr = tokenized_dataset_fr.apply(lambda x: x + ["<eos>"])

	return tokenized_dataset_fr, tokenized_dataset_it, vocab


def sequence2index(corpus: pd.DataFrame, vocab: Vocab):  # end of sentence
	encoded = []
	for sentence in corpus:
		encoded.append([vocab[token] for token in sentence])
	return encoded
