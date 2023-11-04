import pandas as pd
import spacy
import torchtext
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from dao.Dataset import LSTMDataset

spacy.load('fr_core_news_sm')
spacy.load('it_core_news_sm')


# spacy.load('de_core_news_sm')
# spacy.load('en_core_web_sm')

def prepare_dataset(corpus: pd.DataFrame, config: dict):
	logger.info("[prepare_dataset] preparing dataset")
	# modify configuration

	train, test = train_test_split(corpus, test_size=0.2, random_state=42)
	train, val = train_test_split(train, test_size=0.1, random_state=42)

	logger.info(f"[split_dataset] len train: {int(0.7 * len(corpus))} | len validation: {int(0.1 * len(corpus))} | len "
				f"test: "
				f"{int(0.2 * len(corpus))}")

	train = train.reset_index(drop=True)
	test = test.reset_index(drop=True)
	val = val.reset_index(drop=True)

	col = train.columns.values
	train_dataset = LSTMDataset(corpus_fr=train[col[0]], corpus_it=train[col[1]])
	train_loader = DataLoader(train_dataset,
							  batch_size=config.get('batch_size'),
							  drop_last=True,
							  shuffle=True,
							  pin_memory=True,
							  num_workers=8)

	val_dataset = LSTMDataset(corpus_fr=val[col[0]], corpus_it=val[col[1]], negative_sampling=False)
	val_loader = DataLoader(val_dataset,
							batch_size=config.get('batch_size'),
							drop_last=True,
							shuffle=True,
							pin_memory=True)

	test_dataset = LSTMDataset(corpus_fr=test[col[0]], corpus_it=test[col[1]], negative_sampling=False)
	test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True, shuffle=True, pin_memory=True)

	return train_loader, val_loader, test_loader


def sequence2index(corpus: pd.DataFrame, vocab: Vocab):
	encoded = []
	for sentence in corpus:
		encoded.append([vocab[token] for token in sentence])
	return encoded


def create_vocab_and_dataset(corpus: pd.DataFrame):
	# generating vocab from text file
	logger.info("[create_vocab] creating vocabulary")
	tokenizer_fr = torchtext.data.utils.get_tokenizer('spacy', language="fr_core_news_sm")
	tokenizer_it = torchtext.data.utils.get_tokenizer('spacy', language="it_core_news_sm")

	tokenized_dataset_fr = corpus['french'].apply(lambda x: tokenizer_fr(x))
	tokenized_dataset_it = corpus['italian'].apply(lambda x: tokenizer_it(x))

	vocab = torchtext.vocab.build_vocab_from_iterator(pd.concat([tokenized_dataset_it, tokenized_dataset_fr],
																axis=0).sample(frac=1), max_tokens=80000)
	vocab.insert_token('<pad>', 0)
	vocab.insert_token('<eos>', 1)
	vocab.insert_token('<unk>', 2)
	vocab.set_default_index(vocab['<unk>'])

	tokenized_dataset_it = tokenized_dataset_it.apply(lambda x: x + ["<eos>"])
	tokenized_dataset_fr = tokenized_dataset_fr.apply(lambda x: x + ["<eos>"])

	corpus_tokenized = pd.DataFrame({
		'french' : sequence2index(tokenized_dataset_fr, vocab),
		'italian': sequence2index(tokenized_dataset_it, vocab)})

	corpus_tokenized = corpus_tokenized[
		corpus_tokenized.apply(lambda row: len(row['french']) <= 100 and len(row['italian']) <= 100, axis=1)]

	return corpus_tokenized, vocab
