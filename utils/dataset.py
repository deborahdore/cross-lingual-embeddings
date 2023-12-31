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


def prepare_dataset(corpus: pd.DataFrame, config: dict):
	"""
	The prepare_dataset function takes a pandas dataframe and a configuration dictionary as input.
	It splits the corpus into train, validation and test sets. It then creates three DataLoader objects:
	one for training, one for validation and one for testing. The DataLoaders are used to create batches of
	data that can be fed into the model during training/validation/testing.

	:param corpus: pd.DataFrame: Pass the dataframe containing the corpus to be used for training
	:param config: dict: Pass the configuration to the function
	:return: A tuple of 3 dataloaders: train, val, test
	"""
	logger.info("[prepare_dataset] preparing dataset")
	# modify configuration

	train, test = train_test_split(corpus, test_size=0.2, random_state=42)
	train, val = train_test_split(train, test_size=0.1, random_state=42)

	logger.info(f"[split_dataset] len train: {int(0.7 * len(corpus))} | len validation: {int(0.1 * len(corpus))} | "
				f"len test:{int(0.2 * len(corpus))}")

	train = train.reset_index(drop=True)
	test = test.reset_index(drop=True)
	val = val.reset_index(drop=True)

	col = train.columns.values
	train_dataset = LSTMDataset(corpus_fr=train[col[0]], corpus_it=train[col[1]])
	train_loader = DataLoader(train_dataset,
							  batch_size=config.get('batch_size'),
							  pin_memory=True,
							  drop_last=True,
							  shuffle=True)

	val_dataset = LSTMDataset(corpus_fr=val[col[0]], corpus_it=val[col[1]], negative_sampling=False)
	val_loader = DataLoader(val_dataset,
							batch_size=config.get('batch_size'),
							pin_memory=True,
							drop_last=True,
							shuffle=True)

	test_dataset = LSTMDataset(corpus_fr=test[col[0]], corpus_it=test[col[1]], negative_sampling=False)
	test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True, drop_last=True, shuffle=True)

	return train_loader, val_loader, test_loader


def sequence2index(corpus: pd.DataFrame, vocab: Vocab):
	"""
	The sequence2index function takes a corpus of sentences and a vocabulary as input.
	It returns the encoded version of the corpus, where each word is replaced by its index in the vocabulary.

	:param corpus: pd.DataFrame: Pass in the corpus, which is a list of lists
	:param vocab: Vocab: Get the index of a word in the vocabulary
	:return: A list of lists, where each sublist is a sequence of integers
	"""
	encoded = []
	for sentence in corpus:
		encoded.append([vocab[token] for token in sentence])
	return encoded


def create_vocab_and_dataset(corpus: pd.DataFrame):
	# generating vocab from text file
	"""
	The create_vocab_and_dataset function takes a pandas dataframe as input and returns a tokenized dataset,
	a French vocabulary and an Italian vocabulary. The function first creates two tokenizers for the French and
	Italian languages using spaCy. It then uses these to create two vocabularies from the corpus, one for each language.
	The function then adds special tokens  <pad>, <eos> and <unk> to both vocabularies before setting their default index
	to be that of the unknown word token (i.e., 2). Finally, it appends an end-of-sentence tag <eos>

	:param corpus: pd.DataFrame: Pass the dataframe to the function
	:return: A tokenized dataset and two vocabularies
	"""
	logger.info("[create_vocab] creating vocabulary")
	tokenizer_fr = torchtext.data.utils.get_tokenizer('spacy', language="fr_core_news_sm")
	tokenizer_it = torchtext.data.utils.get_tokenizer('spacy', language="it_core_news_sm")

	tokenized_dataset_fr = corpus['french'].apply(lambda x: tokenizer_fr(x))
	tokenized_dataset_it = corpus['italian'].apply(lambda x: tokenizer_it(x))

	vocab_fr = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset_fr, max_tokens=30000)
	vocab_fr.insert_token('<pad>', 0)
	vocab_fr.insert_token('<eos>', 1)
	vocab_fr.insert_token('<unk>', 2)
	vocab_fr.set_default_index(vocab_fr['<unk>'])

	vocab_it = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset_it, max_tokens=30000)
	vocab_it.insert_token('<pad>', 0)
	vocab_it.insert_token('<eos>', 1)
	vocab_it.insert_token('<unk>', 2)
	vocab_it.set_default_index(vocab_it['<unk>'])

	tokenized_dataset_it = tokenized_dataset_it.apply(lambda x: x + ["<eos>"])
	tokenized_dataset_fr = tokenized_dataset_fr.apply(lambda x: x + ["<eos>"])

	corpus_tokenized = pd.DataFrame({
		'french' : sequence2index(tokenized_dataset_fr, vocab_fr),
		'italian': sequence2index(tokenized_dataset_it, vocab_it)})

	corpus_tokenized = corpus_tokenized[
		corpus_tokenized.apply(lambda row: len(row['french']) <= 100 and len(row['italian']) <= 100, axis=1)]

	corpus_tokenized['french'] = corpus_tokenized['french'].apply(lambda x: x[:100] + [0] * (100 - len(x)))
	corpus_tokenized['italian'] = corpus_tokenized['italian'].apply(lambda x: x[:100] + [0] * (100 - len(x)))

	return corpus_tokenized, vocab_fr, vocab_it
