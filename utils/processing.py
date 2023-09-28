import csv
import re
import string
from tokenizers import Tokenizer

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tokenizers.implementations import ByteLevelBPETokenizer

from utils.utils import read_file_to_df, write_df_to_file, read_json, write_json
from loguru import logger


def align_dataset(fr_file: str, eng_fr_file: str, it_file: str, eng_it_file: str, aligned_file: str) -> None:
	logger.info("[align_dataset] creating aligned file")

	fr_sentences, fr_en_sentences = fr_file.split('\n'), eng_fr_file.split('\n')
	it_sentences, it_en_sentences = it_file.split('\n'), eng_it_file.split('\n')

	if (len(fr_sentences) != len(fr_en_sentences)) or (len(it_sentences) != len(it_en_sentences)):
		logger.error("[align_corpus] incorrect files")
		raise Exception("Incorrect file")

	fr_mapping = {en: fr for fr, en in zip(fr_sentences, fr_en_sentences)}
	it_mapping = {en: it for it, en in zip(it_sentences, it_en_sentences)}

	joined_sentences = [(fr_mapping[en], it_mapping[en]) for en in fr_en_sentences if
						en in fr_mapping and en in it_mapping]

	with open(aligned_file, 'w') as out:
		csv_out = csv.writer(out)
		csv_out.writerow(['french', 'italian'])
		csv_out.writerows(joined_sentences)

	out.close()
	logger.info(f"[align_dataset] aligned file saved to {aligned_file}")


def nlp_pipeline(corpus: pd.Series) -> pd.Series:
	logger.info("[nlp_pipeline] lower words")
	corpus = corpus.str.lower()

	logger.info("[nlp_pipeline] transform numbers")
	corpus = corpus.apply(lambda x: re.sub(r'\d+', '0', x))

	logger.info("[nlp_pipeline] remove special characters")
	exclude = set(string.punctuation)  # - {'?', '!', '\''}
	corpus = corpus.apply(lambda x: ''.join([word for word in x if word not in exclude]))
	corpus = corpus.apply(lambda x: re.sub(r'\.\.\.', '', x))

	logger.info("[nlp_pipeline] remove extra spaces")
	corpus = corpus.apply(lambda x: x.strip())
	corpus = corpus.apply(lambda x: re.sub(" +", " ", x))
	corpus = corpus.apply(lambda x: x.strip())

	# logger.info("[nlp_pipeline] remove stopwords")
	# stop_words = set(stopwords.words(language))
	# corpus = corpus.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

	return corpus


def create_vocabulary(corpus: pd.Series) -> []:
	logger.info("[create_vocabulary] construct vocabulary")
	vocab = []
	for seq in corpus:
		vocab += [word for word in seq.split() if word not in vocab]

	return sorted(set(vocab))


def seq2idx(corpus: [], embedding_model: str, language: str) -> []:
	logger.info("[seq2idx] converting words to embeddings using ELMO")
	tokenizer = Tokenizer.from_file(embedding_model.format(lang=language))
	vectorized_seqs = [tokenizer.encode(seq).ids for seq in corpus]
	return vectorized_seqs


# def normalize(corpus: [], vocab: []) -> []:
#     logger.info("[normalize] normalize sequence")
#     scaling_factor = 2 / len(vocab)
#     new_corpus = []
#     for seq in corpus:
#         new_corpus.append([(x * scaling_factor) - 1 for x in seq])
#     return new_corpus


def pad_sequence(vectorized: [], maximum_len: int) -> []:
	logger.info(f"[pad_sequence] pad sequence with max length {maximum_len}")

	seq_lengths = list(map(len, vectorized))
	padded_sequences = []

	for seq, seqlen in zip(vectorized, seq_lengths):
		padded_seq = seq + [0] * (maximum_len - seqlen)
		padded_sequences.append(padded_seq)

	return padded_sequences


def count_words(sentence) -> int:
	words = sentence.split()
	return len(words)


def process_dataset(aligned_file: str, processed_file: str, vocab_file: str, model_config_file: str, plot_file: str,
					embedding_model: str) -> tuple[pd.DataFrame, [], []]:
	logger.info("[process_dataset] processing dataset (0.5 sampled)")
	original_corpus = read_file_to_df(aligned_file).sample(frac=0.5)
	original_corpus = original_corpus.dropna().drop_duplicates().reset_index(drop=True)

	eda(original_corpus, plot_file)

	logger.info("[process_dataset] remove outliers")
	original_corpus = remove_outliers(original_corpus)

	french = nlp_pipeline(original_corpus['french'])
	italian = nlp_pipeline(original_corpus['italian'])

	logger.info("[process_dataset] creating vocabulary")
	vocab_fr = create_vocabulary(french)
	vocab_it = create_vocabulary(italian)

	logger.info("[process_dataset] writing vocabulary to file")
	write_df_to_file(pd.Series(vocab_fr, name='words'), vocab_file.format(lang="fr"))
	write_df_to_file(pd.Series(vocab_it, name='words'), vocab_file.format(lang="it"))

	logger.info("[process_dataset] train embedder")
	train_embedder(embedding_model, vocab_file, "fr")
	train_embedder(embedding_model, vocab_file, "it")

	seq2idx_fr = seq2idx(french, embedding_model, "fr")
	seq2idx_it = seq2idx(italian, embedding_model, "it")

	maximum_len = max(len(max(seq2idx_fr, key=len)), len(max(seq2idx_it, key=len)))
	logger.info(f"[process_dataset] max length of sentences is {maximum_len}")

	logger.info("[process_dataset] modify model configurations")
	config = read_json(model_config_file)

	config['output_dim'] = maximum_len
	config['len_vocab_fr'] = len(vocab_fr)
	config['len_vocab_it'] = len(vocab_it)

	write_json(config, model_config_file)

	logger.info("[process_dataset] padding sequence")
	seq2idx_fr_pad = pad_sequence(seq2idx_fr, maximum_len)
	seq2idx_it_pad = pad_sequence(seq2idx_it, maximum_len)

	seq2idx_it_pad_new = []
	seq2idx_fr_pad_new = []
	# to remove sentences that contain only zeros
	for seq_it, seq_fr in zip(seq2idx_it_pad, seq2idx_fr_pad):
		if sum(seq_it) > 0 and sum(seq_fr) > 0:
			seq2idx_it_pad_new.append(seq_it)
			seq2idx_fr_pad_new.append(seq_fr)

	final_corpus = pd.DataFrame({'french': pd.Series(seq2idx_fr_pad_new), 'italian': pd.Series(seq2idx_it_pad_new)})
	final_corpus = final_corpus.dropna().reset_index(drop=True)

	write_df_to_file(final_corpus, processed_file)

	return final_corpus, vocab_fr, vocab_it


def train_embedder(embedding_model: str, vocab: [], lang: str) -> None:
	tokenizer = ByteLevelBPETokenizer()
	tokenizer.train(vocab.format(lang=lang),
					vocab_size=10000,
					special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
	tokenizer.save(embedding_model.format(lang=lang))


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
	df['french_word_count'] = df['french'].apply(count_words)
	df['italian_word_count'] = df['italian'].apply(count_words)

	# Filter out rows where either French or Italian sentences have more than 200 words
	filtered_df = df[(df['french_word_count'] <= 100) & (df['italian_word_count'] <= 100)].copy()

	filtered_df.drop(['french_word_count', 'italian_word_count'], axis=1, inplace=True)

	return filtered_df


def get_sentence_in_natural_language(sentence: torch.Tensor, embedding_model: str, language: str) -> []:
	tokenizer = Tokenizer.from_file(embedding_model.format(lang=language))
	decoded_text = tokenizer.decode([int(x) for x in sentence.data[0].tolist()])
	return decoded_text


def eda(corpus: pd.DataFrame, plot_file: str) -> None:
	logger.info(f"[eda] corpus is composed of {len(corpus)} sentences")  # 1.665.523

	corpus = corpus.reset_index(drop=True, allow_duplicates=False)

	list_lengths_fr = corpus['french'].apply(lambda x: len(x.split()))
	list_lengths_it = corpus['italian'].apply(lambda x: len(x.split()))

	pd.DataFrame({'italian': list_lengths_it, 'french': list_lengths_fr}).boxplot()
	plt.title('french - italian Sentences Length distribution')
	plt.tight_layout()
	plt.savefig(plot_file.format(file_name="fr_it_sentences_length"))
	plt.close()
