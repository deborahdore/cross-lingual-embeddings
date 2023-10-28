import csv
import re
import string

import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from utils.utils import read_file_to_df, write_df_to_file


def align_dataset(fr_file: str, eng_fr_file: str, it_file: str, eng_it_file: str, aligned_file: str):
	logger.info("[align_dataset] creating aligned dataset")

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


def nlp_pipeline(corpus: pd.Series):
	###### run nlp pipeline

	logger.info("[nlp_pipeline] lower words")
	corpus = corpus.str.lower()

	logger.info("[nlp_pipeline] transform numbers")
	corpus = corpus.apply(lambda x: re.sub(r'\d+', '', x))

	logger.info("[nlp_pipeline] remove special characters")
	exclude = set(string.punctuation)  # - {'?', '!', '\''}
	corpus = corpus.apply(lambda x: ''.join([word for word in x if word not in exclude]))
	corpus = corpus.apply(lambda x: re.sub(r"([.!?])", r" \1", x))
	corpus = corpus.apply(lambda x: re.sub(r'\.\.\.', '', x))

	logger.info("[nlp_pipeline] remove extra spaces")
	corpus = corpus.apply(lambda x: x.strip())
	corpus = corpus.apply(lambda x: re.sub(" +", " ", x))
	corpus = corpus.apply(lambda x: x.strip())

	return corpus


def remove_outliers(df: pd.DataFrame):
	def count_words(sentence) -> int:
		words = sentence.split()
		return len(words)

	df['french_word_count'] = df['french'].apply(count_words)
	df['italian_word_count'] = df['italian'].apply(count_words)

	# Filter out rows where either French or Italian sentences have more than 100 words
	# Filter out rows where either French or Italian sentences have less than 10 words
	filtered_df = df[
		(df['french_word_count'] <= 100) & (df['italian_word_count'] <= 100) & (df['french_word_count'] >= 10) & (
				df['italian_word_count'] >= 10)].copy()

	filtered_df.drop(['french_word_count', 'italian_word_count'], axis=1, inplace=True)

	return filtered_df


def eda(corpus: pd.DataFrame, plot_file: str):
	###### exploratory data analysis
	logger.info(f"[eda] corpus is composed of {len(corpus)} sentences")  # 1.665.523

	corpus = corpus.reset_index(drop=True, allow_duplicates=False)

	list_lengths_fr = corpus['french'].apply(lambda x: len(x.split()))
	list_lengths_it = corpus['italian'].apply(lambda x: len(x.split()))

	pd.DataFrame({'italian': list_lengths_it, 'french': list_lengths_fr}).boxplot()
	plt.title('french - italian Sentences Length distribution')
	plt.tight_layout()
	plt.savefig(plot_file.format(file_name="fr_it_sentences_length"))
	plt.close()


def process_dataset(aligned_file: str, processed_file: str, plot_file: str):
	logger.info("[process_dataset] processing dataset")
	original_corpus = read_file_to_df(aligned_file)
	original_corpus = original_corpus.dropna().drop_duplicates().reset_index(drop=True)

	# exploratory data analysis
	eda(original_corpus, plot_file)

	logger.info("[process_dataset] remove outliers")
	# remove sentences over a certain length
	original_corpus = remove_outliers(original_corpus)

	# run preprocessing
	french = nlp_pipeline(original_corpus['french'])
	italian = nlp_pipeline(original_corpus['italian'])

	final_corpus = pd.DataFrame({'french': pd.Series(french), 'italian': pd.Series(italian)})
	final_corpus = final_corpus.dropna().reset_index(drop=True)
	write_df_to_file(final_corpus, processed_file)

	return final_corpus


def get_until_eos(phrase, vocab):
	eos = vocab['<eos>']
	if eos in phrase:
		phrase = phrase[:phrase.index(eos)]
	return phrase
