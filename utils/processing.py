import csv
import re
import string

import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from utils.utils import read_file_to_df, write_df_to_file


def align_dataset(fr_file: str, eng_fr_file: str, it_file: str, eng_it_file: str, aligned_file: str):
	"""
	The align_dataset function takes in four files:
		- fr_file: a file containing French sentences, one per line.
		- eng_fr_file: a file containing English translations of the French sentences, one per line.
		- it_file: a file containing Italian sentences, one per line.
		- eng_it_file: a file containing English translations of the Italian sentences, one per line.

	:param fr_file: str: Specify the path to the french file
	:param eng_fr_file: str: Specify the path to the english-french file
	:param it_file: str: Specify the path to the italian sentences file
	:param eng_it_file: str: Specify the file containing the english-italian sentences
	:param aligned_file: str: Specify the file where the aligned dataset will be saved
	:return: A csv file with the aligned sentences
	"""
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
	"""
	The nlp_pipeline function takes a pandas Series of text as input and returns the same pandas Series with all words
	lowercased, numbers removed, special characters removed (except for ? ! and '), extra spaces removed,
	and sentences ending in . ! or ? followed by a space.

	:param corpus: pd.Series: Pass the dataframe column to the function
	:return: A series of strings
	"""
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
	"""
	The remove_outliers function takes in a dataframe and returns the same dataframe with outliers removed.
		Outliers are defined as rows where either French or Italian sentences have more than 100 words, or less than
		10 words.
		The function also adds two columns to the inputted dataframe: l2_word_count and l2_word_count, which count the
		number of
		words in each sentence.

	:param df: pd.DataFrame: Specify the dataframe that is being passed into the function
	:return: A dataframe with outliers removed
	"""

	def count_words(sentence) -> int:
		"""
		 The count_words function takes a string as input and returns the number of words in that string.

		 :param sentence: Pass in the sentence that we want to count the words of
		 :return: The number of words in the sentence
		 """
		words = sentence.split()
		return len(words)

	cols = df.columns.values

	df['l1_word_count'] = df[cols[0]].apply(count_words)
	df['l2_word_count'] = df[cols[1]].apply(count_words)

	# Filter out rows where either French or Italian sentences have more than 100 words
	# Filter out rows where either French or Italian sentences have less than 10 words
	filtered_df = df[(df['l1_word_count'] < 100) & (df['l2_word_count'] < 100) & (df['l1_word_count'] >= 10) & (
			df['l2_word_count'] >= 10)].copy()

	filtered_df.drop(['l1_word_count', 'l2_word_count'], axis=1, inplace=True)

	return filtered_df


def eda(corpus: pd.DataFrame, plot_file: str):
	"""
	The eda function performs exploratory data analysis on the corpus.

	:param corpus: pd.DataFrame: Pass the dataframe to the function
	:param plot_file: str: Save the plot as a png file
	"""
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
	"""
	The process_dataset function takes in the aligned file and processed file as input.
	It then reads the aligned_file into a dataframe, drops any rows with missing values,
	drops duplicates and resets the index. It then runs exploratory data analysis on this
	dataframe to get some basic statistics about it (number of sentences per language, etc.).
	The function also removes outliers from this dataset by removing sentences over a certain length.
	Then it runs preprocessing on both languages in order to clean up each sentence before writing them out to disk.

	:param aligned_file: str: Specify the file containing the aligned sentences
	:param processed_file: str: Specify the location of the processed dataset
	:param plot_file: str: Save the plot to a file
	:return: A pandas dataframe
	"""
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
	"""
	The get_until_eos function takes a phrase and a vocabulary as input.
	It then finds the index of the end-of-sentence token in that phrase,
	and returns everything up to that point.

	:param phrase: Store the phrase that is being translated
	:param vocab: Get the index of &lt;eos&gt;
	:return: A phrase until the end of sentence token
	"""
	eos = vocab['<eos>']
	if eos in phrase:
		phrase = phrase[:phrase.index(eos)]
	return phrase
