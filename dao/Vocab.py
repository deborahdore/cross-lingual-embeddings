import nltk


class Vocab:
	def __init__(self):
		self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
		self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
		self.n = 3

	def add_sentence(self, sentence, language):
		for word in nltk.word_tokenize(sentence, language=language):
			self.add_word(word)

	def add_word(self, word):
		self.word2idx[word] = self.n
		self.idx2word[self.n] = word
		self.n += 1

	def get_vocab_len(self):
		return self.n

	def to_word(self, index):
		return self.idx2word.get(index, '<UNK>')

	def to_index(self, word):
		return self.word2idx.get(word, 3)
