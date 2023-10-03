class Vocab:
	def __init__(self):
		self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
		self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}
		self.n = 3

	def add_sentence(self, sentence):
		for word in sentence.split():
			self.add_word(word)

	def add_word(self, word):
		self.word2idx[word] = self.n
		self.idx2word[self.n] = word
		self.n += 1

	def get_max_length(self):
		return self.n
