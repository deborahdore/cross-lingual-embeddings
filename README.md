# cross-lingual-embeddings

_The goal of this project is to create embeddings for italian and french that are aligned in a shared latent space using
an encoder-decoder model_

## **DATASET**

The parallel datasets Italian-English and French-English was obtained from the [_European Parliament Proceedings
Parallel Corpus 1996-2011_](https://www.statmt.org/europarl/). These dataset were already aligned. <br> To obtain an
Italian-English Corpus, the Italian-English and the French-English corpus were joined based on their english sentences
in common. It's worth noting that the two corpora did not have the same number of sentences due to different
translations. In fact, during the European Parliament session, not all languages are translated to english and then
translated back to the target language. Some languages are directly translated from source to target without going
through english first. This creates misalignment in the corpus. Therefore, some loss of information is expected. <br>

* original dataset can be found [here](dataset/fr) and [here](dataset/it)
* aligned dataset can be found [here](dataset/processed/dataset_aligned.csv)
* post processed dataset (cleaned and tokenized) can be found [here](dataset/processed/dataset_preprocessed.csv)

Note that due to the highest volume of words in the corpus (> 800k), the vocabulary of each language contains a large
number of words (> 1 million) therefore, after different trials, it was decided to use BPETokenizer. Byte Pair Encoding
is a subword tokenization technique that breaks down text into smaller subword units, making it especially useful for
handling languages with complex morphologies and handling out-of-vocabulary words.

## **MODELS**

Trained models are available in this [folder](models)

* Italian encoder: [model](models/encoder_it.pt)
* French encoder: [model](models/encoder_fr.pt)
* Italian decoder: [model](models/decoder_it.pt)
* French decoder: [model](models/decoder_fr.pt)
* Latent Space Model: [model](models/latent_space.pt)

<figure style="text-align: center;"><img src="plot/architecture.svg" alt="Architecture of the models"><figcaption>Architecture</figcaption></figure>
