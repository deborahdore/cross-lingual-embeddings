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

<img src="plot/fr_it_sentences_length.svg">

## **MODELS**

Trained models are available in this [folder](models)

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1;">
    <ul>
      <li><a href="models/encoder_it.pt">Italian encoder</a></li>
      <li><a href="models/encoder_fr.pt">French encoder</a></li>
      <li><a href="models/decoder_it.pt">Italian decoder</a></li>
      <li><a href="models/decoder_fr.pt">French decoder</a></li>
      <li><a href="models/latent_space.pt">Latent Space Model</a></li>
    </ul>
  </div>
  <div style="flex: 1;">
    <img src="plot/architecture.svg" alt="Architecture">
  </div>
</div>

## **REPLICABILITY**

Each experiment was conducted on a single machine by running the [main](main.py) script and specifying whether to
process the dataset or not:

`python main.py True  # process dataset` <br>
`python main.py False # skip processing step`

### Environment
- Python 3.9
- Python dependencies: [requirements.txt](requirements.txt)

### Data Processing Script
- [utils/processing.py](utils/processing.py) - processing functions
- [utils/utils.py](utils/utils.py) - utils functions
- [dao/AEDataset.py](dao/AEDataset.py) - dataset class
- [embedder](embedder) tokenizers for italian and english using BPE encoding

### Training & Testing Script
- [train.py](train.py) - training loop
- [test.py](test.py) - evaluation loop
- [dao/model.py](dao/model.py) - model skeleton

### Results
- [plot](plot) - visual results


## **LATENT SPACE PROJECTION**
<img src="plot/latent_space_projection.svg">


