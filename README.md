# cross-lingual-embeddings

_The goal of this project is to create embeddings for italian and french that are aligned in a shared latent space using
an encoder-decoder model_

## **REPLICABILITY**

Each experiment was conducted on a single machine by running the [main](main.py) script and specifying whether to
process the dataset or not, whether to start the optimization pipeline or not and whether to conduct an ablation study
or not (only one between optimization and ablation study can be true):

`python main.py --generate True --optimize True --ablation False # process dataset, optimize model, skip ablation study` <br>
`python main.py --generate False --optimize False --ablation True # skip processing and optimization step, do ablation study`

### Environment

- Python 3.9
- Python dependencies: [requirements.txt](requirements.txt)

### Data Processing Script

- [utils/processing.py](utils/processing.py) - processing functions
- [utils/utils.py](utils/utils.py) - utils functions
- [utils/dataset.py](utils/dataset.py) dataset processing functions

### Training & Testing Script

- [main.py](main.py) - main script
- [train.py](train.py) - training loop
- [test.py](test.py) - evaluation loop
- [optimization.py](optimization.py) - optimizations function
- [study.py](study.py) - ablation study

### Results

- [plot](plot) - visual results
- [ablation_study](ablation_study) - ablation study results

### Configurations

- [model_config.json](config/model_config.json) - model's configurations
- [dao/Model.py](dao/Model.py) - model skeleton
- [dao/Dataset.py](dao/Model.py) - dataset skeleton
- [loss.py](utils/loss.py) - contrastive loss definition

## **DATASET**

The parallel datasets Italian-English and French-English was obtained from the [_European Parliament Proceedings
Parallel Corpus 1996-2011_](https://www.statmt.org/europarl/). These dataset were already aligned. <br> To obtain an
Italian-English Corpus, the Italian-English and the French-English corpus were joined based on their english sentences
in common. It's worth noting that the two corpora did not have the same number of sentences due to different
translations. In fact, during the European Parliament session, not all languages are translated to english and then
translated back to the target language. Some languages are directly translated from source to target without going
through english first. This creates misalignment in the corpus. Therefore, some loss of information is expected. <br>

Due to the extensive volume of words within our corpus, exceeding 800,000, each language's vocabulary encompasses a
substantial number of words, surpassing one million in total. The vocabulary was used to create embeddings for the
sentences by converting each word into their corresponding index in the vocabulary <br>

This conversion technique can pose challenges for neural network training, as it may struggle to achieve effective
reconstruction when dealing with such embeddings with such large numbers. Conversely, normalizing these embeddings can
result in exceedingly small values (e.g., 0.000000003), rendering them impractical for reliable model reconstruction.
Therefore, only the most frequent 30000 words were chosen to be included in the vocabulary.

An alternative solution that was explored was to employ techniques like GloVe or Word2Vec.
However, it's essential to note that these methods are inherently lossy algorithms, which can make the task of
reconstructing the original sentence in natural language more challenging.

<img src="plot/fr_it_sentences_length.svg">

## **MODEL'S ARCHITECTURE**
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

## **LATENT SPACE PROJECTION**

<img src="plot/latent_space_projection.svg">


