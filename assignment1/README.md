# Assignment 1: Deep Average Network (DAN) for Text Classification


In this assignment, I implemented **word dropout** at the input and **neural dropout** at the hidden layers for DAN. Detailed implementations can be found in the `model.py` file.


## Highlights:
- Setting word dropout rate to 0.3 (recommended in the [original paper](https://www.aclweb.org/anthology/P15-1162.pdf)) did not yield any improvement in classification performance.
- Initializing word embeddings with pretrained embeddings showed a significant improvement in results.
- The selected hyperparameters for the experiments are stored in the `run_exp.sh` file.

### Experiments and Results:

#### 1. Stanford Sentiment Treebank (SST)
- **Embeddings**: [GloVe](http://nlp.stanford.edu/data/glove.42B.300d.zip) (`glove.42B.300d.txt`)
- **Results**: 
  - Dev set:
    - Without GloVe: 0.4151
    - With GloVe: 0.4278
  - Test set:
    - Without GloVe: 0.4371
    - With GloVe: 0.4493

**Observation**: There was a 3.71% improvement over the baseline test performance (0.4122) when using GloVe embeddings.

#### 2. IMDb Reviews
- **Embeddings**: [FastText](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip) (`crawl-300d-2M.vec`)
- **Results**:
  - Dev set:
    - Without FastText: 0.9265
    - With FastText: 0.9469

**Observation**: There was a 2.45% improvement over the baseline dev performance (0.9224) when using FastText embeddings.

### Resources:
For easy access and setup, the `setup.py` script is provided. This script:
- Checks if the required pretrained embeddings are already downloaded.
- If not, it fetches the embeddings from the specified URL and then unzips the downloaded files.

### Training Logs:
Two separate log files record the training process:
1. `log_without_pretrained_WE.txt`: Log for training without using pretrained word embeddings.
2. `log_with_pretrained_WE.txt`: Log for training initialized with pretrained word embeddings.
