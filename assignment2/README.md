# Assignment 2: minBERT for sentence classification 


In this assignment, I implemented the **transformer model (BERT)**, **linear projection layer** for classification and **AdamW** optimizer. Detailed implementations can be found in `bert.py`, `classifier.py` and `optimizer.py`.


### Experiments and Results:

#### 1. Stanford Sentiment Treebank (SST)
- **Results**: 
  - Dev set:
    - option `pretrain` (only finetune the last linear projection layer): 0.409
    - option `finetune` (unfreeze all layers for finetuning): 0.525
  - Test set:
    - option `pretrain`: 0.429
    - option `finetune`: 0.529


#### 2. CFIMDB 
- **Results**:
  - Dev set:
    - option `pretrain`: 0.792
    - option `finetune`: 0.963
  - Test set:
    - option `pretrain`: 0.504
    - option `finetune`: 0.512


### Training Logs:
Two separate log files record the training process:
1. `sst-train-log-finetune.txt`: Log for training the model on the SST dataset with `finetune` option. 
2. `cfimdb-train-log-finetune.txt`: Log for training the model on the CFIMDB dataset with `finetune` option.
3. `sst-train-log-pretrain.txt`: Log for training the model on the SST dataset with `pretrain` option.
4. `cfimdb-train-log-pretrain.txt`: Log for training the model on the CFIMDB dataset with `pretrain` option. 
