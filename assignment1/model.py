import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    #raise NotImplementedError()
    
    print(f'Loading embedding file: {emb_file}')
    emb = np.zeros((len(vocab), emb_size), dtype=np.float32)
    # read "glove.42B.300d/glove.42B.300d.txt"
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            word = line[0]
            if word in vocab.word2id:
                idx = vocab.word2id[word]
                emb[idx][:] = np.array(line[1:]).astype(np.float32)

    return emb


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size, padding_idx=0)
        fc = [nn.Dropout(self.args.emb_drop), nn.Linear(self.args.emb_size, self.args.hid_size, bias=True), nn.ReLU()] 
        if self.args.hid_layer > 2:
            for _ in range(self.args.hid_layer-2):
                fc += [nn.Dropout(self.args.hid_drop), nn.Linear(self.args.hid_size, self.args.hid_size, bias=True), nn.ReLU()]  
        fc += [nn.Dropout(self.args.hid_drop), nn.Linear(self.args.hid_size, self.tag_size, bias=True)] 
        self.fc = nn.Sequential(*fc)


    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        # initalize weights and bias
        if isinstance(self.embedding, nn.Embedding):
            nn.init.xavier_uniform_(self.embedding.weight.data)
        
        if isinstance(self.fc, nn.Sequential):
            for layer in self.fc:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data)
                    nn.init.constant_(layer.bias.data, 0)
         
    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(emb))

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        # three 300-d ReLu layers,word dropout probability p=0.3, L2 regularization weight = 1e-5 applied to all parameters

        rw = torch.distributions.bernoulli.Bernoulli(1-self.args.word_drop).sample((x.shape[1], )) # x.shape[1] = seq_length
        x = x[:, rw==1] # word dropout (does not work well)

        x = self.embedding(x) # [batch_size, seq_length, emb_size]
        if self.args.pooling_method == "sum":
            x = torch.sum(x, dim=1)
        elif self.args.pooling_method == "avg":
            x = torch.mean(x, dim=1)
        elif self.args.pooling_method == "max":
            x = torch.max(x, dim=1)
        else:
            raise Exception("Pooling method not applicable")

        scores = self.fc(x)
        return scores

