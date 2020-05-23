import nltk
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import trange
import torch.nn as nn
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset

# nltk.download('punkt')
class Dataset() :
    def __init__(self, vocab_file="./data/vocab.pkl") : #TODO format of vocab? 
        self.vocab = None
        self.embedding_size = 300
        with open(vocab_file, "rb") as fin :
            self.vocab = pickle.load(fin)
        self.vocab['<pad>'] = np.zeros(self.embedding_size)
        self.vocab_keys = sorted(self.vocab.keys())
        self.padding_idx = self.vocab_keys.index("<pad>")
        self.vocab_length = len(self.vocab.keys())

    def word2index(self, sentence) :
        sentence = torch.LongTensor([self.vocab_keys.index(word) for word in sentence])
        return sentence
    
    def max_length(self, sentences): #the sentences object is a list of lists 
        lengths = [len(sentence) for sentence in sentences]
        return max(lengths), lengths
        
    def get_tokens(self, sentences): #here the sentences object is list of texts, pad is the pad token TODO: decide on a pad token
        sentences = [word_tokenize(str(sent)) for sent in sentences]
        maxlen, lengths = self.max_length(sentences)
        for i in trange(len(sentences)) :
            sentences[i] += ['<pad>']*(maxlen-lengths[i])
            sentences[i] = self.word2index(sentences[i])
        return sentences, lengths

    def read_data(self, PATH) :
        df = pd.read_csv(PATH)
        tokens, lengths = self.get_tokens(df['text'].values)
        target = np.vstack((df['V'].values,df['A'].values,df['D'].values)).T
        tokens = torch.stack(tokens, dim=0)
        lengths = torch.LongTensor(lengths)
        target = torch.from_numpy(target)

        print("Tokens: {} Lengths: {} Target: {}".format(tokens.shape, lengths.shape, target.shape))
        return TensorDataset(tokens, lengths, target)

    def get_vocab(self) :
        embeddings = nn.Embedding(self.vocab_length, self.embedding_size, padding_idx=self.padding_idx)
        temp = []
        for i in self.vocab_keys :
            temp.append(torch.FloatTensor(self.vocab[i]))
        temp = torch.stack(temp, dim=0)
        print("Vocab: {}".format(temp.shape))
        embeddings.weight.data.copy_(temp)
        return embeddings


dataset = Dataset()
vocab = dataset.get_vocab()
train_dataset = dataset.read_data('./data/train.csv') 
dev_dataset = dataset.read_data('./data/dev.csv')
test_dataset = dataset.read_data('./data/test.csv')

temp = {
    'vocab' : vocab,
    'train_dataset': train_dataset,
    'dev_dataset': dev_dataset,
    'test_dataset': test_dataset
}

with open("dataset.pkl", "wb") as fout :
    pickle.dump(temp, fout)
        