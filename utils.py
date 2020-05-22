import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import TensorDataset

# nltk.download('punkt')
class Dataset() :
    def __init__(self, vocab_file) : #TODO format of vocab? 
        #self.vocab
        pass

    def word2index(self, sentence) :
        for i in range(len(sentence)): sentence[i] = sorted(self.vocab.keys()).index(sentence[i])
        return sentence
    
    def max_length(self, sentences): #the sentences object is a list of lists 
        lengths = [len(sentence) for sentence in sentences]
        return max(lengths), lengths
        
    def get_tokens(self, sentences, pad): #here the sentences object is list of texts, pad is the pad token TODO: decide on a pad token
        sentences = word_tokenize(sentences)
        maxlen, lengths = self.max_length(sentences)
        for i in range(len(sentences)):
            sentences[i] = [self.word2index(word) for word in sentences[i]]
            sentences[i] += [pad]*(maxlen-len(lengths[i]))
        return sentences, lengths

    def read_data(self, PATH) :
        df_train = pd.read_csv('./data/train.csv') 
        df_dev = pd.read_csv('./data/dev.csv')
        df_test = pd.read_csv('./data/test.csv')
        
        train_tokens, train_lengths = self.get_tokens(df_train['text'], pad) # convert words in sentences to tokens using word2index function then pad to max_len
        dev_tokens, dev_lengths = self.get_tokens(df_dev['text'], pad)
        test_tokens, test_lengths = self.get_tokens(df_test['text'], pad)
        train_tokens, dev_tokens, test_tokens, train_lengths, dev_lengths, test_lengths, train_target, dev_target ,test_target = torch.as_tensor(train_tokens), torch.as_tensor(dev_tokens), torch.as_tensor(test_tokens), torch.as_tensor(train_lengths), torch.as_tensor(dev_lengths), torch.as_tensor(test_lengths), torch.as_tensor(list(zip(df_train['V'], df_train['A'], df_train['D']))), torch.as_tensor(list(zip(df_dev['V'], df_dev['A'], df_dev['D']))), torch.as_tensor(list(zip(df_test['V'], df_test['A'], df_test['D']))) # longest line of code I've ever written :P

        return TensorDataset(train_tokens, train_lengths, train_target), TensorDataset(dev_tokens, dev_lengths, dev_target), TensorDataset(test_tokens, test_lengths, test_target)
        # create a tensor of target and tokens
        # append it to a list

        #use torch.cat to convert list of tensors to 2D tensor:
        # sentences_size : (num of sentences X max_len)
        # source_lengths: (num of sentences X 1)
        # target_size: (num of senteces X 3)

        