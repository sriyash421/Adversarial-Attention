class Dataset() :
    def __init__(self, vocab_file) :
        #self.vocab
        pass

    def word2index(self, sentence) :
        #convert words in a single sentence to index using sorted(self.vocab.keys()).index(word)
        pass
    
    def max_length(self, sentences) :
        #function to get max length of sentences
        #return max_len and list of lengths
        pass
        
    def read_data(self, PATH) :
        pass
        #read csv

        # convert words in sentences to tokens using word2index function then pad to max_len
        # create a tensor of target and tokens
        # append it to a list

        #use torch.cat to convert list of tensors to 2D tensor:
        # sentences_size : (num of sentences X max_len)
        # source_lengths: (num of sentences X 1)
        # target_size: (num of senteces X 3)

        #return TensorDataset(sentences, source_lengths, target_size)