import pandas as pd 
import gensim, pickle, nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class dataset:
    def __init__(self, PATH): #read pickle file for vocab
        self.vocab = pickle.load(PATH, 'rb') #data structure to be decided

    def get_embedding(self): #voacb can be accessed by self.vocab TODO: add gensim fasttext code to load the vectors
        pass

    def split(self, df): #input is dataframe, output is list of dictionary elements
        train, dev, test = [], [], []
        dict_list = df.to_dict(orient='records')
        for Dict in dict_list:
            if Dict['split'] == 'test': test.append(Dict)
            elif Dict['split'] == 'dev': dev.append(Dict)
            else: train.append(Dict)
        print('train:{} dev:{} test:{}'.format(len(train), len(dev), len(test)))
        return train, dev, test

    def tokenize(self, sentence): #tokenize a sentence supplied as a string and get list of tokens
        self.ps = PorterStemmer()
        punctuations = '''!()-[]{};.:'"\,<>/?@#$%^&*_~''' #gives contol over what punctuation to retain  
        for x in sentence.lower(): 
            if x in punctuations: 
                sentence = sentence.replace(x, "") 
        sentence = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', sentence) #remove urls
        string = ''.join(i for i in sentence if ord(i)<128) #remove non unicode characters
        words = word_tokenize(sentence) #### TODO Class importing pe load hoga kya, if word_tokenize is not imported
        for i in range(len(words)): words[i] = self.ps(words[i])
        return words

    def text_target(self, dict_list): #takes in a list of dictionaries and returns the text target tupple
        data = []
        for i in range(len(dict_list)):
            data.append((' '.join(w for w in self.tokenize(dict_list['text'][i])), [dict_list[i]['V'], dict_list[i]['A'], dict_list[i]['D']]))    
        return data

    def get_vocab(self, df):
        vocab = []
        for sentence in df['text']:
            vocab += self.tokenize(sentence)
        vocab = sorted(list(set(vocab)))
        return vocab

    def load_corpus(self, PATH='./EmoBank/corpus/emobank.csv'): #return matrix with word2vec embedding for each word in the sentence, splitting is also done here
        self.data_df = pd.read_csv(PATH)
        self.vocab = self.get_vocab(self.data_df)
        print('Vocabulary has {} words. There are {} rows in the data.'.format(len(self.vocab), len(self.data_df)))
        self.train_dl, self.dev_dl, self.test_dl = split(self.data_df) #dl means a list of dictionaries
        self.train, self.dev, self.test = self.text_target(self.train_dl), self.target(self.dev_dl), self.target(self.test_dl)
        return self.train, self.dev, self.test, self.vocab

