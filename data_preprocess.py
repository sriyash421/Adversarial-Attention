import pandas as pd 
import gensim, pickle, nltk, re
# from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
class LoadData:
    def __init__(self, PATH):
        # self.vocab = pickle.load(PATH, 'rb') #data structure to be decided
        self.train, self.dev, self.test, self.vocab = self.load_corpus(PATH)
        df_train = pd.DataFrame(self.train, columns=['text','V','A','D'])
        df_train.to_csv('./data/train.csv')
        df_dev = pd.DataFrame(self.dev, columns=['text','V','A','D'])
        df_dev.to_csv('./data/dev.csv')
        df_test = pd.DataFrame(self.test, columns=['text','V','A','D'])
        df_test.to_csv('./data/test.csv')
        with open('./data/vocab.txt', 'w') as f:
            for word in self.vocab: f.write(word+'\n')
            f.close()
        print('train:{} dev:{} test:{}'.format(len(df_train), len(df_dev), len(df_test)))
        print('length of vocabulary is {}'.format(len(self.vocab)))

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
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', sentence)
        for url in urls: sentence = sentence.replace(url,'')
        sentence = sentence.encode('ascii', 'ignore').decode('ascii') #better for removing non unicode characters
        words = [word.lower() for word in word_tokenize(sentence)] 
        return words

    def text_target(self, dict_list): #takes in a list of dictionaries and returns the text target tupple
        data = []
        for i in range(len(dict_list)):
            tokens = self.tokenize(dict_list[i]['text'])
            if len(tokens) > 0 :
                data.append([' '.join([w for w in tokens]), dict_list[i]['V'], dict_list[i]['A'], dict_list[i]['D']])    
        return data

    def get_vocab(self, df):
        vocab = []
        for sentence in df['text']:
            vocab += self.tokenize(sentence)
        vocab = sorted(list(set(vocab)))
        return vocab

    def load_corpus(self, PATH=''): #return matrix with word2vec embedding for each word in the sentence, splitting is also done here
        self.data_df = pd.read_csv(PATH)
        self.vocab = self.get_vocab(self.data_df)
        print('Vocabulary has {} words. There are {} rows in the data.'.format(len(self.vocab), len(self.data_df)))
        self.train_dl, self.dev_dl, self.test_dl = self.split(self.data_df) #dl means a list of dictionaries
        self.train, self.dev, self.test = self.text_target(self.train_dl), self.text_target(self.dev_dl), self.text_target(self.test_dl)
        return self.train, self.dev, self.test, self.vocab

d = LoadData('./data/emobank.csv')