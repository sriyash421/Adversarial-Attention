import re
import csv
import pandas as pd
from nltk.tokenize import word_tokenize

data_df = pd.read_csv('./data/emobank.csv')
data_df = data_df.to_dict(orient="records")
lengths = []
for i in data_df:
    sentence = i['text']
    # urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', sentence)
    # for url in urls: sentence = sentence.replace(url,'')
    words = [word.lower() for word in word_tokenize(sentence)]
    lengths.append(len(words))