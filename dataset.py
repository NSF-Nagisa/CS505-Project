import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.7
VALID_SIZE = 0.15

class BayesDataset():
    def __init__(self, data_path, seed_value):
        df = pd.read_csv(data_path+'cleaned_data.csv')
        self.labels2idx = {k: i for i, k in enumerate(df['labels'].unique())}
        self.idx2labels = {k: self.labels2idx[k] for k in self.labels2idx.keys()}
        X = df['cleaned_text']
        y = df.labels.apply(lambda x: self.labels2idx[x]).values
        total_size = len(X)
        train_size = int(TRAIN_SIZE * total_size)
        valid_size = int(VALID_SIZE * total_size)
        test_size = total_size - train_size - valid_size
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, train_size=(train_size+valid_size), test_size=test_size, stratify=y, random_state=seed_value)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, train_size=train_size, test_size=valid_size, stratify=y_train, random_state=seed_value)

class LSTMDataset():
    def __init__(self, data_path, seed_value):
        df = pd.read_csv(data_path+'cleaned_data.csv')
        self.labels2idx = {k: i for i, k in enumerate(df['labels'].unique())}
        self.idx2labels = {k: self.labels2idx[k] for k in self.labels2idx.keys()}
        all_words = ' '.join(df['cleaned_text']).split()
        self.word2idx = {word: i+1 for i, word in enumerate(set(all_words))}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        max_len = np.max([len(text.split()) for text in df.cleaned_text])
        X = self.Tokenize(df.cleaned_text, self.word2idx, max_len)
        y = df.labels.apply(lambda x: self.labels2idx[x]).values
        total_size = len(X)
        train_size = int(TRAIN_SIZE * total_size)
        valid_size = int(VALID_SIZE * total_size)
        test_size = total_size - train_size - valid_size
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, train_size=(train_size+valid_size), test_size=test_size, stratify=y, random_state=seed_value)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, train_size=train_size, test_size=valid_size, stratify=y_train, random_state=seed_value)
        
    def Tokenize(column, word2idx, seq_len):
        # tokenize the columns text using the dictionary
        tokenized = []
        for text in column:
            row = [word2idx[word] for word in text.split()]
            tokenized.append(row)
        # add padding to tokens
        tokens = np.zeros((len(tokenized), seq_len), dtype = int)
        for i, text in enumerate(tokenized):
            zeros = list(np.zeros(seq_len - len(text)))
            new = zeros + text
            tokens[i, :] = np.array(new)
        return tokens
    
    
class BertDataset():
    def __init__(self, data_path, seed_value):
        df = pd.read_csv(data_path+'cleaned_data.csv')
        self.labels2idx = {k: i for i, k in enumerate(df['labels'].unique())}
        self.idx2labels = {k: self.labels2idx[k] for k in self.labels2idx.keys()}
        X = df['cleaned_text'].values
        y = df.labels.apply(lambda x: self.labels2idx[x]).values
        total_size = len(X)
        train_size = int(TRAIN_SIZE * total_size)
        valid_size = int(VALID_SIZE * total_size)
        test_size = total_size - train_size - valid_size
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, train_size=(train_size+valid_size), test_size=test_size, stratify=y, random_state=seed_value)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, train_size=train_size, test_size=valid_size, stratify=y_train, random_state=seed_value)