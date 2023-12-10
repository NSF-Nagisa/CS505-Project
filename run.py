import os
import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from collections import Counter

import transformers
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix

from model import BiLSTMWithAttention, Bert_Classifier
from dataset import BayesDataset, LSTMDataset, BertDataset

def conf_matrix(y, y_pred, title, labels):
  sns.set(font_scale=1.5)
  fig, ax =plt.subplots(figsize=(7.5,7.5))
  ax=sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="YlGnBu", fmt='g', cbar=False, annot_kws={"size":30})
  plt.title(title, fontsize=20)
  ax.xaxis.set_ticklabels(labels, fontsize=10)
  ax.yaxis.set_ticklabels(labels, fontsize=10)
  ax.set_ylabel('True label', fontsize=20)
  ax.set_xlabel('Predicted', fontsize=20)
  plt.show()

def report(y_test, y_pred, labels):
    print('Classification Report:\n',classification_report(y_test, y_pred, target_names=labels))
    conf_matrix(y_test,y_pred,'Confusion Matrix', labels)
    
def load_glove_model(file):
    print("Loading Glove Model")
    glove_model = {}
    with open(file, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float32)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def bert_tokenizer(data):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
          text=sent,
          add_special_tokens=True,  # Add `[CLS]` and `[SEP]` special tokens
          max_length=max_len,  # Choose max length to truncate/pad
          padding='max_length',  # Pad sentence to max length
          return_attention_mask=True  # Return attention mask
          )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

def evaluate_lstm_model(model, test_loader):
    model.eval()
    y_pred_list = []
    y_test_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            y_pred_test = torch.argmax(output, dim=1)
            y_pred_list.extend(y_pred_test.squeeze().tolist())
            y_test_list.extend(labels.squeeze().tolist())

    return y_pred_list, y_test_list

def evaluate_bert_model(model, test_loader):
    model.eval()
    y_pred_list = []
    y_test_list = []

    with torch.no_grad():
        for inputs, masks, labels in test_loader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            output = model(inputs, masks)
            y_pred_test = torch.argmax(output, dim=1)
            y_pred_list.extend(y_pred_test.squeeze().tolist())
            y_test_list.extend(labels.squeeze().tolist())

    return y_pred_list, y_test_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--hidden', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--model_path', type=str, default='model.pt')
    parser.add_argument('--warmup_epoch', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'bert', 'bayes'])

    args = parser.parse_args()

    batch_size = args.batch_size
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if args.model_type == 'lstm':
        dataset = LSTMDataset(args.data_path, args.seed)
        
        embedding_file = args.data_path+'glove.6B.200d.txt'
        embedding_dim = 200
        glove_model = load_glove_model(embedding_file)
        # create start-up embedding matrix
        embedding_matrix = np.zeros((len(dataset.word2idx) + 1, embedding_dim))
        for word, i in dataset.word2idx.items():
            embedding_vector = glove_model.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
        # dataset and dataloader
        train_data = TensorDataset(torch.from_numpy(dataset.X_train), torch.from_numpy(dataset.y_train))
        test_data = TensorDataset(torch.from_numpy(dataset.X_test), torch.from_numpy(dataset.y_test))
        valid_data = TensorDataset(torch.from_numpy(dataset.X_valid), torch.from_numpy(dataset.y_valid))

        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
        
        num_classes = len(dataset.labels2idx)
        hidden_dim = args.hidden
        lstm_layers = args.layers
        dropout = args.dropout
        weight_decay = args.weight_decay
        lr = args.base_learning_rate
        epochs = args.total_epoch
        warmup_epoch = args.warmup_epoch
        
        model = BiLSTMWithAttention(embedding_matrix, hidden_dim, num_classes, lstm_layers, dropout)
        criterion = nn.NLLLoss()
        # learning rate decreases along the training process
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)
        lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / epochs * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=False)
        early_stopping_patience = 10
        early_stopping_counter = 0

        max_valid_acc = 0

        for e in range(epochs):
        
            # train model
            losses = []
            correct = 0
            total = 0
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                #get tensor of predicted values on the training set
                y_pred_train = torch.argmax(output, dim=1)

                correct += torch.sum(y_pred_train==labels).item()
                total += labels.size(0)

                #update learning rate
                lr_scheduler.step()
            avg_loss = np.mean(losses)
            acc = 100 * correct / total
            print(f'In epoch {e+1}, average traning loss is {avg_loss}, accuracy is {acc:.4f}%.')

            # validate model
            valid_losses = []
            valid_correct = 0
            valid_total = 0
            with torch.no_grad():
            
                model.eval()
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model(inputs)
                    valid_loss = criterion(output, labels)
                    valid_losses.append(valid_loss.item())
                    y_pred_val = torch.argmax(output, dim=1)
                    valid_correct += torch.sum(y_pred_val==labels).item()
                    valid_total += labels.size(0)
                valid_avg_loss = np.mean(valid_losses)
                valid_acc = 100 * valid_correct / valid_total

                print(f'In epoch {e+1}, average validation loss is {valid_avg_loss}, accuracy is {valid_acc:.4f}%.')

            #Save model if validation accuracy increases
            if valid_acc >= max_valid_acc:
                torch.save(model.state_dict(), args.data_path + 'lstm_model/state_dict.pt')
                valid_acc_max = valid_acc
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter > early_stopping_patience:
                print('Early stopped at epoch :', e+1)
                break
        
        model.load_state_dict(torch.load(args.data_path + 'lstm_model/state_dict.pt'))
        
        y_pred_list, y_test_list = evaluate_lstm_model(model, test_loader)
        
        report(y_test_list, y_pred_list, dataset.idx2labels.keys())
        
    elif args.model_type == 'bert':
        dataset = BertDataset(args.data_path, args.seed)
        # baseline model: Naive Bayes

        vectorizer = CountVectorizer()
        X_train_cv =  vectorizer.fit_transform(dataset.X_train)
        X_test_cv = vectorizer.transform(dataset.X_test)

        # use TF-IDF to calculate the weight of each word based on its frequency
        tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
        X_train_tf = tf_transformer.transform(X_train_cv)
        
        X_test_tf = tf_transformer.transform(X_test_cv)
        model = MultinomialNB()
        model.fit(X_train_tf, dataset.y_train)

        nb_pred = model.predict(X_test_tf)
        
        report(dataset.y_test, nb_pred, dataset.idx2labels.keys())
        
    elif args.model_type == 'bayes':
        dataset = BayesDataset(args.data_path, args.seed)
        # first we load the pretrained bert tokenizer in transformer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        max_len = 128
        
        train_inputs, train_masks = bert_tokenizer(dataset.X_train)
        val_inputs, val_masks = bert_tokenizer(args.X_valid)
        test_inputs, test_masks = bert_tokenizer(args.X_test)
        
        train_labels = torch.Tensor(dataset.y_train)
        val_labels = torch.Tensor(dataset.y_valid)
        test_labels = torch.Tensor(dataset.y_test)
        
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        
        model = Bert_Classifier(freeze_bert=False).to(device)
        epochs = args.total_epoch
        optimizer = optim.AdamW(model.parameters(), lr=4e-5, eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        # use the learning rate scheduler in transformers
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

        early_stopping_patience = 10
        early_stopping_counter = 0

        max_valid_acc = 0

        for e in range(epochs):
        
            # train model
            losses = []
            correct = 0
            total = 0
            model.train()
            for inputs, masks, labels in train_dataloader:
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.type(torch.LongTensor).to(device)
                optimizer.zero_grad()
                output = model(inputs, masks)
                loss = criterion(output, labels)
                losses.append(loss.item())
                loss.backward()

                # clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                #get tensor of predicted values on the training set
                y_pred_train = torch.argmax(output, dim=1)

                correct += torch.sum(y_pred_train==labels).item()
                total += labels.size(0)

                #update learning rate
                lr_scheduler.step()
                
            avg_loss = np.mean(losses)
            acc = 100 * correct / total
            print(f'In epoch {e+1}, average traning loss is {avg_loss}, accuracy is {acc:.4f}%.')

            # validate model
            valid_losses = []
            valid_correct = 0
            valid_total = 0
            with torch.no_grad():
        
                model.eval()
                for inputs, masks, labels in val_dataloader:
                    inputs, masks, labels = inputs.to(device), masks.to(device), labels.type(torch.LongTensor).to(device)
                    output = model(inputs, masks)
                    valid_loss = criterion(output, labels)
                    valid_losses.append(valid_loss.item())
                    y_pred_val = torch.argmax(output, dim=1)
                    valid_correct += torch.sum(y_pred_val==labels).item()
                    valid_total += labels.size(0)
                valid_avg_loss = np.mean(valid_losses)
                valid_acc = 100 * valid_correct / valid_total

                print(f'In epoch {e+1}, average validation loss is {valid_avg_loss}, accuracy is {valid_acc:.4f}%.')

            #Save model if validation accuracy increases
            if valid_acc >= max_valid_acc:
                torch.save(model.state_dict(), args.data_path + 'bert_model/state_dict.pt')
                valid_acc_max = valid_acc
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter > early_stopping_patience:
                print('Early stopped at epoch :', e+1)
                break
            
        model.load_state_dict(torch.load(args.data_path + 'bert_model/state_dict.pt'))
        
        y_pred_list, y_test_list = evaluate_bert_model(model, test_dataloader)
        
        report(y_test_list, y_pred_list, dataset.idx2labels.keys())