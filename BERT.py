# -*- coding: utf-8 -*-
"""Group09QBUS6850_BERT_2024S2.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tS4UWeZ79SdagTBCsuPTXjZXBWdxFP48

# Run Through GPU

# Import data
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install optuna
!pip install shap
!pip install optuna-integration[xgboost]
!pip install torch==2.1.0 torchvision==0.16.0 torchtext==0.16.0

import warnings
warnings.filterwarnings('ignore')

import pickle
import json
import re
import shap
import torch
import spacy
import random
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

import optuna
import xgboost as xgb
import optuna.visualization as vis
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, log_loss

from gensim.models import Word2Vec
import torch.nn as nn
import torch.optim as optim
from torch import utils
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator
import collections
from torch.nn.utils.rnn import pad_sequence

checkpoint = torch.load('/content/drive/MyDrive/6850/checkpoint_for_BERT.pth')

data = pd.read_csv('/content/drive/MyDrive/6850/news-dataset.csv',sep='\t')

seed_value = checkpoint['seed_value']
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_rng_state(checkpoint['rng_state'])
torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
np.random.set_state(checkpoint['numpy_rng_state'])
random.setstate(checkpoint['random_state'])

"""# 1. Data preprocessing"""

data

data.describe()

data.info()

data.nunique()

dup_content = data[data.duplicated(subset='content', keep=False)]
dup_content.head(20)

content_clean = data.drop_duplicates(subset='content')
content_clean.describe()

title_dup = content_clean[content_clean.duplicated(subset='title', keep=False)]
title_dup.head(20)

"""# 2.0 DISTRIBERT


"""

from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset

index_train_b, index_test_b = train_test_split(content_clean.index, stratify=content_clean['category'], train_size=0.7, random_state=42)
train_b = content_clean.loc[index_train_b, :].copy()
test_b = content_clean.loc[index_test_b, :].copy()

index_train1_b, index_valid1_b = train_test_split(train_b.index, stratify=train_b['category'], train_size=0.7, random_state=42)
sub_train_b = train_b.loc[index_train1_b, :].copy()
sub_valid_b = train_b.loc[index_valid1_b, :].copy()

train_for_nn = sub_train_b.copy()
val_for_nn = sub_valid_b.copy()
test_for_nn = test_b.copy()

le = LabelEncoder()

train_for_nn['category'] = train_for_nn['category'].apply(lambda x: x[0])
val_for_nn['category'] = val_for_nn['category'].apply(lambda x: x[0])
test_for_nn['category'] = test_for_nn['category'].apply(lambda x: x[0])

train_target_br = le.fit_transform(train_for_nn['category'])
val_target_br = le.transform(val_for_nn['category'])
test_target_br = le.transform(test_for_nn['category'])


train_data_b = train_for_nn[['category', 'content']]
train_data_b['category'] = train_target_br

val_data_b = val_for_nn[['category', 'content']]
val_data_b['category'] = val_target_br

test_data_b = test_for_nn[['category', 'content']]
test_data_b['category'] = test_target_br

class TextDataset(utils.data.Dataset):
    def __init__(self, myData):
        super().__init__()
        self.data = myData

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data.iloc[idx, 0], self.data.iloc[idx, 1])

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

train_texts = [str(text) for text in train_data_b['content']]
val_texts = [str(text) for text in val_data_b['content']]
test_texts = [str(text) for text in test_data_b['content']]

# encoding text
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# TensorDataset
train_labels = torch.tensor(train_data_b['category'].values)
val_labels = torch.tensor(val_data_b['category'].values)
test_labels = torch.tensor(test_data_b['category'].values)

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

def collate_batch_advanced(batch):
    input_ids_list, attention_mask_list, target_list = [], [], []

    for input_ids, attention_mask, label in batch:
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        target_list.append(label)

    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attention_mask_list)
    target_list = torch.tensor(target_list)

    return input_ids, attention_masks, target_list

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

batchSize = 16
max_length = 512


train_loader = utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_batch_advanced)
val_loader = utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_batch_advanced)
test_loader = utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_batch_advanced)

class SentimentClassifier(nn.Module):
    def __init__(self, hidden_size, dropout_rate, num_classes=5):
        super(SentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, learning_rate, weight_decay, trial_number=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience = 10
    early_stop_counter = 0

    for epoch in range(30):
        model.train()
        total_train_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()


        average_train_loss = total_train_loss / len(train_loader)


        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()


                _, preds = torch.max(outputs, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)


        average_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_samples

        print(f'Epoch [{epoch + 1}/30], Train Loss: {average_train_loss:.4f}, Valid Loss: {average_val_loss:.4f}, Valid Accuracy: {val_accuracy:.4f}')


        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), f'best_model_trial_{trial_number}.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break

    return best_val_loss, best_val_accuracy

def objective(trial):

    hidden_size = trial.suggest_int('hidden_size', 16, 32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)


    model = SentimentClassifier(hidden_size=hidden_size, dropout_rate=dropout_rate)


    best_val_loss, best_val_accuracy = train_model(model, train_loader, val_loader, learning_rate, weight_decay, trial_number=trial.number)


    return best_val_accuracy

# Optuna study

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)


# print("Best hyperparameters: ", study.best_params)
# print("Best validation accuracy: ", study.best_value)

"""##2.1 Load pretrain BERT

"""

BERT_best_params = checkpoint['best_params']

BERT_before_concat = SentimentClassifier(
    hidden_size=BERT_best_params['hidden_size'],
    dropout_rate=BERT_best_params['dropout_rate'],
)
train_loader = DataLoader(train_dataset, batch_size=BERT_best_params['BATCH_SIZE'], shuffle=True, collate_fn=collate_batch_advanced)
val_loader = DataLoader(val_dataset, batch_size=BERT_best_params['BATCH_SIZE'], shuffle=False, collate_fn=collate_batch_advanced)

"""##2.2 Train the model"""

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, min_delta=0.0003):
        """
        Args:
            patience (int): The number of epochs allowed when validation loss is not improving.
            verbose (bool): If True，print the information.
            min_delta (float): the minimum threhold in improving the validation loss, if less than the threshold identify as no improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1  # If no improvement, add 1 to counter
            if self.counter >= self.patience:
                self.early_stop = True  # early stopping
                if self.verbose:
                    print("Early stopping triggered")

def train_and_validate_BERT(model, train_loader, valid_loader, learning_rate, weight_decay):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    valid_losses = []
    valid_accuracies = []
    train_losses = []
    early_stopper = EarlyStopping(patience=15, verbose=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(50):
        model.train()
        train_loss = 0.0  # Initialize train loss for each epoch
        total_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()


            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)


            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)  # Multiply loss by batch size
            total_samples += labels.size(0)

        average_train_loss = train_loss / total_samples
        train_losses.append(average_train_loss)  # Append average loss for the epoch


        model.eval()
        y_pred_val = []
        y_val = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                y_pred_val.extend(preds.cpu().numpy())
                y_val.extend(labels.cpu().numpy())

        val_accuracy = (np.array(y_pred_val) == np.array(y_val)).mean() * 100
        valid_losses.append(val_loss / len(valid_loader))
        valid_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/50], Train Loss: {train_losses[-1]:.4f}, Valid Loss: {valid_losses[-1]:.4f}, Valid Accuracy: {valid_accuracies[-1]:.2f}%')

        early_stopper(val_loss / len(valid_loader), model)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return max(valid_accuracies), train_losses, valid_losses

#This is our train process.

# _, train_losses, valid_losses = train_and_validate_BERT(
#     BERT_before_concat, train_loader, val_loader,
#     BERT_best_params['learning_rate'], BERT_best_params['weight_decay'])

import torch.nn.functional as F
def predict(model, data_loader):
    model.eval()  # swicth to val model
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_masks, labels in data_loader:

            outputs = model(input_ids=input_ids, attention_mask=attention_masks)


            probabilities = F.softmax(outputs, dim=1)


            predicted_classes = torch.argmax(probabilities, dim=1)


            all_preds.extend(predicted_classes.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().flatten())


    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

"""##2.3 The final test result"""

BERT_before_concat.load_state_dict(checkpoint['model_state_dict'])

y_pred, y_probs, all_labels = predict(BERT_before_concat, val_loader)

accuracy = accuracy_score(all_labels, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# This is the train and validation loss plot. Since we don't train the model, instead we load the pre-trained BERT, we cant draw the plot.

# x_axis = range(0, len(train_losses))
# fig, ax = plt.subplots()
# ax.plot(x_axis, train_losses, label='Training Loss')
# ax.plot(x_axis, valid_losses, label='Validation Loss')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# ax.set_title('BERT Train and Validation Loss per Epoch')
# ax.legend()
# plt.show()

print(classification_report(all_labels, y_pred, digits=3))

"""# 3.0 Save the model and random state"""

torch.save({
    'seed_value': seed_value,
    'model_state_dict': BERT_before_concat.state_dict(),
    'best_params': BERT_best_params,
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state(),
    'numpy_rng_state': np.random.get_state(),
    'random_state': random.getstate()
}, '/content/drive/MyDrive/6850/checkpoint_for_BERT.pth')