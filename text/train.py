import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from time import sleep
import torchmetrics
import torch.optim as optim
from torch.utils import data


lr = 1e-3
batch_size = 64
epochs = 10

filepath_dict = {'yelp':   'data/yelp_labelled.txt',
                 'amazon': 'data/amazon_cells_labelled.txt',
                 'imdb':   'data/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
print(df.iloc[0])

sentences = ['John likes ice cream', 'John hates chocolate.']

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.vocabulary_

vectorizer.transform(sentences).toarray()



df_yelp = df[df['source'] == 'yelp']

sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
   sentences, y, test_size=0.25, random_state=1000)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
X_train

class TextClassifier(nn.Module):
    def __init__(self,):
        super(self, TextClassifier).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))

class CFocalLoss(nn.Module):
    """
    0 <= alpha <= 1
    0 <= gamma <=5
    """
    def __init__(self,alpha=0.5,gamma=2,epsilon=1e-9,weight=None,size_average=True) -> None:
        super(CFocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self,pred,target):
        pred = F.softmax(pred,dim=1)
        tar = torch.tensor([[0.]*pred.shape[1]]*pred.shape[0],dtype=torch.float,device='cuda' if torch.cuda.is_available() else 'cpu')
        for idx1,idx2 in enumerate(target):
            tar[idx1][int(idx2.item())] = 1.
        pred = pred + self.epsilon
        loss = torch.tensor([0.]*pred.shape[0],dtype=torch.float,device='cuda' if torch.cuda.is_available() else 'cpu')
        for idx in range(target.shape[0]):
            loss[idx] = -torch.mean(
                tar[idx]*self.alpha*torch.pow((1-pred[idx]),self.gamma)*torch.log2(pred[idx]) 
                + 
                (1-tar[idx])*self.alpha*torch.pow((pred[idx]),self.gamma)*torch.log2(1-pred[idx])
            )
        return torch.mean(loss)



