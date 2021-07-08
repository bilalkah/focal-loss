import torch
from torchvision.transforms.transforms import Compose
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
import pandas as pd
import numpy as np

directory = "digit-recognizer/"

train = pd.read_csv(directory + "train.csv")
label = np.array(train["label"].values)

y_train = np.zeros(shape=(label.shape[0],10),dtype='float32')
for i in range(label.shape[0]):
    y_train[i][label[i]]=1.0

train.drop("label",inplace=True,axis=1)
x_train = np.array(train.values).reshape(-1,28,28,1)/255.0
x_train = np.resize(x_train,(x_train.shape[0],32,32,1))
print(x_train.shape,y_train.shape)
count = 4132
j = 0
for i in range(y_train.shape[0]):
    if np.array_equal(y_train[i],np.array([1.0,0.,0.,0.,0.,0.,0.,0.,0.,0.],dtype='float32')) and count>100:
        count+=1
print(count)
batch_size = 128
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
dataset = data.TensorDataset(x_train,y_train)

train_dataset,val_set = data.random_split(dataset,[10000,32000])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)