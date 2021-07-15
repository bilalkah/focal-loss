import pandas as pd
import numpy as np

directory = "digit-recognizer/"

train = pd.read_csv(directory + "train.csv")
label = np.array(train["label"].values)

y_train = np.zeros(shape=(label.shape[0],10),dtype='float32')
for i in range(label.shape[0]):
    y_train[i][label[i]]=1

train.drop("label",inplace=True,axis=1)
x_train = np.array([train.values]).reshape(-1,784)/255.0
print(x_train.shape)

zero = np.array([[0.]*28*28])
one = np.array([[0.]*28*28])
two = np.array([[0.]*28*28])
three = np.array([[0.]*28*28])
four = np.array([[0.]*28*28])
five = np.array([[0.]*28*28])
six = np.array([[0.]*28*28])
seven = np.array([[0.]*28*28])
eight = np.array([[0.]*28*28])
nine = np.array([[0.]*28*28])

for _,(data,label) in enumerate(zip(x_train,y_train)):
    if label[0]==1.:
        zero = np.append(zero,np.expand_dims(data, axis=0),axis=0)
    if label[1]==1.:
        one = np.append(one,np.expand_dims(data, axis=0),axis=0)
    if label[2]==1.:
        two = np.append(two,np.expand_dims(data, axis=0),axis=0)
    if label[3]==1.:
        three = np.append(three,np.expand_dims(data, axis=0),axis=0)
    if label[4]==1.:
        four = np.append(four,np.expand_dims(data, axis=0),axis=0)
    if label[5]==1.:
        five = np.append(five,np.expand_dims(data, axis=0),axis=0)
    if label[6]==1.:
        six = np.append(six,np.expand_dims(data, axis=0),axis=0)
    if label[7]==1.:
        seven = np.append(seven,np.expand_dims(data, axis=0),axis=0)
    if label[8]==1.:
        eight = np.append(eight,np.expand_dims(data, axis=0),axis=0)
    if label[9]==1.:
        nine = np.append(nine,np.expand_dims(data, axis=0),axis=0)

zero = np.delete(zero, 0, 0)
one = np.delete(one, 0, 0)
two = np.delete(two, 0, 0)
three = np.delete(three, 0, 0)
four = np.delete(four, 0, 0)
five = np.delete(five, 0, 0)
six = np.delete(six, 0, 0)
seven = np.delete(seven, 0, 0)
eight = np.delete(eight, 0, 0)
nine = np.delete(nine, 0, 0)

mnist = np.append(np.expand_dims(one,axis=0), np.expand_dims(two,axis=0))
"""mnist = np.append(mnist,np.expand_dims(three,axis=0))
mnist = np.append(mnist,np.expand_dims(four,axis=0))
mnist = np.append(mnist,np.expand_dims(five,axis=0))
mnist = np.append(mnist,np.expand_dims(six,axis=0))
mnist = np.append(mnist,np.expand_dims(seven,axis=0))
mnist = np.append(mnist,np.expand_dims(eight,axis=0))
mnist = np.append(mnist,np.expand_dims(nine,axis=0))"""
np.save("./mnist-data",mnist)
print(mnist.shape)