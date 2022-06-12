# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 14:56:37 2022

@author: mobas
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
# =============================================================================
# LOAD DATASET
# =============================================================================
def loadData():
    #import dataset
    iris = sns.load_dataset('iris')
    # print(iris.head())
    sns.pairplot(iris, hue=("species"))

    data = torch.tensor(iris[iris.columns[:4]].values).float()
    labels = torch.zeros(len(data), dtype=torch.long)
    labels[iris.species == "versicolor"] = 1
    labels[iris.species == "virginica"] = 2
    return data, labels

class ANNClass(nn.Module):
    def __init__(self):
        super().__init__()
        
        # input layer
        self.input = nn.Linear(4,128)
        
        self.hidden = nn.Linear(128, 128),
        
        # output layer
        self.output = nn.Linear(128, 3)
        
    # forward pass
    def forward(self,x):
        # pass through input layer
        
        x= self.input(x)
        # apply relu
        x = f.relu(x)
        
        # output layer
        x = self.output(x)
        # x = torch.sigmoid(x)
        
        return x


def createANewModel():
    ANNclassify = ANNClass()
    
    
    # loss function
    lossfun = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=0.01)
    return ANNclassify, lossfun, optimizer





def trainModel(trainProp):
    numepoch = 1000
    data, labels = loadData()
    # initial loss
    losses = torch.zeros(numepoch)
    trainAcc = []
    testAcc = []
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = trainProp)
    print("Training on.....")
    for epoch in range(numepoch):
        # print(epoch)
        yhat = ANNclassify(X_train)
        loss = lossfun(yhat, y_train)
        
    
        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # accuracy
        trainAcc.append( 100*torch.mean((torch.argmax(yhat,axis=1) == y_train).float()).item() )
        
        # test accuracy
        predLabels = torch.argmax(ANNclassify(X_test), axis = 1)
        testAcc.append(100 * torch.mean((predLabels == y_test).float()).item())
    return trainAcc, testAcc


# =============================================================================
# Test model by running it once    
# =============================================================================
# create model
ANNclassify, lossfun, optimizer = createANewModel()

trainAcc, testAcc = trainModel(0.8)

# =============================================================================
# Plot
# =============================================================================
fig = plt.figure(figsize=(10,5))

plt.plot(trainAcc, 'ro-')
plt.plot(testAcc, 'bs-')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend(["Tarin", "Test"])
plt.show()

print("Final train accuracy: ",trainAcc[-1])
print("Final test accuracy: ",testAcc[-1])

# predictions = ANNclassify(data)
# predLabels = torch.argmax(predictions, axis=1)
# totalacc = 100*torch.mean((predLabels == labels).float())
# print(totalacc)














