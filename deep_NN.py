"""
This file contains Simple Feed forward Neural Network from scratch implemented in Pytorch
"""
import matplotlib.pyplot as plt
from random import randint
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as numpy
import pandas as pd
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

class FocalvsNonFocalData(Dataset):

    """ University of Bern Barcelona dataset 
        For dataset description: https://www.upf.edu/web/mdm-dtic/-/1st-test-dataset?inheritRedirect=true#.XLcwG0PhWzI
    """

    def __init__(self, csv_file, root_dir, transform=None):
        xf = pd.read_csv(csv_file['Focal'], header = 0)
        xf.columns = [str(i) for i in range(xf.shape[1])]
        xf[str(xf.shape[1])] = numpy.zeros(xf.shape[0])

        xn = pd.read_csv(csv_file['NFocal'], header = 0)
        xn.columns = [str(i) for i in range(xn.shape[1])]
        xn[str(xn.shape[1])] = numpy.ones(xn.shape[0])

        self.datasamples = xf.append(xn)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return(len(self.datasamples))
    
    def __getitem__(self,index):
        sample = self.datasamples.iloc[index,:-1]
        return(sample.as_matrix())

    def plotsample(self, index):
        sample = self.datasamples.iloc[index,:-1].as_matrix()
        plt.plot(sample)
        plt.xlabel('time')
        plt.ylabel('mV')
        plt.show()

data_set = FocalvsNonFocalData({'Focal':'Focal.csv', 'NFocal':'NFocal.csv'}, 'data/')

dataloader = DataLoader(data_set, batch_size=128, shuffle=True)

class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 3 X 2 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor
        
    def forward(self, X):
        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3) # final activation function
        return o
        
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)
    
    def backward(self, X, y, o):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
        
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))

criteron = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(epochs):
    y_pred = model(x)
    loss = criteron(y_pred, y)
    print("Epoch: ",epoch, "Loss: ",loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
