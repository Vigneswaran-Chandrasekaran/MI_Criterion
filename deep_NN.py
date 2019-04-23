"""
Simple Feed forward Neural Network from scratch implemented in Pytorch with various easy-use and helpful features
"""
import matplotlib.pyplot as plt
from random import randint
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as fun
import torch
import numpy as np
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
        xf[str(xf.shape[1])] = np.zeros(xf.shape[0])

        xn = pd.read_csv(csv_file['NFocal'], header = 0)
        xn.columns = [str(i) for i in range(xn.shape[1])]
        xn[str(xn.shape[1])] = np.ones(xn.shape[0])

        self.datasamples = xf.append(xn)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return(len(self.datasamples))
    
    def __getitem__(self,index):
        sample = self.datasamples.iloc[index,:-1]
        target = self.datasamples.iloc[index,-1]
        return(sample.as_matrix(), target, index)

    def return_dimension(self):
        return(self.datasamples.shape[1])

    def plotsample(self, index):
        sample = self.datasamples.iloc[index,:-1].as_matrix()
        plt.plot(sample)
        plt.xlabel('time')
        plt.ylabel('mV')
        plt.show()

    def return_statistics(self, index, prop = 'all'):
        sample = np.array(self.datasamples.iloc[index,:-1].as_matrix())
        mean = sample.mean()
        std = sample.std()
        var = sample.var()
        minn = sample.min()
        maxx = sample.max()
        if prop == 'all':
            print(" == Statistical values of the sample == ")
            print("Mean: %f Standard Dev: %f Varience: %f Maximum: %f Minimum: %f", mean, std, var, minn, maxx)
            print(" ====================================== ")
        else:
            stat_dict = {'mean':mean, 'std': std, 'var':var, 'min':minn, 'max':maxx}
            if prop in list(stat_dict.keys()):
                print(prop, stat_dict[prop])
            else:
                raise KeyError("The given key is not available. Make sure you used the correct key. Available keys: 'all','mean','std','var','min','max'")

    def visualize_dataset(self, start_index, end_index, criterion = 'all', label = 'all'):
        """
        Implement various clustering plots to visualize the dataset with respect to various staistical properties
        :TODO 
        """
        
class deepNeuralNet(nn.Module):
    
    def __init__(self, input_dim, nh1, nh2, nh3, nh4, out_dim):
        #initialize model parameters and hyperparameters
        super(deepNeuralNet,self).__init__()
        self.input_dim = input_dim
        self.nh1, self.nh2, self.nh3, self.nh4 = nh1, nh2, nh3, nh4
        
        #initalize layers
        self.linear1 = nn.Linear(input_dim, nh1)
        self.linear2 = nn.Linear(nh1, nh2)
        self.linear3 = nn.Linear(nh2, nh3)
        self.linear4 = nn.Linear(nh3, nh4)
        self.linear5 = nn.Linear(nh4, out_dim)

    def forward(self, x):
        #propogate through each layer
        self.out1 = fun.relu(self.linear1(x))
        self.out2 = fun.relu(self.linear2(self.out1))
        self.out3 = fun.relu(self.linear3(self.out2))
        self.out4 = fun.relu(self.linear4(self.out3))
        self.out5 = fun.sigmoid(self.linear5(self.out4))
        return(self.out5)

def calc_MI(X,Y,bins):

   """
   Should employ accurate methods for calculating MI
   refer: https://openreview.net/forum?id=ry_WPG-A-
   """
   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

bins = 20 # ?  #should add function to calculate optimal number of bins by analyzing the activation of neurons. 

for ix in np.arange(n):
    for jx in np.arange(ix+1,n):
        matMI[ix,jx] = calc_MI(A[:,ix], A[:,jx], bins)
data_set = FocalvsNonFocalData({'Focal':'Focal.csv', 'NFocal':'NFocal.csv'}, 'data/')

dataloader = DataLoader(data_set, batch_size=128, shuffle=True, num_workers=4)

model = deepNeuralNet(37, 50, 70, 100, 10, 1)
epochs = 100
criterion = nn.MSELoss()
print(model.parameters)
optimizer = torch.optim.SGD(model.parameters, lr=0.001)

for epoch in range(epochs):
    for x, y, ind in dataloader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print(loss)
