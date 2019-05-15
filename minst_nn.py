import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torchvision
from scipy import ndimage
from sklearn.feature_selection import mutual_info_classif
import numpy as np
train_dataset = dataset.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True )
test_dataset = dataset.MNIST(root = './data', train = False, transform = transforms.ToTensor())
from torch.utils.data import DataLoader
print(len(train_dataset))
exit(1)
batch_size = 100
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

import torch.nn.functional as fun
import torch.nn as nn
class DeepNN(nn.Module):
    def __init__(self, input_dim, nh1, nh2, nh3, nh4, nh5, output_dim):
        super(DeepNN, self).__init__()
        #hyperparameter setting 
        self.input_dim = input_dim
        self.nh1, self.nh2, self.nh3, self.nh4, self.nh5, = nh1, nh2, nh3, nh4, nh5
        self.output_dim = output_dim
        #layer definition 
        self.input_layer = nn.Linear(self.input_dim, self.nh1)
        self.hlayer1 = nn.Linear(self.nh1, self.nh2)
        self.hlayer2 = nn.Linear(self.nh2, self.nh3)
        self.hlayer3 = nn.Linear(self.nh3, self.nh4)
        self.hlayer4 = nn.Linear(self.nh4, self.nh5)
        self.output_layer = nn.Linear(self.nh5, self.output_dim)
        
    def forward(self, x):
        #propogation of each layer
        self.out1 = fun.relu(self.input_layer(x))
        self.out2 = fun.relu(self.hlayer1(self.out1))
        self.out3 = fun.relu(self.hlayer2(self.out2))
        self.out4 = fun.relu(self.hlayer3(self.out3))
        self.out5 = fun.relu(self.hlayer4(self.out4))
        self.out6 = fun.relu(self.output_layer(self.out5))
        return fun.log_softmax(self.out6, dim = 1)

    def feed_MI(self, x, labels):
        for i in range(100):
            mi += mutual_information(x[i], labels)
        return(mi)
        
def mutual_information(self, x, y, sigma=1, normalized=False):
    EPS = np.finfo(float).eps
    bins = (256, 256)
    jh = np.histogram2d(x, y, bins=bins)[0]
    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',output=jh)
    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
            - np.sum(s2 * np.log(s2)))
    return mi

obj = DeepNN(784, 1000, 1200, 1100, 1000, 100, 10)
import torch
#criterion = nn.CCELoss()
optimizer = torch.optim.Adam(obj.parameters(), lr = 0.001)
from torch.autograd import Variable
epochs = 30
train_loss = []
for epoch in range(epochs):
    loss_monitor = []
    for i,(images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)
        optimizer.zero_grad()
        pred = obj(images)
        loss = fun.nll_loss(pred, labels)
        #loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:                              
            loss_monitor.append(loss.item())
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'%(epoch+1, epochs, i+1, len(train_dataset)//batch_size, loss.item()))
    print('Epoch %d/%d Loss: %.5f'%(epoch+1,epochs,sum(loss_monitor)/len(loss_monitor)))
    train_loss.append(sum(loss_monitor)/len(loss_monitor))
    print(" --------------------- ")