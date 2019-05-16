from torch.utils.data import DataLoader
from mutual_infor import mutual_information_2d
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.nn.functional as fun
import torch.nn as nn
import numpy as np
import torchvision
import torch

train_dataset = dataset.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True )
test_dataset = dataset.MNIST(root = './data', train = False, transform = transforms.ToTensor())

batch_size = 6000

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

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

    def return_layer_output(self):
        return self.out1, self.out2, self.out3, self.out4, self.out5, self.out6

obj = DeepNN(784, 1000, 1200, 1100, 1000, 100, 10)

#criterion = nn.CCELoss()
optimizer = torch.optim.Adam(obj.parameters(), lr = 0.001)

epochs = 5
train_loss = []
mutual_info_monitor = []

for epoch in range(epochs):

    loss_monitor = []
    mutual_info = 0
    for i,(images, labels) in enumerate(train_loader):

        images = images.reshape(-1, 28*28)
        optimizer.zero_grad()
        pred = obj(images)
        
        a, b, c, d, e, f = obj.return_layer_output()

        mutual_info += mutual_information_2d(a,b,c,d,e,f,labels)

        loss = fun.nll_loss(pred, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:                              
            loss_monitor.append(loss.item())
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'%(epoch+1, epochs, i+1, len(train_dataset)//batch_size, loss.item()))

    print('Epoch %d/%d Loss: %.5f'%(epoch+1,epochs,sum(loss_monitor)/len(loss_monitor)))
    train_loss.append(sum(loss_monitor)/len(loss_monitor))
    mutual_info_monitor.append(mutual_info)
    print(" --------------------- ")