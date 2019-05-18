from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.nn.functional as fun
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchvision
import torch

master_dataset = dataset.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True )
test_dataset = dataset.MNIST(root = './data', train = False, transform = transforms.ToTensor())
train_dataset, val_dataset = data.random_split(master_dataset, (int(len(master_dataset)*0.8), int(len(master_dataset)*0.2)))

tr_batch_size = 4800
val_batch_size = 1200
train_loader = DataLoader(dataset = train_dataset, batch_size = tr_batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = val_batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = tr_batch_size, shuffle = False)

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
        return fun.softmax(self.out6, dim = 1)

    def return_layer_output(self):

        return self.out1, self.out2, self.out3, self.out4, self.out5, self.out6

model = DeepNN(784, 1000, 1200, 1100, 1000, 100, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 2
step_size = 10
train_loss = []
train_acc = []
val_loss =[]
val_acc = []

for epoch in range(epochs):

    epoch_loss = 0
    epoch_acc = 0
    epoch_val_loss = 0
    epoch_val_acc = 0

    for i,(images, labels) in enumerate(train_loader):

        images = images.reshape(-1, 28*28)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, pred = torch.max(output.data, 1)
        epoch_acc += float((pred == labels.data).sum())
    
    tr_acc = epoch_acc/len(train_dataset)
    tr_loss = epoch_loss/len(train_loader)
    train_loss.append(tr_loss)
    train_acc.append(tr_acc)
    
    for j, (images, labels) in enumerate(val_loader):
    
        images = images.reshape(-1, 28*28)
        output = model(images)
        loss = criterion(output, labels)
        epoch_val_loss += loss.item()
        _, pred = torch.max(output.data, 1)
        epoch_val_acc += float((pred == labels.data).sum())
    
    v_acc = epoch_val_acc/len(val_dataset)
    v_loss = epoch_val_loss/len(val_loader)
    val_loss.append(v_loss)
    val_acc.append(v_acc)
    
    print('Epoch %d/%d : ' %(epoch+1,epochs))
    print('Train loss: %.5f, Train acc: %.5f' %(tr_loss, tr_acc))    
    print('Validation loss: %.5f, Validation acc: %.5f' %(v_loss, v_acc))    
    print(" --------------------- ")

plt.plot(range(1, epochs+1), val_loss, train_loss)
plt.title('Loss monitor')
plt.legend(['Val_loss', 'Train_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(range(1, epochs+1), val_acc, train_acc)
plt.title('Accuracy monitor')
plt.legend(['Val_acc', 'Train_acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

full_state = { 'epoch':epochs, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 
            'train_loss':train_loss, 'train_acc': train_acc,
            'val_loss':val_loss, 'val_acc':val_acc}
            
torch.save(full_state, 'SaveModel')
