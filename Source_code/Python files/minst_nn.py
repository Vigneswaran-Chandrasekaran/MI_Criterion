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
import time


tic = time.time()
print("Downloading dataset and preparing DataLoader")
master_dataset = dataset.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True )
test_dataset = dataset.MNIST(root = './data', train = False, transform = transforms.ToTensor())
train_dataset, val_dataset = data.random_split(master_dataset, (int(len(master_dataset)*0.8), int(len(master_dataset)*0.2)))

tr_batch_size = 4800
val_batch_size = 1200
train_loader = DataLoader(dataset = train_dataset, batch_size = tr_batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = val_batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = tr_batch_size, shuffle = False)
toc = time.time()
print("Finished preparing. Total time elasped: "+str(toc - tic)+" seconds")

class DeepNN(nn.Module):
    """
    Class defining the structure of Deep Neural Network model and layers characterstics
    """
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
        self.out1 = fun.softmax(self.input_layer(x))
        self.out2 = fun.softmax(self.hlayer1(self.out1))
        self.out3 = fun.softmax(self.hlayer2(self.out2))
        self.out4 = fun.softmax(self.hlayer3(self.out3))
        self.out5 = fun.softmax(self.hlayer4(self.out4))
        self.out6 = fun.softmax(self.output_layer(self.out5))
        
        return fun.softmax(self.out6, dim = 1)
    
model = DeepNN(784, 900, 120, 20, 20, 20, 10)
l1 = np.load('weights_900_first_layer.npz')
model.input_layer.weight.data = l1['w']
model.input_layer.bias.data = l1['b']

l2 = np.load('weights_900_second_layer.npz')
model.hlayer1.weight.data = l2['w']
model.hlayer1.bias.data = l2['b']

l3 = np.load('weights_900_third_layer.npz')
model.hlayer2.weight.data = l3['w']
model.hlayer2.bias.data = l3['b']

l4 = np.load('weights_900_fourth_layer.npz')
model.hlayer3.weight.data = l4['w']
model.hlayer3.bias.data = l4['b']

l5 = np.load('weights_900_fifth_layer.npz')
model.hlayer4.weight.data = l5['w']
model.hlayer4.bias.data = l5['b']

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 10
step_size = 10

train_loss = []
train_acc = []
val_loss =[]
val_acc = []
print("Start training...")

tic = time.time()
for epoch in range(epochs):
    tic_epo = time.time()
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
    toc_epo = time.time()
    print('Epoch %d/%d : ' %(epoch+1,epochs))
    print('Train loss: %.5f, Train acc: %.5f' %(tr_loss, tr_acc))    
    print('Validation loss: %.5f, Validation acc: %.5f' %(v_loss, v_acc))    
    print('Time elasped: %f'%(toc_epo - tic_epo))
    print(" --------------------- ")
    
toc = time.time()
print("Finished Training. Total time elasped: %f"%(toc - tic))
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

# Pickle the model with all necessary details
"""
full_state = { 'epoch':epochs, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 
            'train_loss':train_loss, 'train_acc': train_acc,
            'val_loss':val_loss, 'val_acc':val_acc,
            'mut_X':mut_info_X_monitor, 'mut_Y':mut_info_Y_monitor}
            
torch.save(full_state, 'SaveModel_MI')
"""