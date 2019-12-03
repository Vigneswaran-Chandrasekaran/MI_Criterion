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

def bern_dataset(wavelet_family = 'db10', level = 4):
    
    data = np.load("../Healthcare_signal_processing/Datasets/Barca/Focal_all_32bit.npz")
    data['d'].shape
    focal = np.array( np.split(data['d'], 3750) )
    data = np.load("../Healthcare_signal_processing/Datasets/Barca/NFocal_all_32bit.npz")
    nfocal = np.array( np.split(data['d'], 3750) )
    db = pywt.Wavelet(wavelet_family)
    a4 = []; d4 = []; d3 = []; d2 = []; d1 = []

    for samp in focal:
        cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(samp, db, level = level)
        a4.append(cA4)
        d4.append(cD4)
        d3.append(cD3)
        d2.append(cD2)
        d1.append(cD1)

    for samp in nfocal:
        cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(samp, db, level = level)
        a4.append(cA4)
        d4.append(cD4)
        d3.append(cD3)
        d2.append(cD2)
        d1.append(cD1)

    a4 = np.array(a4)
    d4 = np.array(d4)
    d3 = np.array(d3)
    d2 = np.array(d2)
    d1 = np.array(d1)

    return [a4, d4, d3, d2, d1]

data = bern_dataset()
print(data.shape)

"""
print("Downloading dataset and preparing DataLoader")
master_dataset = dataset.CIFAR10(root = './data', train = True, transform = transforms.ToTensor(), download = True )
test_dataset = dataset.CIFAR10(root = './data', train = False, transform = transforms.ToTensor())
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
        self.out1 = fun.relu(self.input_layer(x))
        self.out2 = fun.relu(self.hlayer1(self.out1))
        self.out3 = fun.relu(self.hlayer2(self.out2))
        self.out4 = fun.relu(self.hlayer3(self.out3))
        self.out5 = fun.relu(self.hlayer4(self.out4))
        self.out6 = fun.relu(self.output_layer(self.out5))
        
        return fun.softmax(self.out6, dim = 1)
    
model = DeepNN(1024, 1100, 120, 20, 20, 20, 10)

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

        images = images.sum(axis = 1)
        images = images.reshape(-1, 32 * 32)
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
    
        images = images.sum(axis = 1)
        images = images.reshape(-1, 32 * 32)
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

full_state = { 'epoch':epochs, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 
            'train_loss':train_loss, 'train_acc': train_acc,
            'val_loss':val_loss, 'val_acc':val_acc,
            'mut_X':mut_info_X_monitor, 'mut_Y':mut_info_Y_monitor}
            
torch.save(full_state, 'SaveModel_MI')
"""