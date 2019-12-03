from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as fun
import torch
import time

class Bern_dataset(torch.utils.data.Dataset):

    def __init__(self):
        focal_data = np.genfromtxt('Focal.csv', delimiter=',')
        nfocal_data = np.genfromtxt('NFocal.csv', delimiter=',')
        foc_lab = np.ones(3750) * 1
        nfoc_lab = np.ones(3750) * 0
        self.Y = np.concatenate((foc_lab, nfoc_lab))
        self.X = np.concatenate((focal_data, nfocal_data))
        #self.X = normalize(self.X)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def get_dataloader(tr_batch_size = 64, val_batch_size = 64):
    tic = time.time()
    print("Preparing DataLoader")
    data = Bern_dataset()
    train_dataset, val_dataset = torch.utils.data.random_split(data, (int(len(data) * 0.8), int(len(data) * 0.2)))
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = tr_batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = val_batch_size, shuffle = True)
    toc = time.time()
    print("Completed")
    print("Time elasped: "+str(round(toc - tic, 2))+" seconds")
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

class DeepNN(torch.nn.Module):
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
        self.out1 = torch.relu(self.input_layer(x))
        self.out2 = torch.relu(self.hlayer1(self.out1))
        self.out3 = torch.relu(self.hlayer2(self.out2))
        self.out4 = torch.relu(self.hlayer3(self.out3))
        self.out5 = torch.relu(self.hlayer4(self.out4))
        self.out6 = torch.relu(self.output_layer(self.out5))
        
        return torch.softmax(self.out6, dim = 1)

"""  
model = DeepNN(37, 90, 120, 20, 20, 20, 2)
model = model.float()

#criterion = nn.Loss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 10000)
train_loader, val_loader, len_tr, len_va = get_dataloader()
epochs = 100

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
    
    for i, (samp, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.float()
        output = model(samp.float())
        output = output.float()
        loss = criterion(output, labels.long())
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, pred = torch.max(output.data, 1)
        epoch_acc += float((pred.float() == labels.data).sum())

    tr_acc = epoch_acc/len_tr
    tr_loss = epoch_loss/len(train_loader)
    train_loss.append(tr_loss)
    train_acc.append(tr_acc)
    
    
    for j, (samp, labels) in enumerate(val_loader):
    
        output = model(samp.float())
        output = output.float()
        labels = labels.float()
        loss = criterion(output, labels.long())
        epoch_val_loss += loss.item()
        _, pred = torch.max(output.data, 1)
        epoch_val_acc += float((pred.float() == labels.data).sum())
    
    v_acc = epoch_val_acc/len_va
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
"""

X = np.genfromtxt('feat.csv', delimiter=',')

a = np.ones(200) * 0
b = np.ones(300) * 1
#c = np.ones(100) * 2
Y = np.concatenate((a, b))


from keras.utils import to_categorical
Y = to_categorical(Y)

# first neural network with keras tutorial
from keras.models import Sequential
from keras.layers import Dense

X = (X - np.amin(X, axis=0)) / (np.amax(X, axis=0) - np.amin(X, axis=0))
# load the dataset

# define the keras model
model = Sequential()
model.add(Dense(50, input_dim=X.shape[1], activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(2, activation='softmax'))
# compile the keras model
from keras.optimizers import RMSprop
optim = RMSprop(learning_rate=0.1)

model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])


#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
hist = model.fit(X, Y, epochs=100, batch_size=256, shuffle=True, verbose=2)
plt.plot(hist.history['accuracy'])
plt.show()

plt.plot(hist.history['loss'])
plt.show()
