from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.nn.functional as fun
import matplotlib.pyplot as plt
import keras.backend as K
import torch.nn as nn
import numpy as np
import torchvision
import torch
import keras
import time

def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = K.expand_dims(K.sum(K.square(X), axis=1), 1)
    dists = x2 + K.transpose(x2) - 2*K.dot(X, K.transpose(X))
    return dists

def get_shape(x):
    dims = K.cast( K.shape(x)[1], K.floatx() ) 
    N    = K.cast( K.shape(x)[0], K.floatx() )
    return dims, N

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    lprobs = K.logsumexp(-dists2, axis=1) - K.log(N) - normconst
    h = -K.mean(lprobs)
    return dims/2 + h

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)


def mutual_estimate(activation, labels):

    label_probability = []
    Y_samples = K.placeholder(ndim = 2)
    noise_variance = 0.05
    labels = list(labels.clone().detach().numpy())
    unique, counts = np.unique(labels, return_counts = True)
    for i in unique:
        p_i = counts[i-1]/len(labels)
        label_probability.append(p_i)
    entropy_func_upper = K.function([Y_samples,], [entropy_estimator_kl(Y_samples, noise_variance),])
    act_MI_X = []
    act_MI_Y = []
    for layer in activation:
    
        data = layer.clone().detach().numpy()
        H_Y_upper = entropy_func_upper([data,])[0]
        H_Y_given_X = kde_condentropy(data, noise_variance)
        aa = np.random.rand(data.shape[0], data.shape[1])
        H_Y_upper = entropy_func_upper([aa,])[0]
        act_MI_X.append(H_Y_upper - H_Y_given_X)         
    
    for layer in activation:
    
        hm_given_Y = 0
        data = layer.clone().detach().numpy()
        H_Y_upper = entropy_func_upper([data,])[0]
        
        for i in range(len(unique)-1):
            label_rows = []
            for j in range(len(layer)):
                if labels[j] == i+1:
                    label_rows.append(j)
            data_label = data[label_rows]

            H_cond_upper = entropy_func_upper([data_label,])[0]
            hm_given_Y += label_probability[i] * H_cond_upper
            
        act_MI_Y.append(H_Y_upper - hm_given_Y)          
    return(np.array(act_MI_X), np.array(act_MI_Y))

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

    def return_activation(self):
    
        return([self.out1, self.out2, self.out3, self.out4, self.out5, self.out6]) 

model = DeepNN(784, 1024, 1200, 1200, 1200, 200, 10)
for mod in model.modules():
    print(mod)
exit(1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 1
step_size = 10

train_loss = []
train_acc = []
val_loss =[]
val_acc = []
mut_info_X_monitor = []
mut_info_Y_monitor = []
print("Start training...")

tic = time.time()
for epoch in range(epochs):
    tic_epo = time.time()
    epoch_loss = 0
    epoch_acc = 0
    epoch_val_loss = 0
    epoch_val_acc = 0
    mut_I_X = np.zeros(6)
    mut_I_Y = np.zeros(6)
    
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
        mutx, muty = mutual_estimate(model.return_activation(), labels)
        mut_I_X += mutx
        mut_I_Y += muty

    tr_acc = epoch_acc/len(train_dataset)
    tr_loss = epoch_loss/len(train_loader)
    train_loss.append(tr_loss)
    train_acc.append(tr_acc)
    mut_info_X_monitor.append(mut_I_X)
    mut_info_Y_monitor.append(mut_I_Y)
    
    
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