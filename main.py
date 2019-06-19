from torch.utils import data
from memory_profiler import profile
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
import math
import warnings
warnings.filterwarnings("ignore")

#@profile     # to see the memory profile of the function uncomment the  @profile decorator
def dataset_load(tr_batch_size, val_batch_size, val_split):
    #Download and prepare dataset chunks by DataLoader
    tic = time.time()
    print("Downloading dataset and preparing DataLoader")
    master_dataset = dataset.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True )
    test_dataset = dataset.MNIST(root = './data', train = False, transform = transforms.ToTensor())
    #Train and validation data is split with specified ratio
    train_dataset, val_dataset = data.random_split(master_dataset, (int(len(master_dataset)*(1.0-val_split)), int(len(master_dataset)*val_split)))
    #define dataloaders with defined batch size for training and validation
    train_loader = DataLoader(dataset = train_dataset, batch_size = tr_batch_size, shuffle = True)
    # validation data is shuffled as validation set is used in pretraining and so to avoid any particular class bias
    val_loader = DataLoader(dataset = val_dataset, batch_size = val_batch_size, shuffle = True)
    # shuffling test dataset is not required
    test_loader = DataLoader(dataset = test_dataset, batch_size = tr_batch_size, shuffle = False)
    toc = time.time()
    print("Finished preparing. Total time elasped: "+str(toc - tic)+" seconds")
    return( train_loader, val_loader, test_loader)

def estimate_mutual_info(X, neurons, bins = 5):
    #Estimate Mutual Information between Input data X and Neuron's activations
    neuronal_MI = np.zeros(neurons.shape[1])
    index = 0
    for neuron in neurons.T:
        #loop over each neuron 
        # neuron is the activation of particular neuron for each input data X
        for dim in X.T:
            # loop over each dimension to estimate MI
            # we assume each dimension is independent to each other..
            # Ref: https://stats.stackexchange.com/questions/413511/mutual-information-between-multi-dimensional-and-single-dimensional-variables      
            if np.amax(dim) != np.amin(dim):
                #check whether the dimension have only one value throughout 
                #normalize the values for faster computation
                dim = (dim - np.amin(dim)) / (np.amax(dim) - np.amin(dim))
                neuron = (neuron - np.amin(neuron)) / (np.amax(neuron) - np.amin(neuron))
                #build histogram for joint X and Y
                bins_xy = np.histogram2d(dim, neuron, bins)[0]
                #histogram for X and Y marginal
                bins_x = np.histogram(dim, bins)[0]
                bins_y = np.histogram(neuron, bins)[0]
                # check any sum is zero, although previos condition checks.. a defensive program
                if np.sum(bins_x) != 0 and np.sum(bins_y) != 0 and np.sum(bins_xy) != 0:
                    #calculate marginal probabilities
                    p_x = bins_x / np.sum(bins_x)
                    p_y = bins_y / np.sum(bins_y)
                    #calculate joint probability
                    p_xy = bins_xy / np.sum(bins_xy)
                    #estimate entropy 
                    H_x = -1 * np.sum( p_x * np.log(p_x))
                    H_y = -1 * np.sum(p_y * np.log(p_y))
                    H_xy = -1 * np.sum(p_xy * np.log(p_xy))
                    #Mutual Information of the particular dimension and neuron out put 
                    # I(X;Y) = H(X) + H(Y) - H(X,Y)
                    sum = H_x + H_y - H_xy
                    # check any is NaN.. This occurs sometimes because, in p_X * log(p_X), sometimes p_X becomes 0
                    # making the whole term zero, but however in numpy this turns to become NaN...
                    # TODO: Find a good way to handle this
                    if not math.isnan(sum):
                        # add the MI value to the corresponding neuron index
                        neuronal_MI[index] += sum
        index += 1    
    # return array of mutual information corresponding to each neuron
    return(neuronal_MI)

def check_k_Helly(bin_avg_new, bin_avg, clusters_new, clusters, k_helly):
    # Using k-helly property to check wheteher the stopping condition is satisfied or not
            

#@profile
def pre_train_model(model, val_loader):
    # list of all model's hidden layers
    layers = [model.input_layer, model.hlayer1, model.hlayer2, model.hlayer3, model.hlayer4]
    # Hyperparameter k-value which determines number of bins for the particular layer
    # This is calaculated based on Partial Information Decomposition 
    k_value = [4, 8, 5, 3, 2]
    # k-helly stopping criterion's k-value to chec degree of overlap
    k_helly = [5, 5, 5, 5, 5]
    # loop over the layers 
    for l_indx in range(len(layers)):
        # get the layer's parameters and detach it for updation 
        print("Working on layer: "+str(l_indx))
        w_matrix = layers[l_indx].weight.data.clone().detach().numpy()
        b_matrix = layers[l_indx].bias.data.clone().detach().numpy()
        #load the input data X
        # use the validation dataset images as X
        for _, (images, _) in enumerate(val_loader):
            # reshape images 
            images = images.reshape(-1, 28*28).clone().detach().numpy()
            # find the activaion by,
            # act( X.W_T + b)
            activation = np.dot(images,w_matrix.T) + b_matrix 
            activation = 1/(1 + np.exp(-activation))
            # Estimate Mutual Information for each neuron
            tic_est = time.time()
            print("Estimation Information Theorotic quantities")
            neuronal_MI = estimate_mutual_info(images, activation, bins = 2)
            toc_est = time.time()
            print("Elasped time for estimation: "+str(round(toc_est-tic_est,1))+" seconds")
            print("----- Remeber: this elasped time is commenstruate with time required for pre-training the particular layer -----")
            # Get the index of sorted neurons based on MI value
            index_sorted = np.argsort(neuronal_MI)[::-1]
            #create clusters of given k value
            clusters = np.array_split(index_sorted, k_value[l_indx])
            # calculate the bin's avergage MI
            bin_avg = []
            for i in clusters:
                bin_avg.append(np.sum(neuronal_MI[np.ix_(i)]) / neuronal_MI[np.ix_(i)].shape[0])
            # start tuning parameters !!!
            # hyperparameters defined 
            iteration = 0
            decay_factor = 100    # decay factor for step-size
            stopping_criteria = True
            while stopping_criteria:
                print("Iteration: "+str(iteration))
                for bin in range(len(clusters)):
                    #TODO: parameter updation very important.... High priority    
                    # w_matrix have n number of rows with each row representing particular neuron connection
                    w_matrix[np.ix_(clusters[bin])] -=  0.0001
                    b_matrix[np.ix_(clusters[bin])] -=  0.0001
                #calculate activation for the updated weights 
                activation = np.dot(images,w_matrix.T) + b_matrix 
                activation = 1/(1 + np.exp(-activation))
                #estimate MI for the tuned parameters
                neuronal_MI = estimate_mutual_info(images, activation, bins = 2)
                index_sorted = np.argsort(neuronal_MI)[::-1]
                #create new clusters from the estimated weight
                for neuron_mi in neuronal_MI:
                    if neuron_mi >= bin_avg[0]:
                        

                iteration += 1
                if iteration == 3:
                    stopping_criteria = False
            layers[l_indx].weight.data = w_matrix
            layers[l_indx].bias.data = b_matrix

            exit(1)
    return(model)

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


if __name__ == '__main__':        

    tr_batch_size = 4800
    val_batch_size = 12000
    val_split = 0.2
    train_loader, val_loader, test_loader = dataset_load(tr_batch_size, val_batch_size, val_split)
    model = DeepNN(784, 20, 120, 20, 20, 20, 10)
    print("Pretraining phase..")
    pre_trained_model = pre_train_model(model, val_loader)