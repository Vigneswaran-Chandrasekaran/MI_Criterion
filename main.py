from torch.utils import data
from memory_profiler import profile
from scipy.stats import entropy
from torch.utils.data import DataLoader
from fast_histogram import histogram1d, histogram2d
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

class DeepNN(nn.Module):
    # Class defining the structure of Deep Neural Network model and layers characterstics
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
    # function to simulate forward propogation
    def forward(self, x):
        #propogation of each layer
        self.out1 = fun.softmax(self.input_layer(x))
        self.out2 = fun.softmax(self.hlayer1(self.out1))
        self.out3 = fun.softmax(self.hlayer2(self.out2))
        self.out4 = fun.softmax(self.hlayer3(self.out3))
        self.out5 = fun.softmax(self.hlayer4(self.out4))
        self.out6 = fun.softmax(self.output_layer(self.out5))
        #output layer
        return fun.softmax(self.out6, dim = 1)

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
    print("Number of neurons computed:")
    for neuron in neurons.T:
        if index % 100 == 0:
            print(index, end = " ", flush = True)
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
                bins_xy = histogram2d(dim, neuron, bins, range = [[0,1],[0,1]])
                #histogram for X and Y marginal
                bins_x = histogram1d(dim, bins, range = [0,1])
                bins_y = histogram1d(neuron, bins, range = [0,1])
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


def check_k_strong_helly():
    # Using k-helly property to check wheteher the stopping condition is satisfied or not
    return True
            
#@profile
def pre_train_model(model, val_loader):
    # list of all model's hidden layers
    layers = [model.input_layer, model.hlayer1, model.hlayer2, model.hlayer3, model.hlayer4]
    # TODO save file with parameters to avoid error for unknown file
    # Hyperparameter k-value which determines number of bins for the particular layer
    # This is calaculated based on Partial Information Decomposition 
    k_value = [10, 8, 5, 3, 2]
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
            global actX, act1, act2, act3, act4
            # reshape images 
            images = images.reshape(-1, 28*28).clone().detach().numpy()
            # find the activaion by,
            # act( X.W_T + b)
            # We need to freeze each layer one by one and use the updated weights
            # from before layer, hence we have the immmediate before data as actX
            # When the paricular layer is called, subsequent sequential of outputs
            # are calculated and final activation is calculated
            if l_indx == 0:    # First layer, which have actX as input images [immediate input to the layer]
                actX = images   
            elif l_indx == 1:  # Second layer
                layer1_param = np.load('weights_900_first_layer.npz') #load the saved updated weights and biases
                # act1 contains the activations when the weights are updated, this acts as a input
                # for subsequent layers
                act1, actX = 1 / (1 + np.exp(-1 * (np.dot(images, layer1_param['w'].T) + layer1_param['b'])))  
            elif l_indx == 2:  # Third layer
                layer2_param = np.load('weights_900_second_layer.npz')
                act2, actX = 1 / ( 1 + np.exp(-1 * np.dot(act1, layer2_param['w'].T) + layer2_param['b']))
            elif l_indx == 3:  # Fourth layer
                layer3_param = np.load('weights_900_third_layer.npz')
                act3, actX = 1 / (1 + np.exp(-1 * (np.dot(act2,layer3_param['w'].T) + layer3_param['b'])))
            elif l_indx == 4:   # Fifth layer
                layer4_param = np.load('weights_900_fourth_layer.npz')
                act4, actX = 1 / (1 + np.exp(-1 * (np.dot(act3,layer4_param['w'].T) + layer4_param['b'])))
            # get the activation of the layer when actX , w_matrix and b_matrix are known
            activation = 1 / (1 + np.exp(-1 * (np.dot(actX,w_matrix.T) + b_matrix )))
            # Estimate Mutual Information for each neuron
            tic_est = time.time()
            print("Estimation Information Theorotic quantities")
            neuronal_MI = estimate_mutual_info(images, activation, bins = 5)
            toc_est = time.time()
            print("Elasped time for estimation: "+str(round(toc_est-tic_est,2))+" seconds")
            print("----- This elasped time is commenstruate with time required for the particular layer -----")
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
            decay_factor = 0.01    # decay factor for step-size
            stopping_criteria = True

            while stopping_criteria:
                print("Iteration: "+str(iteration))
                for bin in range(len(clusters)):
                    #TODO: parameter updation very important.... High priority    
                    # w_matrix have n number of rows with each row representing particular neuron connection
                    # Divergence between current bin and best bin(target)
                    divergence = entropy(clusters[bin_avg.index(max(bin_avg))], clusters[bin])
                    print("Divergence: "+str(divergence))
                    w_matrix[np.ix_(clusters[bin])] -=  divergence * (1/(iteration+1))*decay_factor
                    b_matrix[np.ix_(clusters[bin])] -=  divergence * (1/(iteration+1))*decay_factor
                #calculate activations for the updated weights 
                activation = 1 / (1 + np.exp(-1 * (np.dot(actX,w_matrix.T) + b_matrix )))
                #estimate MI for the tuned parameters
                neuronal_MI_after_update = estimate_mutual_info(images, activation, bins = 5)
                index_sorted_after_update = np.argsort(neuronal_MI_after_update)[::-1]
                clusters_after_update = np.array_split(index_sorted_after_update, k_value[l_indx])
                # calculate the bin's avergage MI
                bin_avg_after_update = []
                for i in clusters_after_update:
                    bin_avg_after_update.append(np.sum(neuronal_MI_after_update[np.ix_(i)]) / neuronal_MI_after_update[np.ix_(i)].shape[0])
                iteration += 1
                if iteration == 5 or check_k_strong_helly():   #TODO: define function to check k-helly property criteria
                    if iteration == 5:
                        print("Number of Iterations == Max_iteration!")
                    else:
                        print("K-Helly property is satisfied!")
                    print("Finish training the layer"+str(l_indx))
                    stopping_criteria = False
                    np.savez("Param_900_1_kl_div", w = w_matrix, b = b_matrix)
                    exit(1)
                    layers[l_indx].weight.data = w_matrix
                    layers[l_indx].weight.data = b_matrix
                    #file_name = input("Enter the file name to save the layer's parameters: ")
                    #np.save(file_name, w = w_matrix, b = b_matrix)
    return(model)

if __name__ == '__main__':        

    tr_batch_size = 4800
    val_batch_size = 12000
    val_split = 0.2
    train_loader, val_loader, test_loader = dataset_load(tr_batch_size, val_batch_size, val_split)
    actX = 0; act1 = 0; act2 = 0; act3 = 0; act4 = 0
    model = DeepNN(784, 900, 20, 20, 20, 20, 10)
    print("Pretraining phase..")
    pre_trained_model = pre_train_model(model, val_loader)