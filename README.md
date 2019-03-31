# InformationFlow_in_NeuralNetwork
Visualize the flow of information along various layers of Neural Network.

Neural Networks are encoders of Information, when a huge information is given to Network, they are encoded and at each layer 
relevant information about the target class is retained and other information is tailored out. 

Therefore Mutual information I(X;Y) = H(X) − H(X|Y) [X= Input data, Y= Target class], increases from first layer to last layer.
While training, the mutual information is compressed and expands in such a way that Maximum inforamtion regarding Y is retained as 
well as maximum irrelevant information for Y in X is removed. 

This repo can be used to check how the auto-encoding and flow of information in Neural Network is taking place, which can be 
checked during the training phase and check the contribution of each hidden layer in minimizing last layer's cross-entropy error.

