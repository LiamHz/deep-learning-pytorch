import torch

def activation(x):
    return 1/(1+torch.exp(-x))

# Generate some data
torch.manual_seed(7)

features = torch.randn((1, 5))
weights = torch.rand_like(features)
bias = torch.rand((1, 1))

# Compute the output of the network
output = activation(torch.sum(features * weights) + bias)

# Compute the output using matrix multiplication
# Weights tensor has to be transposed
output = activation(torch.mm(features, weights.view(5, 1)) + bias)
print("aaa", output)


###############################
# Stacking neurons in PyTorch #
###############################

# Generate some data
torch.manual_seed(7)

features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1] # must match number of input features
n_hidden = 2
n_output = 1

# Weights for input to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# Bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

# Calculate the output of this network
h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)

print(output)


#############################
# Numpy to PyTorch and back #
#############################

import numpy as np
a = np.random.rand(4, 3)
print(a)

# Turn Numpy array into PyTorch tensor
b = torch.from_numpy(a)
print(b)

# Turn PyTorch tensor into Numpy array
b = b.numpy()
