import numpy as np
import torch

import helper

import matplotlib.pyplot as plt

# Setup the dataset
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
# plt.show()


# Flatten image tensor
inputs = images.view(images.shape[0], -1)

# Network parameters
W1 = torch.randn(784, 256)
B1 = torch.randn(256)

W2 = torch.randn(256, 10)
B2 = torch.randn(10)

# Network functions
def activation(x):
    return 1/(1+torch.exp(-x))

def softmax(x):
    # print(x.shape)
    # denominator = 0
    # softmax_x = []
    # for k in range (len(x)):
    #     denominator += torch.exp(x[k])

    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

# forward propagation
hidden = activation(torch.mm(inputs, W1) + B1)
output = softmax(torch.mm(hidden, W2) + B2)

# print("Shape should be (64, 10)")
# print(output.shape)
# print("Output should sum to 1")
# print(output.sum(dim=1))

# print(output)

##################################
# Building Networks with PyTorch #
##################################
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        # Weights and bias tensors are automatically created
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)    # dim=1 calculates softmax across columns

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

# model = Network()
# print(model)

##################################################
# Create network with torch.nn.functional module #
##################################################

import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        # Weights and bias tensors are automatically created
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)

        return x

##################
# Custom Network #
##################
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

model = Network()
# print(model)

# Set biases to all zeros
model.fc1.bias.data.fill_(0)

# Custom weight initialization
model.fc1.weight.data.normal_(std=0.01)

# Grab some data
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is
images.resize_(images.shape[0], 1, 784)

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)

#######################
# Using nn.Sequential #
#######################

# Hyperparameters
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))

print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)
