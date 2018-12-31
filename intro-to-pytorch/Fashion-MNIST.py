import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import helper
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# # Test image loading
# image, label = next(iter(trainloader))
# helper.imshow(image[0,:])
# plt.show()

# Parmeters
device = torch.device('cuda')
input_dimension = 784
hidden_layer_1_dimension = 128
hidden_layer_2_dimension = 64
output_dimension = 10

# Model Architecture
model = nn.Sequential(
            nn.Linear(input_dimension, hidden_layer_1_dimension),
            nn.ReLU(),
            nn.Linear(hidden_layer_1_dimension, hidden_layer_2_dimension),
            nn.ReLU(),
            nn.Linear(hidden_layer_2_dimension, output_dimension),
            nn.LogSoftmax(dim=1)
        ).to(device)

print(model)

# Loss and optimizer
criterion = nn.NLLLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for t in range(10):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1).cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        output = model(images).to(device)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

import helper

# Test out network
dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]

# Convert 2D image to 1D vector
img = img.resize_(1, 784).cuda()

# Class probabilities (softmax) for img
ps = torch.exp(model(img)).cpu()

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28).cpu(), ps, version='Fashion')
plt.show()
