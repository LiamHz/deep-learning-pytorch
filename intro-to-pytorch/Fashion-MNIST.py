import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import helper
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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
hidden_layer_1_dimension = 256
hidden_layer_2_dimension = 128
hidden_layer_3_dimension = 64
output_dimension = 10
dropout_rate = 0.2
epochs = 10

# Model Architecture
model = nn.Sequential(
            nn.Linear(input_dimension, hidden_layer_1_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_1_dimension, hidden_layer_2_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_2_dimension, hidden_layer_3_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_3_dimension, output_dimension),
            nn.LogSoftmax(dim=1)
        ).to(device)

print(model)

# Loss and optimizer
criterion = nn.NLLLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_loss = []
validation_loss = []
# Training loop
for t in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1).cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        log_ps = model(images).to(device)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Training loss: {running_loss/len(trainloader)}")
    training_loss.append(running_loss/len(trainloader))

    # Evaluate model on validation set
    # Turn off gradients
    accuracy = 0
    with torch.no_grad():
        # Set model to evaluation mode
        # Turn off dropout for inference
        model.eval()

        for images, labels in testloader:
            images = images.view(images.shape[0], -1).cuda()
            ps = torch.exp(model(images))

            # Get the top-1 prediction and class
            top_p, top_class = ps.topk(1, dim=1)

            # top_class is tensor with shape (64, 1)
            # labels is 1D with shape (64)
            # labels has to be reshaped to have equality work
            correct_prediction = top_class == labels.view(*top_class.shape).cuda()

            # correct_prediction is a ByteTensor and has to be converted to a FloatTensor for torch.mean
            accuracy += torch.mean(correct_prediction.type(torch.FloatTensor))
        print(f"Accuracy: {accuracy/len(testloader)}")
        validation_loss.append(accuracy/len(testloader))

    # Set model back to train mode
    model.train()


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


# Plot training and validation loss over epochs
plt.plot(training_loss, 'r')   # Plot in red
plt.plot(validation_loss, 'b') # Plot in blue

# Legend
red_patch = mlines.Line2D([], [], color='red', label='Training Loss')
blue_patch = mlines.Line2D([], [], color='blue', label='Validation Loss')
plt.legend(handles=[red_patch, blue_patch])

# Only show integer values for x axis
xint = range(0, len(training_loss))
plt.xticks(xint)

# Axis labels
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
