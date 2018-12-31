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
image, label = next(iter(trainloader))
helper.imshow(image[0,:])
plt.show()

# Parmeters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_dimension = 784
hidden_layer_1_dimension = 256
hidden_layer_2_dimension = 128
hidden_layer_3_dimension = 64
output_dimension = 10
dropout_rate = 0.2
epochs = 2
load_model = True
save_model = False

# Model Architecture
model = nn.Sequential(
            nn.Linear(input_dimension, hidden_layer_1_dimension),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_1_dimension, hidden_layer_2_dimension),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_2_dimension, hidden_layer_3_dimension),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_3_dimension, output_dimension),
            nn.LogSoftmax(dim=1)
        ).to(device)

print(model)

if load_model:
    state_dict = torch.load("checkpoint.pth")
    model.load_state_dict(state_dict)

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

training_loss = []
validation_loss = []
# Training loop
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        log_ps = model(images).to(device)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluate model on validation set
    # Turn off gradients
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            # Set model to evaluation mode
            # Turn off dropout for inference
            model.eval()

            for images, labels in testloader:
                images = images.view(images.shape[0], -1).to(device)
                labels = labels.to(device)

                log_ps = model(images).to(device)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                # Get the top-1 prediction and class
                top_p, top_class = ps.topk(1, dim=1)

                # top_class is tensor with shape (64, 1)
                # labels is 1D with shape (64)
                # labels has to be reshaped to have equality work
                num_correct_predictions = top_class == labels.view(*top_class.shape).to(device)

                # correct_prediction is a ByteTensor and has to be converted to a FloatTensor for torch.mean
                accuracy += torch.mean(num_correct_predictions.type(torch.FloatTensor))

        training_loss.append(running_loss/len(trainloader))
        validation_loss.append(test_loss/len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

        # Set model back to train mode
        model.train()

if save_model:
    torch.save(model.state_dict(), 'checkpoint.pth')

# Plot training and validation loss over epochs
plt.plot(training_loss, label="Training loss")
plt.plot(validation_loss, label="Test loss")
plt.legend(frameon=False)

# Axis labels
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()


# Test out network
model.eval()

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]

# Convert 2D image to 1D vector
img = img.resize_(1, 784).to(device)

with torch.no_grad():
    output = model.forward(img)

# Class probabilities (softmax) for img
ps = torch.exp(output).cpu()

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28).cpu(), ps, version='Fashion')
plt.show()
