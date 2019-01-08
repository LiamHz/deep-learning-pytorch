import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import models

from helper import *

# Get the "features" portion of VGG19
# The classifier portion is not needed for style transfer
vgg = models.vgg19(pretrained=True).features

# Freeze all VGG parameters
# Since only activations are measured for each conv layer
# Network weights aren't modified
for param in vgg.parameters():
    param.requires_grad_(False)

# Move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

print(vgg)

# Load in content and style image
content = load_image('images/gandalf.jpg').to(device)
# Resize style to match content, makes code easier
style = load_image('images/tokidoki.jpg', shape=content.shape[-2:]).to(device)

# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))
plt.show()

def get_features(image, model, layers=None):
    """
    Run an image forward through a model and get the features for a set of layers.
    Default layers are for VGGNet matching Gatys et al (2016)
    """

    # Dictionary of layers needed from VGG for style transfer
    # Style representation is first conv layer from each block
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # Content representation
                  '28': 'conv5_1'}

    features = {}
    x = image.to(device)
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

def gram_matrix(tensor):
    """
    Calculate the Gram Matrix of a given tensor
    """

    # Get the batch_size, depth, height, and width of the Tensor
    # reshape it, so we're multiplying the features for each channel
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)

    # Calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram

# Get content and style features only once before forming the target image
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create a third target image (copy of content iamge)
# This is the image that will have its style iteratively changed
target = content.clone().requires_grad_(True).to(device)

# Weights for each style layer
# Weighting earlier layers more will result in larger style artifacts
# conv4_2 is excluded because it is the content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

# A higher style weight may make content unrecognizable
content_weight = 1  # alpha in paper
style_weight = 1e6  # beta in paper

# Show target image every x steps
show_every = 400

# Optimizer hyperparameters
optimizer = optim.Adam([target], lr=0.003)

# How many iterations to update content image
steps = 2000

print("Beginning training")
for ii in range(1, steps+1):
    if ii % 100 == 0:
        print("Step #", ii, '/', steps+1)

    # Calculate the content loss
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    # Initialize style loss to 0
    # The world was perfect before the model started training
    style_loss = 0

    # Iterate through each style layer and add to the style loss
    for layer in style_weights:
        # Get the target (goal) style representation for the layer
        target_feature = target_features[layer]
        _, d, h, w = target_feature.shape

        # Calculate the target gram matrix
        target_gram = gram_matrix(target_features[layer])

        # Get the style representation
        style_gram = style_grams[layer]

        # Calculate the weighted style loss for one layer
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)


    #Calculate the *total* loss
    total_loss = (content_loss * content_weight) + (style_loss * style_weight)

    # Update target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Display intermediate images and print the loss
    if  ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()


# Display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))
plt.show()
