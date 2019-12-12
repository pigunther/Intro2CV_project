from torchvision import transforms
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure
from torch import optim
from dataset import Normalize, ToTensor, ImagesDataset, prepare_train_valid_loaders

path = '../dataset/state-farm-distracted-driver-detection/'

IMG_SIZE = 64

driver_imgs_list = pd.read_csv(path + 'driver_imgs_list.csv')


# how many samples per batch to load
batch_size = 128
# percentage of training set to use as validation
valid_size = 0.2

# Define a transform to normalize the data
tsfm = transforms.Compose([Normalize(), ToTensor()])

# Load the training data and test data
trainset = ImagesDataset(driver_imgs_list, transform=tsfm)

# prepare data loaders
train_loader, valid_loader = prepare_train_valid_loaders(trainset,
                                                         valid_size,
                                                         batch_size)

print(train_loader)

for batch in train_loader:
    print(batch)
    plt.imshow(batch['image'][0].permute(2, 1, 0))
    plt.title(batch['answer'][0])
    plt.show()
    break