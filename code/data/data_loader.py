from torchvision import transforms
import pandas as pd
from pathlib import Path
import torch
from keypoint_dataset import Normalize, ToTensor, KeypointsDataset, prepare_train_valid_loaders
from  mlp import MLP, CNN
from train import train
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure
from torch import optim

path = '../../dataset/state-farm-distracted-driver-detection/'

IMG_SIZE = 64

driver_imgs_list = pd.read_csv(path + 'driver_imgs_list.csv')


# how many samples per batch to load
batch_size = 128
# percentage of training set to use as validation
valid_size = 0.2

# Define a transform to normalize the data
tsfm = transforms.Compose([Normalize(), ToTensor()])

# Load the training data and test data
trainset = KeypointsDataset(train_df, transform=tsfm)
testset = KeypointsDataset(test_df, train=False, transform=tsfm)

# prepare data loaders
train_loader, valid_loader = prepare_train_valid_loaders(trainset,
                                                         valid_size,
                                                         batch_size)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MLP(input_size=IMG_SIZE*IMG_SIZE*3, output_size=48,
#             hidden_layers=[128, 64], drop_p=0.1)
# model = model.to(device)
# model = model.double()

# criterion = torch.nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.003)

# train_losses, valid_losses = train(train_loader, valid_loader,
#                                    model,criterion, optimizer, device,
#                                    n_epochs=50,
#                                    saved_model='model.pt')

model = CNN(outputs=48)
model = model.to(device)
model = model.double()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
train(train_loader, valid_loader, model, criterion, optimizer, device, n_epochs=50, saved_model='cnn2.pt')

# print(train_losses, valid_losses)
torch.save(model.state_dict(), '../model_saved')