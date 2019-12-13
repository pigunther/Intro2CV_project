import sys

import pandas as pd
import torch
from torch import optim
from torchvision import transforms
from torchvision.models.detection import keypoint_rcnn

from dataset import Normalize, ToTensor, ImagesDataset, prepare_train_valid_loaders, KeypointsDataset

sys.path.insert(1, './pose_estimation')
# print(sys.path)
from keypoint_resnet import MLP
from train import train

path = '../dataset/state-farm-distracted-driver-detection/'

IMG_SIZE = 64

# driver_imgs_list = pd.read_csv(path + 'driver_imgs_list.csv')
keypoints_list = pd.read_csv(path + 'keypoints_list.csv')

# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# Define a transform to normalize the data
tsfm = transforms.Compose([Normalize(), ToTensor()])

# Load the training data and test data
# trainset = ImagesDataset(driver_imgs_list, transform=tsfm)
trainset = KeypointsDataset(keypoints_list, transform=tsfm)
# prepare data loaders
train_loader, valid_loader = prepare_train_valid_loaders(trainset,
                                                         valid_size,
                                                         batch_size)

# print(train_loader)

# for batch in train_loader:
#     print(batch)
#     plt.imshow(batch['image'][0].permute(2, 1, 0))
#     plt.title(batch['answer'][0])
#     plt.show()
#     break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model = keypoint_rcnn.keypointrcnn_resnet50_fpn(pretrained=True)
pretrained_model.eval()
model = MLP(input_size=batch_size, output_size=10, hidden_layers=[10, 10], pretrained_model_flag=False, drop_p=0.1)
model = model.to(device)
model = model.double()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
train(train_loader, valid_loader, model, criterion, optimizer, device, n_epochs=150, saved_model='model.pt')
