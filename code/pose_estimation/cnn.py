import torch
import torch.nn.functional as F
from torch import nn


class MyCNN(nn.Module):
    def __init__(self):
        '''
        Build a forward network with arbitrary hidden layers.
        Arguments
            ---------
            input_size (integer): size of the input layer
            output_size (integer): size of the output layer
            hidden_layers (list of integers):, the sizes of each hidden layers
        '''
        super(MyCNN, self).__init__()
        self.image_layer = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(4032, 200),
            nn.ReLU(True),
            nn.BatchNorm1d(200),
            nn.Linear(200, 34),
        )
        self.common_layers = nn.Sequential(
            nn.Linear(68, 50),
            nn.ReLU(True),
            nn.BatchNorm1d(50),
            nn.Linear(50, 45),
            nn.ReLU(True),
            nn.BatchNorm1d(45),
            nn.Linear(45, 30),
            nn.ReLU(True),
            nn.BatchNorm1d(30)
        )
        # self.common_layers = [nn.Linear(68, 50), nn.Linear(50, 45), nn.Linear(45, 30)]
        # self.dropout = nn.Dropout(0.1)
        # self.batch_norms = [nn.BatchNorm1d(50), nn.BatchNorm1d(45), nn.BatchNorm1d(30)]
        self.output = nn.Linear(30, 10)

    def forward(self, x, kp):
        # kp = x[1]
        # x = x[0]
        # print(x)

        x = self.image_layer(x)
        # print(x.shape, kp.shape)
        x = torch.cat([x, kp], 1)
        # print(x.shape)
        # for i in range(len(self.common_layers)):
        #     layer = self.common_layers[i]
        #     blayer = self.batch_norms[i]
        #     x = layer(x)
        #     x = blayer(x)
        #     x = F.relu(x)
        x = self.common_layers(x)
        x = self.output(x)
        x = F.softmax(x)
        return x


CNN_simple = nn.Sequential(
    nn.BatchNorm2d(3),
    nn.Conv2d(3, 32, (3, 3)),
    nn.BatchNorm2d(32),
    nn.MaxPool2d((3, 3)),
    nn.Conv2d(32, 64, (3, 3)),
    nn.ReLU(True),
    nn.BatchNorm2d(64),
    nn.MaxPool2d((3, 3)),
    nn.Flatten(),
    nn.Linear(4032, 200),
    nn.ReLU(True),
    nn.BatchNorm1d(200),
    nn.Linear(200, 10),
    nn.Softmax()
)