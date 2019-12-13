from torch import nn, optim
import torch
import torch.nn.functional as F
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, pretrained_model_flag, pretrained_model=None, drop_p=0.5):
        '''
        Build a forward network with arbitrary hidden layers.
        Arguments
            ---------
            input_size (integer): size of the input layer
            output_size (integer): size of the output layer
            hidden_layers (list of integers):, the sizes of each hidden layers
        '''
        super(MLP, self).__init__()
        # hidden layers
        # layer_sizes = [(input_size, hidden_layers[0])] \
        #               + list(zip(hidden_layers[:-1], hidden_layers[1:]))
        layer_sizes = [(input_size, hidden_layers[0])]
        self.hidden_layers = nn.ModuleList([nn.Linear(h1, h2)
                                            for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(drop_p)
        self.pretrained_model_flag = pretrained_model_flag
        if pretrained_model_flag:
            self.resnet = pretrained_model
            for param in self.resnet.parameters():
                param.require_grad = False

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        if self.pretrained_model_flag:
            # flatten inputs
            print(x.shape)
            # resnet_predictions = []
            # for image in x:
            #     resnet_predictions.append(self.resnet(image))
            # x = torch.tensor(resnet_predictions)

            self.resnet.eval()
            x = self.resnet(x)
            resnet_predictions = []
            for pred in x:
                resnet_predictions.append(pred['keypoints'].to(torch.int16).cpu()[0][:10, :2])
            print(resnet_predictions[0].shape)
            print(resnet_predictions[0])
            x = torch.stack(resnet_predictions).to(torch.double)

            print(x.shape)
            for layer in self.hidden_layers:
                x = F.relu(layer(x.T))
                x = self.dropout(x)
            print(x.shape)
            x = self.output(x)
            print(x.shape)
            x = F.softmax(x, dim=10)
        else:
            # print(x.shape)
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
                # x = self.dropout(x)
            x = self.output(x)
            x = F.softmax(x, dim=0)

        return x