import numpy as np
import torch
import ast

def train(train_loader, valid_loader, model, criterion, optimizer, device,
          n_epochs=100, saved_model='model.pt'):
    '''
    Train the model

    Args:
        train_loader (DataLoader): DataLoader for train Dataset
        valid_loader (DataLoader): DataLoader for valid Dataset
        model (nn.Module): model to be trained on
        criterion (torch.nn): loss funtion
        optimizer (torch.optim): optimization algorithms
        n_epochs (int): number of epochs to train the model
        saved_model (str): file path for saving model

    Return:
        tuple of train_losses, valid_losses
    '''

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity

    train_losses = []
    valid_losses = []

    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        train_num = 0
        for batch in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(torch.DoubleTensor(string_to_list(batch['keypoints'])).to(device))
            # print('output shape: ', output.shape)
            # print('batch shape: ', batch['keypoints'].to(device).shape)
            # return 1, 1
            # calculate the loss

            answer = torch.zeros(len(batch['answer']), 10)
            for i, cl in enumerate(batch['answer']):
                answer[i][int(cl[1:])] = 1.0
            # print(output.shape, answer.shape)
            loss = criterion(output.double(), answer.to(device).double())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * 10
            # print('train step ', train_num)
            train_num += 1
            if train_num > 10:
                break

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        train_num = 0

        for batch in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(torch.DoubleTensor(string_to_list(batch['keypoints'])).to(device))
            # calculate the loss
            answer = torch.zeros(len(batch['answer']), 10)
            for i, cl in enumerate(batch['answer']):
                answer[i][int(cl[1:])] = 1.0
            loss = criterion(output.double(), answer.to(device).double())
            # update running validation loss
            valid_loss += loss.item() * 10
            # print('val step ', train_num)
            train_num += 1
            if train_num > 10:
                break

        # print training/validation statistics
        # calculate average Root Mean Square loss over an epoch
        train_loss = np.sqrt(train_loss / len(train_loader.sampler.indices))
        valid_loss = np.sqrt(valid_loss / len(valid_loader.sampler.indices))

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'
              .format(epoch + 1, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), saved_model)
            valid_loss_min = valid_loss
            print(answer, output)

    return train_losses, valid_losses

def string_to_list(list_str):
    for i in range(len(list_str)):
        list_str[i] = list_str[i].replace('[', '').replace(']', '').replace('\n', '').replace('  ', ' ').split(' ')
        list_str[i] = [int(s) for s in list_str[i]]
    # lists = [int(s) for s in strs]
    # print(list_str)
    return list_str