import os
import sys
sys.path.append(os.path.abspath("."))

from src.utils import io_util
from sklearn.neural_network import MLPClassifier

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from tqdm import tqdm, trange

train_on_gpu = torch.cuda.is_available()

class CNN(nn.Module):   
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(8 * 16 * 16, 12)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def main(config):
    train_x, train_y, _ = io_util.load_img(config["model"]["dataloader"])
    test_x, _, test_img = io_util.load_img(config["model"]["dataloader"], mode="test")

    N, img_size = train_x.shape[0], train_x.shape[1]
    train_x = train_x.reshape((N, 3, img_size, -1))
    train_X = torch.from_numpy(train_x)
    train_Y = torch.from_numpy(train_y)

    N, img_size = test_x.shape[0], test_x.shape[1]
    test_x = test_x.reshape((N, 3, img_size, -1))
    test_X = torch.from_numpy(test_x)

    model = CNN()
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    print(model)


    batch_size = 128
    n_epochs = 15

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        permutation = torch.randperm(train_X.size()[0])

        training_loss = []
        for i in tqdm(range(0,train_X.size()[0], batch_size)):

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_X[indices], train_Y[indices]
            
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs,batch_y)

            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        training_loss = np.average(training_loss)
        print('epoch: \t', epoch, '\t training loss: \t', training_loss)




if __name__ == "__main__":
    config = io_util.load_yaml("src/experiments/config.yml", True)
    main(config)