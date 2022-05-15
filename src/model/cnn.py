import os
import pdb
import sys
sys.path.append(os.path.abspath("."))

from src.utils import io_util, util
from sklearn.neural_network import MLPClassifier

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, SGD

from tqdm import tqdm, trange

train_on_gpu = torch.cuda.is_available()

class CNN(nn.Module):   
    def __init__(self, cfg, img_w):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, cfg["conv2d_1_out"], kernel_size=cfg["conv2d_1_kernel_size"], stride=1, padding=1),
            nn.BatchNorm2d(cfg["conv2d_1_out"]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=cfg["pool_1_kernel_size"], stride=cfg["pool_1_kernel_size"]),
            nn.Conv2d(cfg["conv2d_1_out"], cfg["conv2d_2_out"], kernel_size=cfg["conv2d_2_kernel_size"], stride=1, padding=1),
            nn.BatchNorm2d(cfg["conv2d_2_out"]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=cfg["pool_2_kernel_size"], stride=cfg["pool_2_kernel_size"]),
        )

        new_img_w = int(img_w / (cfg["pool_1_kernel_size"]*cfg["pool_2_kernel_size"]))

        self.linear_layers = nn.Sequential(
            nn.Linear(cfg["conv2d_2_out"]*new_img_w*new_img_w, 12), 
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def train_cnn(model, train_X, train_Y, test_X, config):

    lr = config["model"]["optimizer"]["lr"]
    if config["model"]["optimizer"]["type"] == "Adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif config["model"]["optimizer"]["type"] == "SGD":
        optimizer = SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    print(model)

    batch_size = config["model"]["cnn"]["batch_size"]
    n_epochs = config["model"]["cnn"]["epoch"]
    max_val_acc = 0

    all_train_acc = []
    all_val_acc = []
    all_loss = []

    N_train = train_X.shape[0]
    train_num = int(N_train*config["model"]["cnn"]["train_percent"])
    val_num = N_train-train_num


    for epoch in range(1, n_epochs+1):

        permutation = torch.randperm(train_X.size()[0])

        training_loss = []
        prediction = []
        gt = []
        for i in range(0,train_num, batch_size):

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_X[indices], train_Y[indices]
            padding = (0, batch_size-batch_x.shape[0])
            batch_x = F.pad(batch_x, (0, 0, 0, 0, 0, 0, 0, batch_size-batch_x.shape[0]))
            batch_y = F.pad(batch_y, padding)

            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs,batch_y.long())
            
            pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            prediction.append(pred)
            gt.append(batch_y)

            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        training_loss = np.average(training_loss)
        prediction = (np.stack(prediction)).reshape(-1)[:train_num]
        gt = (torch.stack(gt).cpu().numpy().reshape(-1))[:train_num]
        train_acc = ((prediction == gt).sum())/prediction.shape[0]

        indices = permutation[train_num:]
        val_X, val_Y = train_X[indices], train_Y[indices]
        if torch.cuda.is_available():
            val_X = val_X.cuda()
        with torch.no_grad():
            outputs = model(val_X)
        val_pred = np.argmax(outputs.cpu().numpy(), axis=1)
        val_acc = ((val_pred == val_Y.numpy()).sum())/val_pred.shape[0]

        if val_acc > max_val_acc:
            if torch.cuda.is_available():
                test_X = test_X.cuda()
            with torch.no_grad():
                outputs = model(test_X)
            test_pred = np.argmax(outputs.cpu().numpy(), axis=1)

        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)
        all_loss.append(training_loss)

        io_util.write_log(train_log, 'epoch[{}], Loss: {}, Train Acc: {}, Val Acc: {}'.format(epoch, training_loss.round(7), train_acc.round(4), val_acc.round(4)))

    util.show_train(all_loss, all_train_acc, all_val_acc)
    return model, test_pred


def main(config, train_log):

    io_util.write_log(train_log, config)

    train_x, train_y, _ = io_util.load_img(config["model"]["dataloader"])
    test_x, _, test_img = io_util.load_img(config["model"]["dataloader"], mode="test")

    N_train, img_size = train_x.shape[0], train_x.shape[1]
    train_x = train_x.reshape((N_train, 3, img_size, -1))
    train_X = torch.from_numpy(train_x) 
    train_Y = torch.from_numpy(train_y)

    N_test, img_size = test_x.shape[0], test_x.shape[1]
    test_x = test_x.reshape((N_test, 3, img_size, -1))
    test_X = torch.from_numpy(test_x)

    model = CNN(config["model"]["cnn"], img_size)
    model, test_pred = train_cnn(model, train_X, train_Y, test_X, config)
    
    io_util.generate_csv(test_pred, test_img, config["model"]["results"])



if __name__ == "__main__":
    config = io_util.load_yaml("src/experiments/config.yml", True)
    train_log = io_util.init_logger(config["model"]["results"]["training_log"], "train")
    main(config, train_log)