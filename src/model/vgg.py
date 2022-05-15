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
from torchvision import models

from tqdm import tqdm, trange


def train_vgg(model, train_X, train_Y, test_X, config):

    lr = config["model"]["optimizer"]["lr"]
    if config["model"]["optimizer"]["type"] == "Adam":
        optimizer = Adam(model.classifier[6].parameters(), lr=lr)
    elif config["model"]["optimizer"]["type"] == "SGD":
        optimizer = SGD(model.classifier[6].parameters(), lr=lr)
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    batch_size = config["model"]["vgg"]["batch_size"]
    n_epochs = config["model"]["vgg"]["epoch"]

    N_train = train_X.shape[0]
    train_num = int(N_train*config["model"]["cnn"]["train_percent"])


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

        io_util.write_log(train_log, 'epoch[{}], Loss: {}, Train Acc: {}'.format(epoch, training_loss.round(7), train_acc.round(4)))

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

    model = models.vgg16_bn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Sequential(nn.Linear(4096, 12))
    for param in model.classifier[6].parameters():
        param.requires_grad = True

    model, test_pred = train_vgg(model, train_X, train_Y, test_X, config)
    
    # io_util.generate_csv(test_pred, test_img, config["model"]["results"])



if __name__ == "__main__":
    config = io_util.load_yaml("src/experiments/config.yml", True)
    train_log = io_util.init_logger(config["model"]["results"]["training_log"], "train")
    main(config, train_log)