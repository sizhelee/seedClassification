import os
import pdb
import sys
sys.path.append(os.path.abspath("."))
import json

from src.utils import io_util, util
from model.models import build_model, build_optimizer

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm, trange
import argparse


def train(model, train_X, train_Y, test_X, config, train_log, test_img):

    model_name = config["model"]["name"]

    if torch.cuda.is_available():
        train_on_gpu = config["model"][model_name]["use_gpu"]
    else:
        train_on_gpu = False
    io_util.write_log(train_log, model)

    # build optimizer
    optimizer = build_optimizer(model, config["model"]["optimizer"])
    
    # build criterion
    criterion = nn.CrossEntropyLoss()

    if train_on_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    # get train parameters
    batch_size = config["model"][model_name]["batch_size"]
    n_epochs = config["model"][model_name]["epoch"]
    max_val_acc = 0

    N_train = train_X.shape[0]
    train_percent = config["model"][model_name]["train_percent"]
    train_num = int(N_train*train_percent)

    # begin training
    all_train_acc = []
    all_val_acc = []
    all_loss = []

    for epoch in range(1, n_epochs+1):

        permutation = torch.randperm(train_X.size()[0])

        # training process
        training_loss = []
        prediction = []
        gt = []
        for i in range(0, train_num-batch_size+1, batch_size):

            # generate a batch of data and label
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_X[indices], train_Y[indices]
            if train_on_gpu:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
            # calculate the grad and update the network
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs,batch_y.long())
            loss.backward()
            optimizer.step()

            # save the predictions
            pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            prediction.append(pred)
            gt.append(batch_y)
            training_loss.append(loss.item())
            
        training_loss = np.average(training_loss)
        train_acc = util.cal_acc(prediction, gt, train_num)


        # validing process
        indices = permutation[train_num:]
        val_X, val_Y = train_X[indices], train_Y[indices]
        if train_on_gpu:
            val_X = val_X.cuda()

        with torch.no_grad():
            outputs = model(val_X)
        val_pred = np.argmax(outputs.cpu().numpy(), axis=1)
        val_acc = util.cal_acc(val_pred, val_Y.numpy(), None, False)
        # val_acc = ((val_pred == val_Y.numpy()).sum())/val_pred.shape[0]

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            if train_on_gpu:
                test_X = test_X.cuda()

            with torch.no_grad():
                outputs = model(test_X)
            test_pred = np.argmax(outputs.cpu().numpy(), axis=1)
            io_util.generate_csv(test_pred, test_img, config["model"], epoch=epoch, verbose=False)

        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)
        all_loss.append(training_loss)

        io_util.write_log(train_log, 'epoch[{}], Loss: {}, Train Acc: {}, Val Acc: {}'.format(epoch, training_loss.round(7), train_acc.round(4), val_acc.round(4)))

    util.show_train(model_name, all_loss, all_train_acc, all_val_acc)
    return model, test_pred


def main(config):
    
    model_name = config["model"]["name"]
    train_log = io_util.init_logger(config["model"]["results"]["result_path"], "train", model_name)
    io_util.write_log(train_log, json.dumps(config, indent=4))

    # Load train and test data
    train_x, train_y, _ = io_util.load_img(config["model"]["dataloader"])
    test_x, _, test_img = io_util.load_img(config["model"]["dataloader"], mode="test")

    N_train, img_size = train_x.shape[0], train_x.shape[1]
    train_x = train_x.reshape((N_train, 3, img_size, -1))
    train_X = torch.from_numpy(train_x) 
    train_Y = torch.from_numpy(train_y)

    N_test, img_size = test_x.shape[0], test_x.shape[1]
    test_x = test_x.reshape((N_test, 3, img_size, -1))
    test_X = torch.from_numpy(test_x)

    # Load model
    model_name = config["model"]["name"]
    model = build_model(model_name, config)
    model, test_pred = train(model, train_X, train_Y, test_X, config, train_log, test_img)
    
    io_util.generate_csv(test_pred, test_img, config["model"])