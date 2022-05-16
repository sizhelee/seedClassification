import numpy as np
import torch

import matplotlib.pyplot as plt
import argparse



def label2onehot(label):
    '''
    input: label(tensor) N*1
    output: onehot(tensor) N*12
    '''
    N = label.shape[0]
    onehot = torch.zeros(N, 12)
    one = torch.ones_like(onehot)

    onehot.scatter_(dim=1, index=label.reshape(N,-1).long(), src=one)
    return onehot


def cal_acc(prediction, gt, train_num, distribute=True):
    if not distribute:
        acc = ((prediction == gt).sum())/prediction.shape[0]
    else:
        prediction = (np.stack(prediction)).reshape(-1)[:train_num]
        gt = (torch.stack(gt).cpu().numpy().reshape(-1))[:train_num]
        acc = ((prediction == gt).sum())/prediction.shape[0]
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description="Seed Classification")
    parser.add_argument(
        "--config_path",
        default="src/config.yml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--model",
        default="cnn", 
        help="type of model",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint path",
    )
    params = vars(parser.parse_args())

    return params


def show_train(model_name, loss_value, accuracy_train, accuracy_val):

    plt.subplot(121)
    plt.title("Loss")
    plt.xlabel("iteration")
    plt.plot(loss_value, label="loss")

    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(accuracy_train, label="train_acc")
    plt.plot(accuracy_val, label="val_acc")

    plt.suptitle("Results")
    plt.legend()
    plt.savefig("./results/{}/train.png".format(model_name))
    plt.show()
    