import numpy as np
import torch

import matplotlib.pyplot as plt

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

def show_train(loss_value, accuracy_train, accuracy_val):

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
    plt.savefig("./results/cnn/train.png")
    plt.show()
    