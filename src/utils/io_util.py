import yaml
import os
import cv2
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
from pandas import DataFrame

class2id = {
    "Black-grass": 0,
    "Charlock": 1,
    "Cleavers": 2,
    "Common Chickweed": 3,
    "Common wheat": 4,
    "Fat Hen": 5,
    "Loose Silky-bent": 6,
    "Maize": 7,
    "Scentless Mayweed": 8,
    "Shepherds Purse": 9,
    "Small-flowered Cranesbill": 10,
    "Sugar beet": 11
}

id2class = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed",
 "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed", 
 "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet"]


def load_yaml(file_path, verbose=True):
    with open(file_path, "r") as f:
        yml_file = yaml.load(f, Loader=yaml.SafeLoader)
    if verbose:
        print("Load yaml file from {}".format(file_path))
    return yml_file


def load_img(cfg, mode="train"):

    X = []
    Y = []
    img_name = []

    folder_path = cfg["{}_path".format(mode)]
    img_size = (cfg["img_width"], cfg["img_height"])

    if mode == "train":
        folders = os.listdir(folder_path)
        with tqdm(total=4750) as pbar:
            pbar.set_description('Loading {} data from {}'.format(mode, folder_path))
            for folder in folders:
                imgs = os.listdir("{}/{}".format(folder_path, folder))
                for img in imgs:
                    img_path = "{}/{}/{}".format(folder_path, folder, img)
                    I = cv2.imread(img_path)
                    I = cv2.resize(I, img_size, interpolation=cv2.INTER_CUBIC)
                    X.append(I)
                    Y.append(class2id[folder])
                    img_name.append(img)
                    pbar.update(1)

    else:
        imgs = os.listdir(folder_path)
        with tqdm(total=794) as pbar:
            pbar.set_description('Loading {} data from {}'.format(mode, folder_path))
            for img in imgs:
                img_path = "{}/{}".format(folder_path, img)
                I = cv2.imread(img_path)
                I = cv2.resize(I, img_size, interpolation=cv2.INTER_CUBIC)
                X.append(I)
                img_name.append(img)
                pbar.update(1)

    X = np.stack(X)
    Y = np.array(Y)
    print("data size: {}, label size: {}".format(X.shape, Y.shape))
    return X, Y, img_name


def generate_csv(predict, img_name, cfg):
    data = {
        "file": img_name, 
        "species": [id2class[i] for i in predict]
    }
    df = DataFrame(data)
    df.to_csv(cfg["result_path"], index=False)
    