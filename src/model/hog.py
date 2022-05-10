import os
import sys
sys.path.append(os.path.abspath("."))

from src.utils import io_util
from skimage.feature import hog
from sklearn import svm

import numpy as np

def main(config):
    train_x, train_y, _ = io_util.load_img(config["model"]["dataloader"])
    test_x, _, test_img = io_util.load_img(config["model"]["dataloader"], mode="test")

    pixels_per_cell = (config["model"]["img_encoder"]["hog"]["pixel_x"], config["model"]["img_encoder"]["hog"]["pixel_y"])
    cells_per_block = (config["model"]["img_encoder"]["hog"]["cell_x"], config["model"]["img_encoder"]["hog"]["cell_y"])
    
    train_X = []
    for img in train_x:
        fd = hog(img, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, multichannel=True, feature_vector=True)
        train_X.append(fd)
    train_X = np.array(train_X)

    C = config["model"]["svm"]["C"]
    kernel = config["model"]["svm"]["kernel"]

    clf = svm.SVC(C=C, kernel=kernel)
    clf.fit(train_X, train_y)
    print("Training Score: {}".format(clf.score(train_X, train_y)))
    print("Finished Training!")

    test_X = []
    for img in test_x:
        fd = hog(img, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, multichannel=True, feature_vector=True)
        test_X.append(fd)
    test_X = np.array(test_X)

    test_Y = clf.predict(test_X)
    io_util.generate_csv(test_Y, test_img, config["model"]["results"])



if __name__ == "__main__":
    config = io_util.load_yaml("src/experiments/config.yml", True)
    main(config)
