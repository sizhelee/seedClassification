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
    
    train_X = []
    for img in train_x:
        fd = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2), multichannel=True, feature_vector=True)
        train_X.append(fd)
    train_X = np.array(train_X)

    clf = svm.SVC(C=0.01, kernel="poly")
    clf.fit(train_X, train_y)
    print("Training Score: {}".format(clf.score(train_X, train_y)))
    print("Finished Training!")

    test_X = []
    for img in test_x:
        fd = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2), multichannel=True, feature_vector=True)
        test_X.append(fd)
    test_X = np.array(test_X)

    test_Y = clf.predict(test_X)
    io_util.generate_csv(test_Y, test_img, config["model"]["results"])



if __name__ == "__main__":
    config = io_util.load_yaml("src/experiments/config.yml", True)
    main(config)
