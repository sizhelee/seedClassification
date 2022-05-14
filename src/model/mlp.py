import os
import sys
sys.path.append(os.path.abspath("."))

from src.utils import io_util
from sklearn.neural_network import MLPClassifier

import numpy as np

def main(config):
    train_x, train_y, _ = io_util.load_img(config["model"]["dataloader"])
    test_x, _, test_img = io_util.load_img(config["model"]["dataloader"], mode="test")

    N = train_x.shape[0]
    train_x = train_x.reshape((N, -1))

    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(512, 256), random_state=1)
    clf.fit(train_x, train_y)
    print("Training Score: {}".format(clf.score(train_x, train_y)))

    N = test_x.shape[0]
    test_x = test_x.reshape((N, -1))

    test_Y = clf.predict(test_x)
    print(test_Y)
    io_util.generate_csv(test_Y, test_img, config["model"]["results"])

if __name__ == "__main__":
    config = io_util.load_yaml("src/experiments/config.yml", True)
    main(config)
