import os
import sys
sys.path.append(os.path.abspath("."))

from src.utils import io_util
from skimage.feature import hog
from sklearn import svm

import numpy as np
import cv2

# calculate image feature for keypoints in an image
def calcSiftFeature(img):
    sift = cv2.xfeatures2d.SIFT_create()
    _, features = sift.detectAndCompute(img, None)
    return features


# calculate feature center
def learnVocabulary(features, cfg):

    wordCnt = cfg["wordCnt"]
    max_iter = cfg["cret_maxiter"]
    eps = cfg["cret_eps"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

    flags = cv2.KMEANS_RANDOM_CENTERS
    kmeans_maxiter = cfg["kmeans_maxiter"]
    compactness, labels, centers = cv2.kmeans(features, wordCnt, None, criteria, kmeans_maxiter, flags)
    return centers


# calculate image feature of an image
def calcFeatVec(features, centers, cfg):
    wordCnt = cfg["wordCnt"]
    featVec = np.zeros((1, wordCnt))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (wordCnt, 1)) - centers
        sqSum = (diffMat**2).sum(axis=1)
        dist = sqSum**0.5
        sortedIndices = dist.argsort()
        featVec[0][sortedIndices[0]] += 1
    return featVec


# build word bags
def build_center(train_x, cfg):

    features = np.float32([]).reshape(0, 128)
    for img in train_x:
        img_f = calcSiftFeature(img)
        features = np.append(features, img_f, axis=0)

    centers = learnVocabulary(features, cfg)
    wordbag_path = cfg["wordbag_path"]
    np.save(wordbag_path, centers)
    print('wordbag:',centers.shape)

    return centers


# calculate feature of a dataset
def cal_vec(centers, data_x, cfg):
    # centers = np.load(config["preprocess"]["wordbag_path"])
    wordCnt = cfg["wordCnt"]
    X = np.float32([]).reshape(0, wordCnt)#存放训练集图片的特征

    for img in data_x:
        img_f = calcSiftFeature(img)
        img_vec = calcFeatVec(img_f, centers, cfg)
        X = np.append(X, img_vec, axis=0)

    print('data_vec:',X.shape)
    print('image features vector done!')
    return X


def main(config):
    cfg_sift = config["model"]["img_encoder"]["sift"]

    train_x, train_y, _ = io_util.load_img(config["model"]["dataloader"])
    test_x, _, test_img = io_util.load_img(config["model"]["dataloader"], mode="test")

    centers = build_center(train_x, cfg_sift)

    train_X = cal_vec(centers, train_x, cfg_sift)   # 4750*50

    C = config["model"]["svm"]["C"]
    kernel = config["model"]["svm"]["kernel"]
    clf = svm.SVC(C=C, kernel=kernel)
    clf.fit(train_X, train_y)
    print("Training Score: {}".format(clf.score(train_X, train_y)))

    test_X = cal_vec(centers, test_x, cfg_sift)
    test_X = np.float32(test_X)

    test_Y = clf.predict(test_X)
    io_util.generate_csv(test_Y, test_img, config["model"])
