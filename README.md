# Seed Classification

 Seed Classification Homework for CVDL 2022.

 Here is the [Kaggle Link](https://www.kaggle.com/competitions/plant-seedlings-classification).

## Introduction

A model to deal with the plant seed classification task, compared the performance of traditional cv methods(SIFT+KMeans, HOG+SVM, kernelSVM) and deep learning methods(MLP, CNN, VGG, ResNet with various optimizers) and also implement some data augmentation and regularization.

## Main Results

Method          | sift+kMeans   | hog+SVM   | MLP   | CNN   | ResNet    | VGG
----------------|---------------|-----------|-------|-------|----------|----
Test Acc   | 23.866        | 31.989    | 36.083| 54.408| 52.141    | 70.025

## Structure

```python
seedClassification
        ├──── data
        ├──── src
        │       ├──── model
        │       │       ├──── cnn.py
        │       │       ├──── hog.py
        │       │       ├──── mlp.py
        │       │       ├──── models.py
        │       │       └──── sift.py
        │       ├──── utils
        │       │       ├──── img_util.py
        │       │       ├──── io_util.py
        │       │       └──── util.py
        │       ├──── config.yml
        │       ├──── train.py
        │       └──── test.py
        └──── results
```

### 1. Dependencies

This repository is implemented based on [Pytorch](https://pytorch.org/) with Anaconda.

```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
conda install tqdm, opencv, sklearn
```

### 2. Prepare Data

Download the plant-seedlings-classification dataset using the follow Kaggle API

```bash
kaggle competitions download -c plant-seedlings-classification
```

Unzip the dataset and place it in the `/data` folder.

### 3. Quick Start

This code below will load data to your GPU, if you do not have a GPU or do not have enough memory on GPU, set `use_gpu` of the related model to `False`.

#### To train the model

```bash
python src/train.py --config_path src/config.yml \
                    --model cnn \
```

The parameter `model` can be chosen among `hog`, `sift`, `mlp`, `cnn`, `resnet`, `vgg`.

#### To test the pre-train model

```bash
python src/test.py --config_path src/config.yml \
                   --model cnn \
                   --checkpoint CHECKPOINT_PATH\
```

The parameter `model` can be chosen among `cnn`, `resnet`, `vgg`.
