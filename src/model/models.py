import torch.nn as nn
from torchvision import models
from torch.optim import Adam, SGD, Adagrad


class CNN(nn.Module):   
    def __init__(self, cfg, img_w):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, cfg["conv2d_1_out"], kernel_size=cfg["conv2d_1_kernel_size"], stride=1, padding=1),
            nn.BatchNorm2d(cfg["conv2d_1_out"]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=cfg["pool_1_kernel_size"], stride=cfg["pool_1_kernel_size"]),
            nn.Conv2d(cfg["conv2d_1_out"], cfg["conv2d_2_out"], kernel_size=cfg["conv2d_2_kernel_size"], stride=1, padding=1),
            nn.BatchNorm2d(cfg["conv2d_2_out"]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=cfg["pool_2_kernel_size"], stride=cfg["pool_2_kernel_size"]),
        )

        new_img_w = int(img_w / (cfg["pool_1_kernel_size"]*cfg["pool_2_kernel_size"]))

        self.linear_layers = nn.Sequential(
            nn.Linear(cfg["conv2d_2_out"]*new_img_w*new_img_w, 12), 
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def build_model(name, cfg):
    if name == "cnn":
        model = CNN(cfg["model"]["cnn"], cfg["model"]["dataloader"]["img_width"])
    elif name == "vgg":
        model = models.vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Sequential(nn.Linear(4096, 12))
    elif name == "resnet":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 12)
    return model


def build_optimizer(model, cfg):
    lr = cfg["lr"]
    if cfg["type"] == "Adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif cfg["type"] == "SGD":
        optimizer = SGD(model.parameters(), lr=lr)
    elif cfg["type"] == "Adagrad":
        optimizer = Adagrad(model.parameters(), lr=lr)
    else:
        raise NotImplementedError
    return optimizer