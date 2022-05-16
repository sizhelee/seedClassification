from model import cnn, hog, mlp, sift
from utils import io_util, util

import argparse


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
params = vars(parser.parse_args())

model_name = params["model"]

config = io_util.load_yaml(params["config_path"], True)
config["model"]["name"] = model_name

if model_name in ["cnn", "vgg", "resnet"]:
    cnn.main(config)
elif model_name == "hog":
    hog.main(config)
elif model_name == "sift":
    sift.main(config)
elif model_name == "mlp":
    mlp.main(config)
