from model import cnn, hog, mlp, sift
from utils import io_util, util

params = util.parse_args()

model_name = params["model"]

config = io_util.load_yaml(params["config_path"], True)
config["model"]["name"] = model_name
if params["checkpoint"] is not None:
    config["model"]["resume_path"] = params["checkpoint"]

if model_name in ["cnn", "vgg", "resnet"]:
    cnn.main(config)
elif model_name == "hog":
    hog.main(config)
elif model_name == "sift":
    sift.main(config)
elif model_name == "mlp":
    mlp.main(config)
