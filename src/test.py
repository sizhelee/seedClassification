from utils import io_util, util

import numpy as np
import torch


params = util.parse_args()

model_name = params["model"]
config = io_util.load_yaml(params["config_path"], True)
config["model"]["name"] = model_name

assert params["checkpoint"] is not None
assert model_name in ["cnn", "vgg", "resnet"]

test_x, _, test_img = io_util.load_img(config["model"]["dataloader"], mode="test")
N_test, img_size = test_x.shape[0], test_x.shape[1]
test_x = test_x.reshape((N_test, 3, img_size, -1))
test_X = torch.from_numpy(test_x)

model = io_util.load_checkpoint(params["checkpoint"], model_name, config)
with torch.no_grad():
    outputs = model(test_X)
test_pred = np.argmax(outputs.cpu().numpy(), axis=1)
io_util.generate_csv(test_pred, test_img, config["model"], epoch=-1, verbose=False)