# Copyright (c) Facebook, Inc. and its affiliates.


import os
from urllib import request
import torch
import pickle

## Define the weights you want and where to store them
dataset = "scannet"
encoder = "_masked" # or ""
epoch = 1080
base_url = "https://dl.fbaipublicfiles.com/3detr/checkpoints"
local_dir = "/tmp/"

### Downloading the weights
weights_file = f"{dataset}{encoder}_ep{epoch}.pth"
metrics_file = f"{dataset}{encoder}_ep{epoch}_metrics.pkl"
local_weights = os.path.join(local_dir, weights_file)
local_metrics = os.path.join(local_dir, metrics_file)

url = os.path.join(base_url, weights_file)
request.urlretrieve(url, local_weights)
print(f"Downloaded weights from {url} to {local_weights}")

url = os.path.join(base_url, metrics_file)
request.urlretrieve(url, local_metrics)
print(f"Downloaded metrics from {url} to {local_metrics}")

# weights can be simply loaded with pytorch
weights = torch.load(local_weights, map_location=torch.device("cpu"))
print("Weights loaded successfully.")

# metrics can be loaded with pickle
with open(local_metrics, "rb") as fh:
    metrics = pickle.load(fh)
print("Metrics loaded successfully.")