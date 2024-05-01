"""
extract_features-CLAM.py
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.getcwd())
from models.resnet_custom import resnet50_baseline
from utils.config import GPUHandler, load_config
from utils.dataset import FungalDataLoaderMIL


if __name__ == "__main__":
    g_args = load_config(config_file="config.yaml", key="gpu")
    d_args = load_config(config_file="config.yaml", key="dataset")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=g_args["device_index"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = resnet50_baseline(pretrained=True)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    fdl = FungalDataLoaderMIL(
        d_args["slide_dir"],
        d_args["annot_dir"],
        d_args["data_dir_name"],
    )
    fdl.load_slides()
    fdl.create_splits()
    fdl.split_info()

    if d_args["save_slides"]:
        fdl.segregate_slides()
        fdl.save_slides()

    fdl.downsample_slides(
        size=d_args["downsample_size"],
        factor=d_args["downsample_factor"],
    )
    fdl.extract_patches(size=d_args["patch_size"], overlap=d_args["overlap"])
    fdl.save_patches()

    fdl.extract_features_torch(feature_extractor=model, device=device)
    fdl.save_features_torch()
