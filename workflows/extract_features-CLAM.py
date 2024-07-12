"""
extract_features-CLAM.py
"""

import os
import sys

import torch

sys.path.append(os.getcwd())
from models.resnet_custom import resnet50_baseline
from utils.config import Config, GPUHandler
from utils.dataset import FungalDataLoaderMIL


def main(args):
    g_args, d_args = args["gpu"], args["dataset"]

    gpu = GPUHandler()
    gpu.set(device_index=g_args["device_index"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    feature_extractor = resnet50_baseline(pretrained=True)
    fdl.extract_features_torch(feature_extractor, device=device)
    fdl.save_features_torch()


if __name__ == "__main__":
    args = Config.from_args("Extract features from the dataset.")
    main(args)
