"""
create_dataset-MIL.py
"""

import os
import sys

sys.path.append(os.getcwd())
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

from utils.config import GPUHandler, load_config
from utils.dataset import FungalDataLoaderMIL

if __name__ == "__main__":
    config_file = "config.yaml"
    g_args = load_config(config_file, key="gpu")
    d_args = load_config(config_file, key="dataset")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=g_args["device_index"])

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

    feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    fdl.extract_features(feature_extractor)
