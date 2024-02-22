"""
create_dataset.py
"""

import os
import sys

sys.path.append(os.getcwd())
import tensorflow as tf

from utils.config import GPUHandler, load_config
from utils.dataset import FungalDataLoader

if __name__ == "__main__":
    args = load_config(config_file="config.yaml", key="dataset")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=-1)

    fdl = FungalDataLoader(
        args["data_dir_name"],
        args["slide_dir"],
        args["annot_dir"],
    )
    fdl.load_slides()
    fdl.create_splits()
    fdl.split_info()

    if args["save_slides"]:
        fdl.segregate_slides()
        fdl.save_slides()

    fdl.downsample_slides(
        size=args["downsample_size"],
        factor=args["downsample_factor"],
    )
    fdl.extract_patches()

    fdl.get_annotations()
    fdl.verify_annotations()
    fdl.filter_patch_annotations()
    fdl.patches_info()

    transformations = [
        lambda x: tf.image.flip_left_right(x),
        lambda x: tf.image.flip_up_down(x),
        lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x)),
        lambda x: tf.image.rot90(x, k=1),  # 90 degree
        lambda x: tf.image.rot90(x, k=2),  # 180 degree
        lambda x: tf.image.rot90(x, k=3),  # 270 degree
    ]
    fdl.segregate_patches()
    fdl.perform_augmentation(transformations, use_augment=args["use_augment"])
    fdl.augment_info()

    for fold in fdl.create_kfold_splits(n_splits=args["n_splits"]):
        fdl.save_patches()

    if args["create_zip"]:
        fdl.zip_data_dir()

    # fdl.dataset_info()
