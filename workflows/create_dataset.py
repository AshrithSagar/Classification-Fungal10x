"""
create_dataset.py
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.append(os.getcwd())
from utils.config import GPUHandler, load_config
from utils.dataset import FungalDataLoader


if __name__ == "__main__":
    args = load_config(config_file="config.yaml", key="dataset")

    gpu = GPUHandler()
    gpu.check()

    fdl = FungalDataLoader(
        args["data_dir_name"],
        args["slide_dir"],
        args["annot_dir"],
    )
    fdl.load_slides()
    fdl.create_splits()
    for fold in fdl.create_kfold_splits():
        fdl.split_info()
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
        fdl.patches_split_info()
        fdl.segregate_patches()
        fdl.perform_augmentation(use_augment=args["use_augment"])
        fdl.augment_info()
        fdl.save_patches()
        if args["create_zip"]:
            fdl.zip_data_dir()
        # fdl.dataset_info()
