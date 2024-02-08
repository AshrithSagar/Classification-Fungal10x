"""
create_dataset.py
"""

import os
from utils.config import load_config
from utils.dataset import FungalDataLoader


if __name__ == "__main__":
    args = load_config(config_file="config.yaml")

    fdl = FungalDataLoader(
        args["data_dir_name"],
        args["slide_dir"],
        args["annot_dir"],
    )

    fdl.load_slides()
    fdl.create_splits()
    fdl.create_kfold_splits()
    fdl.split_info()
    fdl.segregate_slides()
    fdl.save_slides()
    fdl.downsample_slides()
    fdl.extract_patches()
    fdl.get_annotations()
    fdl.verify_annotations()
    fdl.filter_patch_annotations()
    fdl.patches_info()
    fdl.patches_split_info()
    fdl.segregate_patches()
    fdl.perform_augmentation()
    fdl.augment_info()
    fdl.save_patches()
    fdl.zip_data_dir()
    fdl.dataset_info()
