"""
generate_heatmaps.py
"""

import os
import sys

sys.path.append(os.getcwd())
import numpy as np
from models.EfficientNetB0 import get_EfficientNetB0

from utils.config import GPUHandler, load_config
from utils.dataset import FungalDataLoader
from utils.heatmaps import Heatmaps
from utils.trainer import ModelTrainer


if __name__ == "__main__":
    d_args = load_config(config_file="config.yaml", key="dataset")
    t_args = load_config(config_file="config.yaml", key="trainer")
    h_args = load_config(config_file="config.yaml", key="heatmaps")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=t_args["gpu"])

    fdl = FungalDataLoader(
        d_args["slide_dir"],
        d_args["annot_dir"],
    )
    fdl.load_slides()
    fdl.create_splits()
    fdl.split_info()
    fdl.downsample_slides(
        size=d_args["downsample_size"],
        factor=d_args["downsample_factor"],
    )
    fdl.extract_patches()
    fdl.get_annotations()

    for fold in t_args["folds"]:
        fold_dir = f"fold_{fold}"
        print(f"Running on fold: {fold}")

        exp_dir = os.path.join(
            t_args["exp_base_dir"], t_args["model_args"]["exp_name"], fold_dir
        )

        preds_file = os.path.join(exp_dir, "preds.csv")
        if t_args["overwrite_preds"] or not os.path.exists(preds_file):
            mt = ModelTrainer(
                exp_dir=exp_dir,
                data_dir=os.path.join(t_args["model_args"]["data_dir"], fold_dir),
                model_args=t_args["model_args"],
                model_params=t_args["model_params"],
            )
            mt.load_dataset(use_augment=t_args["use_augment"])
            t_args["model_args"]["exp_dir"] = mt.exp_dir
            mt.model, mt.callbacks_list, mt.epochs_done = get_EfficientNetB0(t_args)
            mt.info()

            predictions = mt.predict(
                fdl.x_test_patches, overwrite=t_args["overwrite_preds"]
            )
        else:
            predictions = np.loadtxt(preds_file, delimiter=",")

        hm = Heatmaps(exp_dir)
        hm.save(
            save_dir=h_args["save_dir"],
            slides_annot=fdl.x_test_annot,
            slide_names=fdl.x_test_slide_names,
            predictions=predictions,
            slide_labels=fdl.y_test_slides,
            patches_shape=fdl.patches_shape,
            patch_size=h_args["patch_size"],
            cmap=h_args["cmap"],
            overlap=h_args["overlap"],
            percentile_scale=h_args["percentile_scale"],
            percentile_score=h_args["percentile_score"],
            alpha=h_args["alpha"],
            blur=h_args["blur"],
        )
