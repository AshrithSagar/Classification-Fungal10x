"""
generate_heatmaps.py
"""

import os
import sys

sys.path.append(os.getcwd())
import numpy as np

from utils.config import GPUHandler, load_config
from utils.dataset import FungalDataLoader
from utils.heatmaps import Heatmaps
from utils.model import ModelMaker, ModelTrainer


if __name__ == "__main__":
    g_args = load_config(config_file="config.yaml", key="gpu")
    d_args = load_config(config_file="config.yaml", key="dataset")
    m_args = load_config(config_file="config.yaml", key="model")
    t_args = load_config(config_file="config.yaml", key="trainer")
    h_args = load_config(config_file="config.yaml", key="heatmaps")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=g_args["device_index"])

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

        exp_dir = os.path.join(t_args["exp_base_dir"], t_args["exp_name"], fold_dir)

        preds_file = os.path.join(exp_dir, "preds.csv")
        if t_args["overwrite_preds"] or not os.path.exists(preds_file):
            model_params = m_args[f'model-{m_args["_select"]}']
            mt = ModelTrainer(
                exp_dir=exp_dir,
                data_dir=os.path.join(t_args["data_dir"], fold_dir),
                model_args=t_args,
                model_params=model_params,
            )
            mt.load_dataset(t_args["subset_size"], t_args["use_augment"])
            mdl = ModelMaker(
                model_args=t_args,
                model_params=model_params,
                exp_dir=mt.exp_dir,
                model=m_args["_select"],
            )
            mt.model, mt.callbacks_list, mt.epochs_done = (
                mdl.model,
                mdl.callbacks_list,
                mdl.epochs_done,
            )
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
            save_ext=h_args["file_extension"],
        )
