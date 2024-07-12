"""
generate_heatmaps-MIL.py
"""

import os
import sys

sys.path.append(os.getcwd())
import numpy as np

from utils.config import GPUHandler, line_separator, load_config
from utils.dataset import FungalDataLoaderMIL
from utils.heatmaps import Heatmaps
from utils.model import ModelMaker, ModelTrainerMIL

if __name__ == "__main__":
    g_args = load_config(config_file="config.yaml", key="gpu")
    d_args = load_config(config_file="config.yaml", key="dataset")
    m_args = load_config(config_file="config.yaml", key="model")
    t_args = load_config(config_file="config.yaml", key="trainer")
    h_args = load_config(config_file="config.yaml", key="heatmaps")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=g_args["device_index"])

    print(f'Loading test slides from directory: {d_args["slide_dir"]}')
    fdl = FungalDataLoaderMIL(
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
    fdl.extract_patches(d_args["patch_size"])
    fdl.get_annotations()
    all_patches = fdl.x_test_patches
    slides = fdl.x_test_annot.numpy()
    slide_names = fdl.x_test_slide_names
    slide_labels = fdl.y_test_slides
    fdl.load_features(
        data_dir=t_args["features_dir"],
        subset_size=t_args["subset_size"],
        batch_size=t_args["batch_size"],
        shuffle=False,
    )
    features = fdl.test_ds
    print(line_separator)

    for fold in t_args["folds"]:
        fold_dir = f"fold_{fold}"
        print(f"Running on fold: {fold}")

        exp_dir = os.path.join(t_args["exp_base_dir"], t_args["exp_name"], fold_dir)
        preds_filename = f"{t_args['predictions_file']}.csv"
        preds_file = os.path.join(exp_dir, preds_filename)

        if t_args["overwrite_preds"] or not os.path.exists(preds_file):
            print(f"Generating predictions to {preds_filename}")
            model_params = m_args[f'model-{m_args["_select"]}']
            mt = ModelTrainerMIL(
                exp_dir=exp_dir,
                data_dir=t_args["features_dir"],
                model_args=t_args,
                model_params=model_params,
            )
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
                features,
                save_file=preds_filename,
                overwrite=t_args["overwrite_preds"],
            )

        else:
            print(f"Loading predictions from {preds_filename}")
            predictions = np.loadtxt(preds_file, delimiter=",")

        hm = Heatmaps(exp_dir)
        hm.save(
            save_dir=h_args["save_dir"],
            slides=slides,
            slide_names=slide_names,
            predictions=predictions,
            slide_labels=slide_labels,
            patches_shape=fdl.patches_shape,
            patch_size=h_args["patch_dims"],
            cmap=h_args["cmap"],
            overlap=h_args["overlap"],
            percentile_scale=h_args["percentile_scale"],
            percentile_score=h_args["percentile_score"],
            alpha=h_args["alpha"],
            blur=h_args["blur"],
            invert_preds=h_args["invert_preds"],
            use_plt=h_args["use_plt"],
            save_ext=h_args["file_extension"],
        )
