"""
generate_heatmaps.py
"""

import os
import sys

sys.path.append(os.getcwd())
import numpy as np

from utils.config import Config, GPUHandler, line_separator
from utils.dataset import FungalDataLoader
from utils.heatmaps import Heatmaps
from utils.model import ModelMaker, ModelTrainer


def main(args):
    g_args, d_args, m_args = args["gpu"], args["dataset"], args["model"]
    t_args, h_args = args["trainer"], args["heatmaps"]

    gpu = GPUHandler()
    gpu.set(device_index=g_args["device_index"])

    if h_args["source_dir"]:
        print(f'Loading slides from directory: {h_args["source_dir"]}')
        source_dir_annot = (
            h_args["source_dir_annot"]
            if h_args["source_dir_annot"]
            else h_args["source_dir"]
        )

        fdl = FungalDataLoader(
            h_args["source_dir"],
            source_dir_annot,
        )
        fdl.load_slides()
        fdl.get_downsample_dims(
            size=h_args["downsample_size"],
            factor=h_args["downsample_factor"],
        )
        fdl.slide_dataset = fdl.downsample(fdl.slide_dataset)
        fdl.get_stride(h_args["patch_dims"][0:2], h_args["overlap"])
        all_patches, fdl.patches_shape = fdl.get_patches(fdl.slide_dataset)
        slides = fdl.annot_dataset
        slide_names = fdl.slide_names
        slide_labels = fdl.slide_labels
        print(line_separator)

    else:
        print(f'Loading test slides from directory: {d_args["slide_dir"]}')
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
        all_patches = fdl.x_test_patches
        slides = fdl.x_test_annot.numpy()
        slide_names = fdl.x_test_slide_names
        slide_labels = fdl.y_test_slides

    for fold in t_args["folds"]:
        fold_dir = f"fold_{fold}"
        print(f"Running on fold: {fold}")

        exp_dir = os.path.join(t_args["exp_base_dir"], t_args["exp_name"], fold_dir)
        preds_filename = f"{t_args['predictions_file']}.csv"
        preds_file = os.path.join(exp_dir, preds_filename)

        if t_args["overwrite_preds"] or not os.path.exists(preds_file):
            print(f"Generating predictions to {preds_filename}")
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
                all_patches,
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


if __name__ == "__main__":
    args = Config.from_args("Generate heatmaps for the trained model.")
    main(args)
