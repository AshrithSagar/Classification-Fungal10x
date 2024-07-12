"""
model_train-MIL.py
"""

import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.ops.numpy_ops import np_config

sys.path.append(os.getcwd())
from utils.config import GPUHandler, load_config
from utils.dataset import FungalDataLoaderMIL
from utils.model import ModelMaker, ModelTrainer

np_config.enable_numpy_behavior()


if __name__ == "__main__":
    config_file = "config.yaml"
    g_args = load_config(config_file, key="gpu")
    d_args = load_config(config_file, key="dataset")
    m_args = load_config(config_file, key="model")
    t_args = load_config(config_file, key="trainer")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=g_args["device_index"])

    fdl = FungalDataLoaderMIL(
        d_args["slide_dir"],
        d_args["annot_dir"],
        d_args["data_dir_name"],
    )
    fdl.load_features(
        data_dir=t_args["features_dir"],
        subset_size=t_args["subset_size"],
        batch_size=t_args["batch_size"],
        shuffle=False,
    )

    for fold_data in fdl.create_kfold_splits(
        n_splits=d_args["n_splits"],
        batch_size=t_args["batch_size"],
        run_only=t_args["folds"],
    ):
        fold_dir = f"fold_{fdl.fold}"
        print(f"Running on fold: {fdl.fold}")
        model_params = m_args[f'model-{m_args["_select"]}']

        mt = ModelTrainer(
            exp_base_dir=os.path.join(t_args["exp_base_dir"], t_args["exp_name"]),
            exp_name=fold_dir,
            data_dir=t_args["features_dir"],
            model_args=t_args,
            model_params=model_params,
            model_name=m_args["_select"],
            image_dims=d_args["patch_size"],
            MIL=True,
        )
        mt.train_ds, mt.val_ds = fold_data
        mt.test_ds = fdl.test_ds

        mdl = ModelMaker(
            model_args=t_args,
            model_params=model_params,
            exp_dir=mt.exp_dir,
            model=mt.model_name,
        )
        mt.model, mt.callbacks_list, mt.epochs_done = (
            mdl.model,
            mdl.callbacks_list,
            mdl.epochs_done,
        )

        mt.info()
        mt.train()
        mt.evaluate()
