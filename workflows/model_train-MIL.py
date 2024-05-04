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
    g_args = load_config(config_file="config.yaml", key="gpu")
    d_args = load_config(config_file="config.yaml", key="dataset")
    m_args = load_config(config_file="config.yaml", key="model")
    t_args = load_config(config_file="config.yaml", key="trainer")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=g_args["device_index"])

    fdl = FungalDataLoaderMIL(
        d_args["slide_dir"],
        d_args["annot_dir"],
        d_args["data_dir_name"],
    )
    fdl.load_features(
        data_dir=t_args["data_dir"],
        subset_size=t_args["subset_size"],
        batch_size=t_args["batch_size"],
        shuffle=False,
    )

    filter_data = lambda x, y: x
    filter_labels = lambda x, y: y
    get_ds = lambda f: np.concatenate(list(fdl.train_ds.map(f)))
    data, labels = get_ds(filter_data), get_ds(filter_labels)

    kfold = StratifiedKFold(
        n_splits=d_args["n_splits"], shuffle=True, random_state=fdl.seed
    )
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data, labels)):
        if fold not in t_args["folds"]:
            continue

        fold_dir = f"fold_{fold}"
        print(f"Running on fold: {fold}")
        model_params = m_args[f'model-{m_args["_select"]}']

        mt = ModelTrainer(
            exp_base_dir=os.path.join(t_args["exp_base_dir"], t_args["exp_name"]),
            exp_name=fold_dir,
            data_dir=t_args["data_dir"],
            model_args=t_args,
            model_params=model_params,
            model_name=m_args["_select"],
            image_dims=d_args["patch_size"],
            MIL=True,
        )
        mt.train_ds = tf.data.Dataset.from_tensor_slices(
            (data[train_idx], labels[train_idx]),
        ).batch(t_args["batch_size"])
        mt.val_ds = tf.data.Dataset.from_tensor_slices(
            (data[val_idx], labels[val_idx]),
        ).batch(t_args["batch_size"])
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
