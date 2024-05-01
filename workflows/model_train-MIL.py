"""
model_train-MIL.py
"""

import os
import sys

from tensorflow.python.ops.numpy_ops import np_config

sys.path.append(os.getcwd())
from utils.config import GPUHandler, load_config
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

    model_params = m_args[f'model-{m_args["_select"]}']
    mt = ModelTrainer(
        exp_base_dir=t_args["exp_base_dir"],
        exp_name=t_args["exp_name"],
        data_dir=t_args["data_dir"],
        model_args=t_args,
        model_params=model_params,
        model_name=m_args["_select"],
        image_dims=d_args["patch_size"],
        MIL=True,
    )
    mt.load_MIL_features(
        subset_size=t_args["subset_size"],
        batch_size=t_args["batch_size"],
        shuffle=False,
    )

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
