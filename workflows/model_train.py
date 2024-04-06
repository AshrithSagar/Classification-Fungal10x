"""
model_train.py
"""

import os
import sys

sys.path.append(os.getcwd())
from models.model import get_model
from utils.config import GPUHandler, load_config
from utils.trainer import ModelTrainer


if __name__ == "__main__":
    g_args = load_config(config_file="config.yaml", key="gpu")
    t_args = load_config(config_file="config.yaml", key="trainer")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=g_args["device_index"])

    for fold in t_args["folds"]:
        fold_dir = f"fold_{fold}"
        print(f"Running on fold: {fold}")

        mt = ModelTrainer(
            exp_base_dir=os.path.join(
                t_args["exp_base_dir"], t_args["model_args"]["exp_name"]
            ),
            exp_name=fold_dir,
            data_dir=os.path.join(t_args["model_args"]["data_dir"], fold_dir),
            model_args=t_args["model_args"],
            model_params=t_args["model_params"],
        )
        mt.load_dataset(t_args["subset_size"], t_args["use_augment"])
        t_args["model_args"]["exp_dir"] = mt.exp_dir

        mt = get_model(mt, t_args)
        mt.info()
        mt.train()
        mt.evaluate()
