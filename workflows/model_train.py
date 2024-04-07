"""
model_train.py
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import GPUHandler, load_config
from utils.model import ModelMaker, ModelTrainer


if __name__ == "__main__":
    g_args = load_config(config_file="config.yaml", key="gpu")
    m_args = load_config(config_file="config.yaml", key="model")
    t_args = load_config(config_file="config.yaml", key="trainer")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=g_args["device_index"])

    for fold in t_args["folds"]:
        fold_dir = f"fold_{fold}"
        print(f"Running on fold: {fold}")
        model_params = m_args[f'model-{m_args["_select"]}']

        mt = ModelTrainer(
            exp_base_dir=os.path.join(t_args["exp_base_dir"], t_args["exp_name"]),
            exp_name=fold_dir,
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
        mt.train()
        mt.evaluate()
