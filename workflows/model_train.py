"""
model_train.py
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import GPUHandler, load_config
from utils.trainer import ModelTrainer
from models.clamSB_tf import CLAM_SB
from models.EfficientNetB0 import get_EfficientNetB0
from models.ResNet50 import get_ResNet50


if __name__ == "__main__":
    t_args = load_config(config_file="config.yaml", key="trainer")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=t_args["gpu"])

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

        if t_args["model"] == "CLAM_SB":
            t_args["model-CLAM_SB"]["model_args"]["exp_dir"] = mt.exp_dir
            mt.model = CLAM_SB(t_args["model-CLAM_SB"])
        elif t_args["model"] == "EfficientNetB0":
            t_args["model-EfficientNetB0"]["model_args"]["exp_dir"] = mt.exp_dir
            mt.model, mt.callbacks_list, mt.epochs_done = get_EfficientNetB0(
                t_args["model-EfficientNetB0"]
            )
        elif t_args["model"] == "ResNet50":
            t_args["model-ResNet50"]["model_args"]["exp_dir"] = mt.exp_dir
            mt.model, mt.callbacks_list, mt.epochs_done = get_ResNet50(
                t_args["model-ResNet50"]
            )
        else:
            raise ValueError("Invalid model")

        mt.info()
        mt.train()
        mt.evaluate()
