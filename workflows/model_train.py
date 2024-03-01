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
    args = load_config(config_file="config.yaml", key="trainer")

    gpu = GPUHandler()
    gpu.check()
    gpu.set(device_index=args["gpu"])

    for fold in args["folds"]:
        fold_dir = f"fold_{fold}"
        print(f"Running on fold: {fold}")

        mt = ModelTrainer(
            exp_base_dir=os.path.join(
                args["exp_base_dir"], args["model_args"]["exp_name"]
            ),
            exp_name=fold_dir,
            data_dir=os.path.join(args["model_args"]["data_dir"], fold_dir),
            model_args=args["model_args"],
            model_params=args["model_params"],
        )
        mt.load_dataset(use_augment=args["use_augment"])
        args["model_args"]["exp_dir"] = mt.exp_dir

        if args["model"] == "CLAM_SB":
            args["model-CLAM_SB"]["model_args"]["exp_dir"] = mt.exp_dir
            mt.model = CLAM_SB(args["model-CLAM_SB"])
        elif args["model"] == "EfficientNetB0":
            args["model-EfficientNetB0"]["model_args"]["exp_dir"] = mt.exp_dir
            mt.model, mt.callbacks_list, mt.epochs_done = get_EfficientNetB0(
                args["model-EfficientNetB0"]
            )
        elif args["model"] == "ResNet50":
            args["model-ResNet50"]["model_args"]["exp_dir"] = mt.exp_dir
            mt.model, mt.callbacks_list, mt.epochs_done = get_ResNet50(
                args["model-ResNet50"]
            )
        else:
            raise ValueError("Invalid model")

        mt.info()
        mt.train()
        mt.evaluate()
