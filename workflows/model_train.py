"""
model_train.py
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import load_config
from utils.trainer import ModelTrainer
from models.EfficientNetB0 import get_EfficientNetB0


if __name__ == "__main__":
    args = load_config(config_file="config.yaml", key="trainer")

    for fold in args["folds"]:
        fold_dir = f"fold-{fold}"

        mt = ModelTrainer(
            exp_base_dir=os.path.join(
                args["exp_base_dir"], args["model_args"]["exp_name"]
            ),
            exp_name=fold_dir,
            data_dir=args["model_args"]["data_dir"],
            model_args=args["model_args"],
            model_params=args["model_params"],
        )
        mt.check_gpu()
        mt.set_gpu(device_index=0)
        mt.load_dataset(use_augment=False)
        mt.model, mt.callbacks_list, mt.epochs_done = get_EfficientNetB0(args)
        mt.info()
        mt.train()
        mt.evaluate()
