"""
exp_summary.py
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import load_config
from utils.model import ModelSummary


if __name__ == "__main__":
    h_args = load_config(config_file="config.yaml", key="heatmaps")
    t_args = load_config(config_file="config.yaml", key="trainer")

    ms = ModelSummary(
        exp_base_dir=t_args["exp_base_dir"],
        exp_name=t_args["model-EfficientNetB0"]["model_args"]["exp_name"],
    )
    ms.get_results(heatmaps=h_args["save_dir"], folds=t_args["folds"])
