"""
exp_summary.py
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import load_config
from utils.trainer import ModelSummary


if __name__ == "__main__":
    args = load_config(config_file="config.yaml", key="trainer")

    ms = ModelSummary(
        exp_base_dir=args["exp_base_dir"],
        exp_name=args["model_args"]["exp_name"],
    )
    results = ms.get_cv_results(folds=args["folds"])
