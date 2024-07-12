"""
exp_summary.py
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import Config
from utils.model import ModelSummary


def main(args):
    h_args, t_args = args["heatmaps"], args["trainer"]

    ms = ModelSummary(
        exp_base_dir=t_args["exp_base_dir"],
        exp_name=t_args["exp_name"],
    )
    ms.get_results(heatmaps=h_args["save_dir"], folds=t_args["folds"])


if __name__ == "__main__":
    args = Config.from_args("Find the CV metrics and zip the model results.")
    main(args)
