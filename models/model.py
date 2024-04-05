"""
model.py
"""

from models.clamSB_tf import CLAM_SB
from models.EfficientNetB0 import get_EfficientNetB0
from models.ResNet50 import get_ResNet50


def get_model(mt, t_args):

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

    return mt
