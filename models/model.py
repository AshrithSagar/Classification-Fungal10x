"""
model.py
"""


def freeze_layers(model, freeze, verbose=True):
    """
    freeze: Ratio in which to freeze the first few layers
    """
    model.trainable = True

    freeze_upto = round(freeze * len(model.layers))
    train_rest = len(model.layers) - freeze_upto

    for layer in model.layers[:freeze_upto]:
        layer.trainable = False
    for layer in model.layers[train_rest:]:
        layer.trainable = True

    if verbose:
        print(f"{freeze_upto} layers freezed; {train_rest} layers trainable")

    return model


def get_model(mt, t_args):

    if t_args["model"] == "CLAM_SB":
        from models.clamSB_tf import CLAM_SB

        t_args["model-CLAM_SB"]["model_args"]["exp_dir"] = mt.exp_dir
        mt.model = CLAM_SB(t_args["model-CLAM_SB"])

    elif t_args["model"] == "EfficientNetB0":
        from models.EfficientNetB0 import get_EfficientNetB0

        t_args["model-EfficientNetB0"]["model_args"]["exp_dir"] = mt.exp_dir
        mt.model, mt.callbacks_list, mt.epochs_done = get_EfficientNetB0(
            t_args["model-EfficientNetB0"]
        )

    elif t_args["model"] == "ResNet50":
        from models.ResNet50 import get_ResNet50

        t_args["model-ResNet50"]["model_args"]["exp_dir"] = mt.exp_dir
        mt.model, mt.callbacks_list, mt.epochs_done = get_ResNet50(
            t_args["model-ResNet50"]
        )

    else:
        raise ValueError("Invalid model")

    return mt
