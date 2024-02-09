"""
config.py
"""

import os
import yaml


def load_config(config_file="config.yaml", key=None):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if key:
        return config.get(key, {})
    else:
        return config
