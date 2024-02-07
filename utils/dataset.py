"""
dataset.py
"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold


class FungalDataLoader:
    def __init__(self, data_dir_name, slide_dir, annot_dir, seed=42):
        self.seed = seed
        self.data_dir_name = data_dir_name
        self.slide_dir = slide_dir
        self.annot_dir = annot_dir
        self.slide_dims = (1200, 1600)
        tf.random.set_seed(seed)

    def load_slides(self):
        def load(images_dir):
            dataset, labels, names = [], [], []

            for image_name in sorted(os.listdir(images_dir)):
                if image_name.endswith(".tif"):
                    image = Image.open(os.path.join(images_dir, image_name))
                    dataset.append(np.array(image))
                    label = 1 if "F" in image_name else 0.0  # Infer label
                    labels.append(label)
                    names.append(image_name)

            self.slide_dataset = np.array(dataset)
            self.slide_labels = np.array(labels)
            self.slide_names = names

            return self.slide_dataset, self.slide_labels, self.slide_names

        self.slide_dataset, self.slide_labels, self.slide_names = load(self.slide_dir)
        self.annot_dataset, _, _ = load(self.annot_dir)

    def downsample(self, size=None, factor=None, preserve_aspect_ratio=None):
        if factor:
            downsample_size = tuple([int(x / factor) for x in self.slide_dims])
        else:
            downsample_size = size

        print(f"Downsample size: {downsample_size}")
        downsampled_slides = tf.image.resize(
            self.slides,
            downsample_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
        )

        return downsampled_slides

    def create_splits(self, test_split=0.2):
        def split(dataset, labels):
            return train_test_split(
                dataset,
                labels,
                test_size=test_split,
                random_state=self.seed,
                stratify=labels,
            )

        self.x_slides, self.x_test_slides, self.y_slides, self.y_test_slides = split(
            self.slide_dataset, self.slide_labels
        )
        self.x_names, self.x_test_slide_names, self.y_names, _ = split(
            self.slide_names, self.slide_labels
        )
        self.x_annot, self.x_test_annot, self.y_annot, self.y_test_annot = split(
            self.annot_dataset, self.slide_labels
        )

    def create_kfold_splits(self, n_splits=5, run_only=None):
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        for fold, (train_idx, val_idx) in enumerate(
            kfold.split(self.slide_dataset, self.slide_labels)
        ):
            if run_only and fold not in run_only:
                continue

            self.x_train_slides, self.x_val_slides = (
                self.slide_dataset[train_idx],
                self.slide_dataset[val_idx],
            )
            self.y_train_slides, self.y_val_slides = (
                self.slide_labels[train_idx],
                self.slide_labels[val_idx],
            )

            self.split_info()
            yield self.x_train_slides, self.y_train_slides, self.x_val_slides, self.y_val_slides

    def split_info(self):
        slides_split_verbose = (
            lambda x: f"{x.shape[0]} (F:{int(np.sum(x))}, NF:{int(len(x) - np.sum(x))})"
        )
        print("Training slides:", slides_split_verbose(self.y_train_slides))
        print("Validation slides:", slides_split_verbose(self.y_val_slides))
        print("Test slides:", slides_split_verbose(self.y_test_slides))

    def save_slides(self, data_dir, sub_dir, label, dataset, save_ext="png"):
        sub_dir = os.path.join(data_dir, sub_dir)
        os.makedirs(sub_dir, exist_ok=True)
        os.makedirs(os.path.join(sub_dir, label), exist_ok=True)

        for idx, img in tqdm(enumerate(dataset)):
            img_file = os.path.join(sub_dir, label, f"{idx:04}.{save_ext}")
            pil_img = Image.fromarray((img.numpy() * 1).astype(np.uint8))
            pil_img.save(img_file)
