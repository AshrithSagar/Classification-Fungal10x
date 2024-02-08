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
                    label = 1 if "F" in image_name else 0  # Infer label
                    labels.append(label)
                    names.append(image_name)

            self.slide_dataset = np.array(dataset)
            self.slide_labels = np.array(labels)
            self.slide_names = names

            return self.slide_dataset, self.slide_labels, self.slide_names

        self.slide_dataset, self.slide_labels, self.slide_names = load(self.slide_dir)
        self.annot_dataset, _, _ = load(self.annot_dir)

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

        for self.fold, (train_idx, val_idx) in enumerate(
            kfold.split(self.x_slides, self.y_slides)
        ):
            if run_only and self.fold not in run_only:
                continue

            self.x_train_slides, self.x_val_slides = (
                self.x_slides[train_idx],
                self.x_slides[val_idx],
            )
            self.y_train_slides, self.y_val_slides = (
                self.y_slides[train_idx],
                self.y_slides[val_idx],
            )

            self.x_train_annot, self.x_val_annot = (
                self.x_annot[train_idx],
                self.x_annot[val_idx],
            )
            self.y_train_annot, self.y_val_annot = (
                self.y_annot[train_idx],
                self.y_annot[val_idx],
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

    def segregate_slides(self):
        def segregate(slides, labels):
            f_idx = np.where(labels == 1)[0]
            f_imgs = tf.gather(slides, f_idx)

            nf_idx = np.where(labels == 0)[0]
            nf_imgs = tf.gather(slides, nf_idx)

            return f_imgs, nf_imgs

        self.x_train_slides_fungal, self.x_train_slides_nonfungal = segregate(
            self.x_train_slides, self.y_train_slides
        )
        self.x_val_slides_fungal, self.x_val_slides_nonfungal = segregate(
            self.x_val_slides, self.y_val_slides
        )
        self.x_test_slides_fungal, self.x_test_slides_nonfungal = segregate(
            self.x_test_slides, self.y_test_slides
        )

        self.x_train_annot_fungal, self.x_train_annot_nonfungal = segregate(
            self.x_train_annot, self.y_train_annot
        )
        self.x_val_annot_fungal, self.x_val_annot_nonfungal = segregate(
            self.x_val_annot, self.y_val_annot
        )
        self.x_test_annot_fungal, self.x_test_annot_nonfungal = segregate(
            self.x_test_annot, self.y_test_annot
        )

    def save_slides(self, save_ext="png"):
        def save(data_dir, sub_dir, label, dataset):
            sub_dir = os.path.join(data_dir, sub_dir)
            os.makedirs(sub_dir, exist_ok=True)
            os.makedirs(os.path.join(sub_dir, label), exist_ok=True)

            for idx, img in tqdm(enumerate(dataset)):
                img_file = os.path.join(sub_dir, label, f"{idx:04}.{save_ext}")
                pil_img = Image.fromarray((img.numpy() * 1).astype(np.uint8))
                pil_img.save(img_file)

    def downsample_slides(self, size=None, factor=None, preserve_aspect_ratio=True):
        def downsample(slides):
            return tf.image.resize(
                slides, downsample_size, preserve_aspect_ratio=preserve_aspect_ratio
            )

        downsample_size = None
        if factor:
            downsample_size = tuple([int(x / factor) for x in self.slide_dims])
        if size:
            downsample_size = size
        if not downsample_size:
            raise ValueError("Either size or downsample factor must be provided.")

        self.x_train_slides = downsample(self.x_train_slides)
        self.x_val_slides = downsample(self.x_val_slides)
        self.x_test_slides = downsample(self.x_test_slides)
        self.x_annot = downsample(self.x_annot)
        self.x_val_annot = downsample(self.x_val_annot)
        self.x_test_annot = downsample(self.x_test_annot)
        print(f"Downsampled to size: {downsample_size}")

    def extract_patches(self, size=(224, 224), overlap=0.5):
        def get_patches(dataset):
            all_patches = tf.image.extract_patches(
                images=dataset,
                sizes=(1, *self.patch_dims, 1),
                strides=(1, *stride, 1),
                rates=(1, 1, 1, 1),
                padding="VALID",
            )
            num_patches = all_patches.shape[1] * all_patches.shape[2]
            depth = dataset.shape[3]
            all_patches = tf.reshape(
                all_patches, (-1, num_patches, *self.patch_dims, depth)
            )
            return all_patches

        self.patch_dims = size
        stride = tuple(int(s * (1 - overlap)) for s in self.patch_dims)
        self.x_train_patches = get_patches(self.x_train_slides)
        self.x_val_patches = get_patches(self.x_val_slides)
        self.x_test_patches = get_patches(self.x_test_slides)

    def get_annotations(self, threshold=200):
        def get_annot(patches):
            patches_gray = tf.image.rgb_to_grayscale(patches)
            patches_binary = tf.where(
                patches_gray > 250,
                x=tf.ones((*self.patch_dims, 1)),
                y=tf.zeros((*self.patch_dims, 1)),
            )
            patches_nonzero = tf.math.count_nonzero(patches_binary, [2, 3])
            patches_nonzero = tf.squeeze(patches_nonzero, axis=[-1])
            patch_labels = tf.where(
                patches_nonzero > threshold,
                x=tf.ones(patches_nonzero.shape),
                y=tf.fill(patches_nonzero.shape, -1.0),
            )

            print(
                f"Non-zero patches: {tf.math.count_nonzero(patches_nonzero > threshold).numpy()}"
            )
            return patch_labels, patches_binary

        self.x_train_patch_labels, self.x_train_patches_annot = get_annot(
            self.x_train_annot,
        )
        self.x_val_patch_labels, self.x_val_patches_annot = get_annot(
            self.x_val_annot,
        )
        self.x_test_patch_labels, self.x_test_patches_annot = get_annot(
            self.x_test_annot,
        )

        self.annot_patches_labels, _ = get_annot(self.annot_dataset)

    def verify_annotations(self):
        nonfungal_indices = [
            index for index, label in enumerate(self.slide_labels) if label == 0
        ]

        annot_slides_check = tf.reduce_any(self.annot_patches_labels == 1, 1)
        wrong_labels_slide_annot = len(
            set(np.where(annot_slides_check == True)[0]).intersection(nonfungal_indices)
        )
        wrong_labels_slide_annot_indices = set(
            np.where(annot_slides_check == True)[0]
        ).intersection(nonfungal_indices)

        print(f"Annot True: {np.count_nonzero(annot_slides_check == True)};")
        print(f"Should be {np.count_nonzero(self.slide_labels == True)})")

        print(f"Annot False: {np.count_nonzero(annot_slides_check == False)};")
        print(f"Should be {np.count_nonzero(self.slide_labels == False)})")

        print(f"Wrongly labelled slide annotations: {wrong_labels_slide_annot}")

        print("Wrongly labelled slide annotations indices:")
        print(wrong_labels_slide_annot_indices, end=", ")

    def filter_patch_annotations(self):
        def filter_patch_annot(y_slides, x_patch_labels):
            a = tf.cast(y_slides, dtype="float32")
            b = tf.expand_dims(a, 1)
            c = tf.expand_dims(tf.ones(x_patch_labels.shape[-1]), 0)
            d = tf.linalg.matmul(b, c)
            e = tf.cast(d, dtype="bool")

            f = x_patch_labels == -1
            g = tf.math.logical_xor(e, f)
            return g, d

        self.x_train_patches_filtermask, self.y_train_patches = filter_patch_annot(
            self.y_train_slides, self.x_train_patch_labels
        )
        self.x_val_patches_filtermask, self.y_val_patches = filter_patch_annot(
            self.y_val_slides, self.x_val_patch_labels
        )
        self.x_test_patches_filtermask, self.y_test_patches = filter_patch_annot(
            self.y_test_slides, self.x_test_patch_labels
        )

        self.x_train = tf.boolean_mask(
            self.x_train_patches,
            self.x_train_patches_filtermask,
        )
        self.y_train = tf.boolean_mask(
            self.y_train_patches,
            self.x_train_patches_filtermask,
        )

        self.x_val = tf.boolean_mask(
            self.x_val_patches,
            self.x_val_patches_filtermask,
        )
        self.y_val = tf.boolean_mask(
            self.y_val_patches,
            self.x_val_patches_filtermask,
        )

        self.x_test = tf.boolean_mask(
            self.x_test_patches,
            self.x_test_patches_filtermask,
        )
        self.y_test = tf.boolean_mask(
            self.y_test_patches,
            self.x_test_patches_filtermask,
        )

    def patches_info(self):
        def patches_verbose(check_bool, patches_filtermask, patch_labels):
            mask = patches_filtermask == check_bool
            total = tf.math.count_nonzero(mask).numpy()
            fungal = tf.math.count_nonzero(
                tf.math.logical_and(mask, (patch_labels == 1))
            ).numpy()
            nonfungal = tf.math.count_nonzero(
                tf.math.logical_and(mask, (patch_labels == -1))
            ).numpy()
            state = "considered" if check_bool else "discarded"
            return f"patches {state}: {total}; (F:{fungal}, NF:{nonfungal})"

        dataset_info = [
            ("Train", self.x_train_patches_filtermask, self.x_train_patch_labels),
            ("Validation", self.x_val_patches_filtermask, self.x_val_patch_labels),
            ("Test", self.x_test_patches_filtermask, self.x_test_patch_labels),
        ]
        for name, filtermask, patch_labels in dataset_info:
            print(name, patches_verbose(True, filtermask, patch_labels))
            print(name, patches_verbose(False, filtermask, patch_labels))

    def patches_split_info(self):
        patches_split_verbose = (
            lambda x: f"{x.shape[0]} (F:{int(np.sum(x == 1))}, NF:{int(np.sum(x == 0))})"
        )
        print("Training patches:", patches_split_verbose(self.y_train))
        print("Validation patches:", patches_split_verbose(self.y_val))
        print("Test patches:", patches_split_verbose(self.y_test))

    def save_patches(self, data_dir, sub_dir, label, patches, save_ext="png"):
        sub_dir = os.path.join(data_dir, sub_dir)
        os.makedirs(sub_dir, exist_ok=True)
        os.makedirs(os.path.join(sub_dir, label), exist_ok=True)

        for idx, patch in tqdm(enumerate(patches)):
            patch = patch[0].numpy()
            patch_file = os.path.join(sub_dir, label, f"{idx:04}.{save_ext}")
            pil_img = Image.fromarray((patch * 1).astype(np.uint8))
            pil_img.save(patch_file)
