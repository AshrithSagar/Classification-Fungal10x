"""
dataset.py
"""

import os
import shutil
import numpy as np
import pandas as pd
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
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

            return np.array(dataset), np.array(labels), names

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

            self.x_train_slide_names, self.x_val_slide_names = (
                self.x_names[train_idx],
                self.x_names[val_idx],
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
        def segregate(slides, labels, slide_names=None):
            f_idx = np.where(labels == 1)[0]
            nf_idx = np.where(labels == 0)[0]

            f_imgs = tf.gather(slides, f_idx)
            nf_imgs = tf.gather(slides, nf_idx)

            if slide_names:
                f_slide_names = slide_names[f_idx]
                nf_slide_names = slide_names[nf_idx]
                return f_imgs, nf_imgs, f_slide_names, nf_slide_names
            else:
                return f_imgs, nf_imgs

        datasets = [
            (
                self.x_train_slides,
                self.y_train_slides,
                self.x_train_slide_names,
                "train",
            ),
            (self.x_val_slides, self.y_val_slides, self.x_val_slide_names, "val"),
            (self.x_test_slides, self.y_test_slides, self.x_test_slide_names, "test"),
            (self.x_train_annot, self.y_train_annot, None, "train"),
            (self.x_val_annot, self.y_val_annot, None, "val"),
            (self.x_test_annot, self.y_test_annot, None, "test"),
        ]

        for slides, labels, slide_names, dataset_type in datasets:
            f_imgs, nf_imgs, f_slide_names, nf_slide_names = segregate(
                slides, labels, slide_names
            )
            setattr(self, f"x_{dataset_type}_slides_fungal", f_imgs)
            setattr(self, f"x_{dataset_type}_slides_nonfungal", nf_imgs)
            if slide_names:
                setattr(self, f"x_{dataset_type}_slide_names_fungal", f_slide_names)
                setattr(self, f"x_{dataset_type}_slide_names_nonfungal", nf_slide_names)

    def save_slides(self, save_ext="png"):
        def save(data_dir, sub_dir, label, dataset, slide_names):
            sub_dir = os.path.join(data_dir, sub_dir)
            os.makedirs(sub_dir, exist_ok=True)
            os.makedirs(os.path.join(sub_dir, label), exist_ok=True)

            for img, name in tqdm(zip(dataset, slide_names)):
                img_file = os.path.join(
                    sub_dir, label, f"{os.path.splitext(name)[0]}.{save_ext}"
                )
                pil_img = Image.fromarray((img.numpy() * 1).astype(np.uint8))
                pil_img.save(img_file)

        data_dir = f"dataset/{self.data_dir_name}-slides"
        os.makedirs(data_dir, exist_ok=True)
        print(data_dir)

        save_dir = os.path.join(data_dir, f"fold_{self.fold}")
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir)

        save(
            save_dir,
            "train",
            "fungal",
            self.x_train_slides_fungal,
            self.x_train_slide_names_fungal,
        )
        save(
            save_dir,
            "train",
            "nonfungal",
            self.x_train_slides_nonfungal,
            self.x_train_slide_names_nonfungal,
        )
        save(
            save_dir,
            "val",
            "fungal",
            self.x_val_slides_fungal,
            self.x_val_slide_names_fungal,
        )
        save(
            save_dir,
            "val",
            "nonfungal",
            self.x_val_slides_nonfungal,
            self.x_val_slide_names_nonfungal,
        )
        save(
            save_dir,
            "test",
            "fungal",
            self.x_test_slides_fungal,
            self.x_test_slide_names_fungal,
        )
        save(
            save_dir,
            "test",
            "nonfungal",
            self.x_test_slides_nonfungal,
            self.x_test_slide_names_nonfungal,
        )

    def downsample_slides(self, size=None, factor=None, preserve_aspect_ratio=True):
        def downsample(slides):
            return tf.image.resize(
                slides,
                self.downsample_size,
                preserve_aspect_ratio=preserve_aspect_ratio,
            )

        self.downsample_size = None
        if factor:
            self.downsample_size = tuple([int(x / factor) for x in self.slide_dims])
        if size:
            self.downsample_size = size
        if not self.downsample_size:
            raise ValueError("Either size or downsample factor must be provided.")

        self.x_train_slides = downsample(self.x_train_slides)
        self.x_val_slides = downsample(self.x_val_slides)
        self.x_test_slides = downsample(self.x_test_slides)
        self.x_train_annot = downsample(self.x_train_annot)
        self.x_val_annot = downsample(self.x_val_annot)
        self.x_test_annot = downsample(self.x_test_annot)
        print(f"Downsampled to size: {self.downsample_size}")

    def extract_patches(self, size=(224, 224), overlap=0.5):
        def get_patches(dataset):
            all_patches = tf.image.extract_patches(
                images=dataset,
                sizes=(1, *self.patch_dims, 1),
                strides=(1, *stride, 1),
                rates=(1, 1, 1, 1),
                padding="VALID",
            )
            patches_shape = (all_patches.shape[1], all_patches.shape[2])
            num_patches = np.prod(patches_shape)
            depth = dataset.shape[3]
            all_patches = tf.reshape(
                all_patches, (-1, num_patches, *self.patch_dims, depth)
            )
            return all_patches, patches_shape

        self.patch_dims = size
        stride = tuple(int(s * (1 - overlap)) for s in self.patch_dims)
        self.x_train_patches, _ = get_patches(self.x_train_slides)
        self.x_val_patches, _ = get_patches(self.x_val_slides)
        self.x_test_patches, self.patches_shape = get_patches(self.x_test_slides)
        self.x_train_annot_patches, _ = get_patches(self.x_train_annot)
        self.x_val_annot_patches, _ = get_patches(self.x_val_annot)
        self.x_test_annot_patches, _ = get_patches(self.x_test_annot)
        self.annot_dataset_patches, _ = get_patches(self.annot_dataset)

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
            self.x_train_annot_patches,
        )
        self.x_val_patch_labels, self.x_val_patches_annot = get_annot(
            self.x_val_annot_patches,
        )
        self.x_test_patch_labels, self.x_test_patches_annot = get_annot(
            self.x_test_annot_patches,
        )

        self.annot_patches_labels, _ = get_annot(self.annot_dataset_patches)

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

        info = [
            ("Train", self.x_train_patches_filtermask, self.x_train_patch_labels),
            ("Validation", self.x_val_patches_filtermask, self.x_val_patch_labels),
            ("Test", self.x_test_patches_filtermask, self.x_test_patch_labels),
        ]
        for name, filtermask, patch_labels in info:
            print(name, patches_verbose(True, filtermask, patch_labels))
            print(name, patches_verbose(False, filtermask, patch_labels))

    def patches_split_info(self):
        patches_split_verbose = (
            lambda x: f"{x.shape[0]} (F:{int(np.sum(x == 1))}, NF:{int(np.sum(x == 0))})"
        )
        print("Training patches:", patches_split_verbose(self.y_train))
        print("Validation patches:", patches_split_verbose(self.y_val))
        print("Test patches:", patches_split_verbose(self.y_test))

    def segregate_patches(self):
        def segregate(dataset, labels):
            f_idx = np.where(labels == 1)[0]
            f_imgs = tf.gather(dataset, f_idx)

            nf_idx = np.where(labels == 0)[0]
            nf_imgs = tf.gather(dataset, nf_idx)

            return f_imgs, nf_imgs

        self.x_train_fungal, self.x_train_nonfungal = segregate(
            self.x_train, self.y_train
        )

    def perform_augmentation(self, use_augment=True):
        def augment_image(image):
            # List of all possible transformations. Original image is included by default
            transformations = [
                lambda x: tf.image.flip_left_right(x),
                lambda x: tf.image.flip_up_down(x),
                lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x)),
                lambda x: tf.image.rot90(x, k=1),  # 90 degree
                lambda x: tf.image.rot90(x, k=2),  # 180 degree
                lambda x: tf.image.rot90(x, k=3),  # 270 degree
            ]

            aug_images = [transform(image) for transform in transformations]
            aug_dataset = tf.convert_to_tensor(aug_images)
            return aug_dataset

        def augment_dataset(train_dataset, target_count=-1):
            augmented_dataset = tf.map_fn(augment_image, train_dataset)
            augmented_dataset = tf.reshape(augmented_dataset, shape=[-1, 224, 224, 3])

            # Take all original images for both datasets
            original_images = train_dataset

            # Make target_count the sum of counts of train_dataset and augmented_dataset
            if target_count == -1:
                target_count = (
                    tf.shape(train_dataset)[0] + tf.shape(augmented_dataset)[0]
                )

            # Take all the original images and fill up the remaining count with augmented samples
            current_count = tf.shape(original_images)[0]
            remaining_count = tf.maximum(
                0, target_count - current_count
            )  # Ensure non-negative value

            # Include original samples and augmented samples to fill up the remaining count
            train_aug = tf.concat(
                [original_images, augmented_dataset[:remaining_count]], axis=0
            )

            train_aug = tf.random.shuffle(train_aug, seed=self.seed)
            return train_aug

        # Take all original images for x_train_fungal and fill up the augmentations for x_train_nonfungal
        if not use_augment:
            self.x_train_fungal_augmented = self.x_train_fungal
            self.x_train_nonfungal_augmented = self.x_train_nonfungal
            return

        self.x_train_fungal_augmented = augment_dataset(
            self.x_train_fungal,
            target_count=-1,
        )
        self.x_train_nonfungal_augmented = augment_dataset(
            self.x_train_nonfungal,
            target_count=len(self.x_train_fungal_augmented),
        )

    def augment_info(self):
        augment_verbose = lambda x, y: f"{x.shape[0]} => {y.shape[0]}"
        print("Train fungal patches:")
        print(augment_verbose(self.x_train_fungal, self.x_train_fungal_augmented))
        print("Train nonfungal patches")
        print(augment_verbose(self.x_train_nonfungal, self.x_train_nonfungal_augmented))

    def save_patches(self):
        def save(data_dir, sub_dir, dataset, labels, save_ext="png"):
            sub_dir = os.path.join(data_dir, sub_dir)
            os.makedirs(sub_dir, exist_ok=True)

            os.makedirs(os.path.join(sub_dir, "fungal"), exist_ok=True)
            os.makedirs(os.path.join(sub_dir, "nonfungal"), exist_ok=True)

            f_idx, nf_idx = 0, 0
            for img, label in tqdm(zip(dataset, labels)):
                label = label.numpy()
                class_dir = "fungal" if label else "nonfungal"
                if label:
                    img_idx = f_idx
                    f_idx += 1
                else:
                    img_idx = nf_idx
                    nf_idx += 1
                img_file = os.path.join(sub_dir, class_dir, f"{img_idx:04}.{save_ext}")
                pil_img = Image.fromarray((img.numpy() * 1).astype(np.uint8))
                pil_img.save(img_file)

        def save2(data_dir, sub_dir, label, dataset, save_ext="png"):
            sub_dir = os.path.join(data_dir, sub_dir)
            os.makedirs(sub_dir, exist_ok=True)
            os.makedirs(os.path.join(sub_dir, label), exist_ok=True)

            for idx, img in tqdm(enumerate(dataset)):
                img_file = os.path.join(sub_dir, label, f"{idx:04}.{save_ext}")
                pil_img = Image.fromarray((img.numpy() * 1).astype(np.uint8))
                pil_img.save(img_file)

        self.data_dir = f"dataset/{self.data_dir_name}-{self.downsample_size[0]}_{self.downsample_size[1]}"
        os.makedirs(self.data_dir, exist_ok=True)
        print(self.data_dir)

        save_dir = os.path.join(self.data_dir, f"fold_{self.fold}")
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir)

        save2(save_dir, "train", "fungal", self.x_train_fungal_augmented)
        save2(save_dir, "train", "nonfungal", self.x_train_nonfungal_augmented)
        save(save_dir, "train_unaugmented", self.x_train, self.y_train)
        save(save_dir, "val", self.x_val, self.y_val)
        save(save_dir, "test", self.x_test, self.y_test)

    def zip_data_dir(self, data_dir=None, sub_dir=None):
        def zip(data_dir):
            zip_name = os.path.basename(os.path.normpath(data_dir))
            print(zip_name)

            if not sub_dir:
                # ZIP entire data_set folder
                shutil.make_archive(zip_name, "zip", data_dir)
            else:
                # ZIP specific subdirectory
                sub_dir_path = os.path.join(data_dir, sub_dir)
                shutil.make_archive(zip_name, "zip", sub_dir_path)

        zip(data_dir or self.data_dir)

    def dataset_info(self, data_dir=None):
        def dir_info(data_dir):
            dir_info = {}
            a_dir_path = os.path.join(
                data_dir,
            )
            for a_dir in os.listdir(a_dir_path):
                b_dir_path = os.path.join(a_dir_path, a_dir)
                b_dict = {}
                for b_dir in os.listdir(b_dir_path):
                    c_dir = os.path.join(b_dir_path, b_dir)
                    b_dict.update({b_dir: len(os.listdir(c_dir))})  # TODO
                dir_info.update({a_dir: b_dict})
            return dir_info

        if not data_dir:
            data_dir = self.data_dir

        dir_info = dir_info(data_dir)
        df_dir_info = pd.DataFrame.from_dict(dir_info, orient="index")
        df_dir_info["total"] = df_dir_info["nonfungal"] + df_dir_info["fungal"]
        print(df_dir_info)
