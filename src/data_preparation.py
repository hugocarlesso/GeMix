import os
import shutil
import random
from typing import Dict, List
import utils.paths as paths

CLASS_MAP = {"0": "normal", "1": "pneumonia", "2": "covid"}


def read_label_file(label_file: str) -> Dict[str, List[str]]:
    """
    Parse a label file and collect image paths per class.

    Args:
        label_file (str): Path to the annotation .txt (filename + class_label).

    Returns:
        Dict[str, List[str]]: Mapping from class_name to list of absolute image paths.
    """
    image_dict = {cls: [] for cls in CLASS_MAP.values()}
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fname, cls_lbl = parts
            cls_name = CLASS_MAP.get(cls_lbl)
            if not cls_name:
                continue
            img_path = os.path.join(paths.IMAGE_DIR, fname)
            if os.path.exists(img_path):
                image_dict[cls_name].append(img_path)
            else:
                print(f"[WARN] Missing image: {img_path}")
    return image_dict


def make_subset(label_file: str, output_dir: str, subset_size: int) -> None:
    """
    Create a balanced subset by randomly sampling up to N images per class.

    Args:
        label_file (str): Path to the annotation file.
        output_dir (str): Destination folder for the new subset (created if needed).
        subset_size (int): Maximum number of images per class.
    """
    os.makedirs(output_dir, exist_ok=True)
    for cls in CLASS_MAP.values():
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    image_dict = read_label_file(label_file)
    for cls_name, img_paths in image_dict.items():
        n = min(len(img_paths), subset_size)
        sampled = random.sample(img_paths, n)
        for src in sampled:
            dst = os.path.join(output_dir, cls_name, os.path.basename(src))
            shutil.copy(src, dst)
        print(f"[INFO] {n} images copied to {cls_name}/ in {output_dir}")


def make_real_dataset_excluding_gan(
    train_label_file: str, gan_dir: str, output_dir: str, subset_size: int
) -> None:
    """
    Build a 'real' dataset by excluding images already used for GAN.

    Args:
        train_label_file (str): Path to the training annotation file.
        gan_dir (str): Directory of the GAN training subset.
        output_dir (str): Destination folder for the real classifier set.
        subset_size (int): Number of images per class to sample.

    Raises:
        ValueError: If there are not enough images after exclusion.
    """
    os.makedirs(output_dir, exist_ok=True)
    for cls in CLASS_MAP.values():
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    # Filenames already used for GAN
    used = set()
    for cls in CLASS_MAP.values():
        cls_folder = os.path.join(gan_dir, cls)
        if os.path.isdir(cls_folder):
            used.update(os.listdir(cls_folder))

    # Load and filer
    image_dict = {cls: [] for cls in CLASS_MAP.values()}
    with open(train_label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fname, cls_lbl = parts
            cls_name = CLASS_MAP.get(cls_lbl)
            if cls_name and fname not in used:
                img_path = os.path.join(paths.IMAGE_DIR, fname)
                if os.path.exists(img_path):
                    image_dict[cls_name].append(img_path)

    # Sample and copy
    for cls_name, img_paths in image_dict.items():
        available = len(img_paths)
        if available < subset_size:
            raise ValueError(
                f"Not enough images for class '{cls_name}': {available} found (need {subset_size})."
            )
        sampled = random.sample(img_paths, subset_size)
        for src in sampled:
            dst = os.path.join(output_dir, cls_name, os.path.basename(src))
            shutil.copy(src, dst)
        print(f"[INFO] {subset_size} images copied to {cls_name}/ in {output_dir}")
