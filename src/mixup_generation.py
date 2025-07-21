import os
import cv2
import random
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils import paths
from PIL import Image
from configs.mixup import DatMixConfig
from utils import legacy

covid_dir = os.path.join(paths.REAL_CLASSIFIER_DIR, "covid")
pneumonia_dir = os.path.join(paths.REAL_CLASSIFIER_DIR, "pneumonia")
normal_dir = os.path.join(paths.REAL_CLASSIFIER_DIR, "normal")
covid_images = sorted([os.path.join(covid_dir, f) for f in os.listdir(covid_dir)])
pneumonia_images = sorted(
    [os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir)]
)
normal_images = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir)])
all_classes = {
    "covid": covid_images,
    "pneumonia": pneumonia_images,
    "normal": normal_images,
}

class_to_onehot = {
    "covid": np.array([1, 0, 0]),
    "pneumonia": np.array([0, 1, 0]),
    "normal": np.array([0, 0, 1]),
}


def load_and_preprocess(img_path: str, img_size: tuple) -> np.ndarray:
    """
    Load a grayscale image, resize and cast to float32.

    Args:
        img_path (str): Path to the image file.
        img_size (tuple): Desired (width, height).

    Returns:
        np.ndarray: Preprocessed image array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    return img.astype(np.float32)


def generate_mixup(cfg: DatMixConfig) -> None:
    """
    Generate simple MixUp images by blending 2 real images.

    Saves images under and writes labels CSV.
    """
    # Creates output directory if it doesn't exist
    os.makedirs(cfg.mixup_output_dir, exist_ok=True)
    n_samples = cfg.nbr_images_to_generate
    csv_path = os.path.join(cfg.mixup_output_dir, cfg.csv_file_name)
    labels = []
    for idx in tqdm(range(n_samples)):
        same_class = random.random() < cfg.same_class_ratio

        if same_class:
            cls = random.choice(list(all_classes.keys()))
            img_list = all_classes[cls]
            img1_path, img2_path = random.sample(img_list, 2)
            label1 = label2 = class_to_onehot[cls]
        else:
            cls1, cls2 = random.sample(list(all_classes.keys()), 2)
            img1_path = random.choice(all_classes[cls1])
            img2_path = random.choice(all_classes[cls2])
            label1 = class_to_onehot[cls1]
            label2 = class_to_onehot[cls2]

        img1 = load_and_preprocess(img1_path, img_size=cfg.img_size)
        img2 = load_and_preprocess(img2_path, img_size=cfg.img_size)

        # MixUp
        alpha = np.random.beta(0.4, 0.4)
        img_mix = alpha * img1 + (1 - alpha) * img2
        label_mix = alpha * label1 + (1 - alpha) * label2

        # save mixed image
        img_save = (img_mix * 255).astype(np.uint8)
        filename = f"mixup_{idx:05d}.png"
        image_path = os.path.join(cfg.mixup_output_dir, filename)
        cv2.imwrite(image_path, img_save)

        # save labels
        labels.append([filename, *label_mix.tolist()])

    # label CSV
    labels_df = pd.DataFrame(
        labels, columns=["filename", "covid", "pneumonia", "normal"]
    )
    labels_df.to_csv(os.path.join(cfg.mixup_output_dir, "labels.csv"), index=False)

    print(f"Finished generating {n_samples} MixUp images.")


def generate_mmixup(cfg: DatMixConfig) -> None:
    """
    Generate MixUp images with a more complex approach, using multiple images.
    """
    # Creates output directory if it doesn't exist
    os.makedirs(cfg.mmix_output_dir, exist_ok=True)
    n_samples = cfg.nbr_images_to_generate
    csv_path = os.path.join(cfg.mmix_output_dir, cfg.csv_file_name)
    labels = []

    for idx in tqdm(range(n_samples)):

        cls = random.choice(list(all_classes.keys()))
        label = class_to_onehot[cls]
        soft_label = np.random.dirichlet(
            [cfg.maj if x == 1 else cfg.min for x in label]
        )

        img1_path = random.choice(covid_images)  # 1 for covid
        img2_path = random.choice(pneumonia_images)  # 2 for pneumonia
        img3_path = random.choice(normal_images)  # 3 for normal

        img1 = load_and_preprocess(img1_path, img_size=cfg.img_size)
        img2 = load_and_preprocess(img2_path, img_size=cfg.img_size)
        img3 = load_and_preprocess(img3_path, img_size=cfg.img_size)

        img_save = soft_label[0] * img1 + soft_label[1] * img2 + soft_label[2] * img3
        filename = f"mmixup_{idx:05d}.png"
        image_path = os.path.join(cfg.mmix_output_dir, filename)
        cv2.imwrite(image_path, img_save)

        labels.append([filename, *soft_label.tolist()])

    # label CSV
    labels_df = pd.DataFrame(
        labels, columns=["filename", "covid", "pneumonia", "normal"]
    )
    labels_df.to_csv(csv_path, index=False)

    print(f"Finished generating {n_samples} MMixUp images.")


def generate_gemix(cfg: DatMixConfig) -> None:
    """
    Generate Gemix images with a more complex approach, using multiple images.
    """
    # Creates output directory if it doesn't exist
    os.makedirs(cfg.gemix_output_dir, exist_ok=True)
    n_samples = cfg.nbr_images_to_generate
    csv_path = os.path.join(cfg.gemix_output_dir, cfg.csv_file_name)
    labels = []
    # load generator
    with open(paths.NETWORK_PKL, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(cfg.device)

    def generate_image(G, z, c):
        img = G(z, c, noise_mode="const")
        img = (img.clamp(-1, 1) + 1) * 127.5
        img = img[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        if img.shape[2] == 1:
            img = img[:, :, 0]

        return Image.fromarray(img)

    for idx in tqdm(range(n_samples)):
        z = torch.randn([1, G.z_dim], device=cfg.device)
        cls = random.choice(list(all_classes.keys()))
        label = class_to_onehot[cls]
        soft_label = np.random.dirichlet(
            [cfg.maj if x == 1 else cfg.min for x in label]
        )

        c_tensor = (
            torch.tensor(soft_label, device=cfg.device).unsqueeze(0).to(torch.float32)
        )
        img_save = generate_image(G, z, c_tensor)

        filename = f"gan_mixup_{idx:05d}.png"
        img_save.save(os.path.join(cfg.gemix_output_dir, filename))
        labels.append([filename] + soft_label.tolist())

    # label CSV
    labels_df = pd.DataFrame(
        labels, columns=["filename", "covid", "pneumonia", "normal"]
    )
    labels_df.to_csv(csv_path, index=False)

    print(f"Finished generating {n_samples} MMixUp images.")
