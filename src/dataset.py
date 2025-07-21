import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms, datasets
from PIL import Image
from utils import paths

class_to_idx = {"covid": 0, "normal": 1, "pneumonia": 2}
idx_to_onehot = {
    class_to_idx["covid"]: [1.0, 0.0, 0.0],
    class_to_idx["normal"]: [0.0, 1.0, 0.0],
    class_to_idx["pneumonia"]: [0.0, 0.0, 1.0],
}


class RealSoftDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        soft_label = torch.tensor(idx_to_onehot[label], dtype=torch.float32)
        return img, soft_label


class GeMixupDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(
            row[["covid", "normal", "pneumonia"]].values.astype("float32")
        )
        return image, label


class PixelMixupDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            row[["covid", "normal", "pneumonia"]].values.astype("float32")
        )
        return image, label


def get_dataset(config):
    # Define transformation pipeline
    transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    # Helper to load soft real dataset
    def load_real():
        raw = datasets.ImageFolder(root=paths.REAL_CLASSIFIER_DIR, transform=transform)
        return RealSoftDataset(raw)

    # Helper to split any dataset into train/val
    def split_dataset(dataset, val_ratio=config.val_ratio, seed=42):
        total = len(dataset)
        val_size = int(val_ratio * total)
        train_size = total - val_size
        return random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

    # Helper to load CSV-based mixup datasets
    def load_mixup(csv_dir, dataset_cls):
        df = pd.read_csv(os.path.join(csv_dir, "labels.csv"))
        df = df[["filename", "covid", "normal", "pneumonia"]]
        return dataset_cls(df, csv_dir, transform)

    # Initialize placeholders
    train_dataset = None
    val_dataset = None
    exp = config.expe_name

    # 1) Real-only
    if exp == "real_only":
        real_ds = load_real()
        train_dataset, val_dataset = split_dataset(real_ds)

    # 2) GeMix mixup only
    elif exp == "gemix_only":
        gan_ds = load_mixup(paths.GEMIX_OUTPUT_DIR, GeMixupDataset)
        train_dataset, val_dataset = split_dataset(gan_ds)

    # 3) Traditional mixup only
    elif exp == "mixup_only":
        trad_ds = load_mixup(paths.MIX_OUTPUT_DIR, PixelMixupDataset)
        train_dataset, val_dataset = split_dataset(trad_ds)

    # 3) Generalized mixup only
    elif exp == "mmixup_only":
        trad_ds = load_mixup(paths.MMIX_OUTPUT_DIR, PixelMixupDataset)
        train_dataset, val_dataset = split_dataset(trad_ds)

    # 4) Real + mixup variants
    else:
        # Split real data
        real_ds = load_real()
        train_real, val_dataset = split_dataset(real_ds)

        # Collect additional parts
        parts = []
        if "mixup" in exp and not "mmixup" in exp:
            parts.append(load_mixup(paths.MIX_OUTPUT_DIR, PixelMixupDataset))
        if "mmixup" in exp:
            parts.append(load_mixup(paths.MMIX_OUTPUT_DIR, PixelMixupDataset))
        if "gemix" in exp:
            parts.append(load_mixup(paths.GEMIX_OUTPUT_DIR, GeMixupDataset))

        # Combine real training split with mixup parts
        train_dataset = ConcatDataset([train_real, *parts])

    return train_dataset, val_dataset


def get_test_loader(config):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    test_dataset = datasets.ImageFolder(paths.TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    return test_loader
