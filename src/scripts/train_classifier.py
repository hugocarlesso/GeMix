import torch
import torch.nn as nn
import tqdm
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from dataset import get_dataset
from utils.model_factory import get_model
from configs.train_config import Config


def training(config, device):
    seed = getattr(config, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare data
    train_dataset, val_dataset = get_dataset(config)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=lambda worker_id: random.seed(seed + worker_id),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=lambda worker_id: random.seed(seed + worker_id),
        generator=generator,
    )

    # Initialize model, loss, optimizer
    model = get_model(config, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_acc = 0.0
    for epoch in range(config.num_epoch):
        model.train()
        total_train_loss = 0.0

        for images, labels in tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config.num_epoch} [Train]"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)

        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm.tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{config.num_epoch} [Val]"
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                targets = torch.argmax(labels, dim=1)
                correct += (preds == targets).sum().item()
                total += labels.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"\nEpoch [{epoch+1}/{config.num_epoch}] - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f}\n"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(config.best_model_path), exist_ok=True)
            torch.save(model.state_dict(), config.best_model_path)
            print(f"Best model saved at epoch {epoch+1} with val acc: {val_acc:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    training(config, device)
