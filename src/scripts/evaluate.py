import os
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from utils import paths
from utils.model_factory import get_model
from dataset import get_test_loader
from configs.test_config import Config


model_name_dico = {
    "effnet": "EfficientNet",
    "resnet101": "ResNet101",
    "resnet50": "ResNet50",
}
expe_dico = {
    "real_mmixup.pth": "Real+MMix-Up",
    "real_gemix.pth": "Real+GeMix",
    "gemix_only.pth": "GeMix-Up",
    "mmix_only.pth": "MMix-Up",
    "real_mmixup_gemix.pth": "Real+MMix-Up+GeMix",
}


def evaluate_model(config):
    """
    Evaluate a trained model on the test dataset and produce metrics and a confusion matrix plot.

    This function:
      - Reloads the model specified by `model_name` and `expe` from the weights folder in `config`.
      - Runs inference on the test set obtained via `get_test_loader(config)`.
      - Computes accuracy, precision, recall, and F1 score (macro-average).
      - Prints a full classification report and raw confusion matrix.
      - Plots and saves a normalized confusion matrix heatmap to the weights folder.

    Parameters
    ----------
    config : object
        Configuration object or namespace containing at least:
          - model_name (str)
          - expe (str)
          - any other settings used by `get_test_loader`.

    Returns
    -------
    dict
        A dictionary with keys:
          - "experiment": base name of the experiment (without extension),
          - "accuracy": overall accuracy,
          - "precision": macro-averaged precision,
          - "recall": macro-averaged recall,
          - "f1_score": macro-averaged F1 score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tick_font_size = 36
    annotation_font_size = 60
    title_font_size = 34
    x_y_font_size = 34
    model_name = config.model_name
    expe = config.expe
    test_loader = get_test_loader(config)
    class_names = test_loader.dataset.classes
    label = expe.split(".")[0]
    print(f"\nEvaluating model: {label}")

    # Load the model and its weights
    model = get_model(config, device)
    model_path = os.path.join(paths.PATH_WEIGHTS_FOLDER, model_name, expe)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {label}"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")

    print(f"\nClassification Report ({label}):\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix ({label}):\n{cm}")

    # Normalize confusion matrix to percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plotting the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        annot_kws={"size": annotation_font_size},
        cbar_kws={"label": "Percentage (%)"},
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=2,
        linecolor="black",
        ax=ax,
    )

    # Title and axes labels
    ax.set_title(
        f"Confusion Matrix {model_name_dico[model_name]}: {expe_dico[expe]}",
        fontsize=title_font_size,
        pad=20,
    )
    ax.set_xlabel("Predicted", fontsize=x_y_font_size, labelpad=10)
    ax.set_ylabel("Actual", fontsize=x_y_font_size, labelpad=10)

    # Axes ticks
    ax.tick_params(axis="x", labelsize=tick_font_size)
    ax.tick_params(axis="y", labelsize=tick_font_size)

    # rotation + alignement si besoin
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)

    for tick in ax.get_yticklabels():
        tick.set_rotation(90)

    # colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=40)
    cbar.set_label("Percentage (%)", fontsize=x_y_font_size)

    plt.tight_layout()
    plt.savefig(
        os.path.join(paths.PATH_WEIGHTS_FOLDER, model_name, f"cm_pct_{label}.png"),
        dpi=300,
    )
    plt.close()

    return {
        "experiment": label,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


if __name__ == "__main__":
    config = Config()
    results = evaluate_model(config)
