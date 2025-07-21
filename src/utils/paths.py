import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(REPO_ROOT)
# Base of the full dataset
BASE_DATA_DIR = os.path.join(REPO_ROOT, "data", "full")
IMAGE_DIR = os.path.join(BASE_DATA_DIR, "images")

# Label files
TRAIN_LABEL_FILE = os.path.join(BASE_DATA_DIR, "train_COVIDx_CT-3A.txt")
TEST_LABEL_FILE = os.path.join(BASE_DATA_DIR, "test_COVIDx_CT-3A.txt")

# Output folders
TEST_DIR = os.path.join(REPO_ROOT, "data", "test")
TRAIN_GAN_DIR = os.path.join(REPO_ROOT, "data", "train_gan")
REAL_CLASSIFIER_DIR = os.path.join(REPO_ROOT, "data", "real_train_classifier")

# GAN
NETWORK_PKL = os.path.join(
    REPO_ROOT, "model_weights", "stylegan", "network-snapshot-000400.pkl"
)

# Mixup folders
MIXUPS_OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "mixups")
GEMIX_OUTPUT_DIR = os.path.join(MIXUPS_OUTPUT_DIR, "gemixup")

# MixUp “traditionnel”
MIX_OUTPUT_DIR = os.path.join(MIXUPS_OUTPUT_DIR, "mixup")
MMIX_OUTPUT_DIR = os.path.join(MIXUPS_OUTPUT_DIR, "mmixup")


# Weights folder
PATH_WEIGHTS_FOLDER = os.path.join(REPO_ROOT, "model_weights")

CSV_FILENAME = "labels.csv"
