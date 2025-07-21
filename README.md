"""
# GeMix – GAN-Mixup CT Classification 🧠✨

**GeMix** is a modular framework to generate GAN-based mixup CT datasets and train robust classifiers.

---

## 📘 Table of Contents

1. [Overview](#overview)  
2. [Highlights](#highlights)  
3. [Installation](#installation)  
4. [Usage](#usage)  
   - [Data Preparation](#data-preparation)  
   - [GAN Training](#gan-training)  
   - [Mixup Generation](#mixup-generation)  
   - [Classifier Training](#classifier-training)  
5. [Data & Project Structure](#data--project-structure)  
6. [Mixup Methods](#mixup-methods)  
7. [Best Practices](#best-practices)  
8. [Citation & Links](#citation--links)  
9. [License](#license)  
10. [Contact](#contact)

---

## 🌟 Overview

GeMix leverages **StyleGAN2-ADA** to generate synthetic CT images and implements various mixup strategies (e.g., standard mixup, generalized mixup, gan-based mixup). These are used to train classifiers for improved robustness and performance on CT classification benchmarks.

---

## ✅ Highlights

- 🎨 Synthetic data generation via StyleGAN2-ADA  
- 🧪 Classifier training with real and synthetic data  
- ⚙️ Fully modular and reproducible pipeline  
- 📁 Predefined folder structure and configuration system  

---

## 🛠 Installation

### 1. Clone the repository:

### 2. Install dependencies:
```bash
pip install -e .
```

### 3. (Optional) Enable Git LFS for large models:
```bash
git lfs install
git lfs track "*.pt" "*.pkl"
git add .gitattributes
git commit -m "Track large models with Git LFS"
```

---

## 🚀 Usage

### 🔹 Data Preparation

1. Download the [COVIDx-CT dataset](https://www.kaggle.com/datasets/hgunraj/covidxct) from Kaggle.

2. Place the files in the following directory structure:

   - `data/full/images/` — contains all CT images  
   - `data/full/labels.csv` — contains image labels

3. Customize dataset split parameters in the configuration file:

   ```bash
   src/configs/data_generation.py
   ```

   This includes:

   - Number of images for GAN training  
   - Number of images for classifier training  
   - Size of the test set

4. Run the data preparation script from the project root:

   ```bash
   python src/scripts/data_prepare.py
   ```

   This script will:

   - Create subset folders (`train_gan/`, `real_train_classifier/`, `test/`)  
   - Distribute the data according to the configuration  
   - Prepare all required folders for GAN and classifier training

---

### 🔹 GAN Training

Use the notebook `src/conditional_gan.ipynb` for training the StyleGAN2-ADA model.

- Training must be performed on zipped datasets, as required by StyleGAN2-ADA.
- Compatible with [Google Colab](https://colab.research.google.com/).

For more information, refer to the official [StyleGAN2-ADA-PyTorch repo](https://github.com/NVlabs/stylegan2-ada-pytorch).

---

### 🔹 Mixup Generation

Generate synthetic mixup datasets using the script:

```bash
python src/scripts/mixup_generation.py
```

This will generate:

- `data/mixups/mixup/` — standard mixup  
- `data/mixups/gemixup/` — gan-based mixup 
- `data/mixups/mmixup/` — generaized mixup 

---

### 🔹 Classifier Training

Train CT classifiers using any of the prepared datasets:

```bash
python src/scripts/train_classifier.py 
```
---

## 🗂 Data & Project Structure

```
data/
├── full/                   # Original dataset (images + labels.csv)
├── train_gan/              # Subset for GAN training (real images)
├── real_train_classifier/  # Subset for real image classifier training (real images)
├── mixups/                 # Mixup datasets (mixup, gemixup, mmixup)
└── test/                   # Final test set (real images)

model_weights/              # Pretrained GAN and classifier weights

src/
├── scripts/                # All executable scripts
├── configs/                # Configurable settings for experiments
├── utils/                  # Data utilities
└── conditional_gan.ipynb   # Notebook for GAN training

setup.py, README.md, LICENSE, .gitignore, .gitattributes
```

---

## 🧪 Mixup Methods

- `mixup`: Interpolation between two real CT images  
- `gemixup`: GAN-generated interpolated images  
- `mmixup`: Interpolation between one image from all classes real CT images 


---

## 🧼 Best Practices

- Use **Git LFS** to track `.pt`, `.pkl`, and other large binary files
- Keep config files versioned for reproducibility
- Validate each mixup method with the same test set for fair comparison

---

## 📚 Citation & Links

**Related Resources**:

- [COVIDx-CT dataset](https://www.kaggle.com/datasets/hgunraj/covidxct)  
- [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 💬 Contact

