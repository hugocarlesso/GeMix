# GAN Mixup CT Classification

This repository provides code for generating mixup datasets using GANs and training classifiers for CT image classification.

## Folder Structure

- `data/`
  - `full/` – Download the full dataset here. Place images in `data/full/images/` and the labels file in `data/full/`.
  - `mixups/` – Contains generated mixup datasets (`gemixup/`, `mixup/`, `mmixup/`).
  - `real_train_classifier/` – Training data for classifier, organized by class.
  - `test/` – Test data, organized by class.
  - `train_gan/` – Training data for GAN.
- `model_weights/` – Pretrained weights for various models.
- `src/`
  - `scripts/` – Contains scripts for data preparation, mixup generation, and training.
  - `utils/paths.py` – Describes the proposed structure of the `data/` folder architecture.
  - Other modules for dataset handling and GAN training.

## Dataset Preparation

1. **Download the Dataset**
   - Download from [Kaggle: covidxct](https://www.kaggle.com/datasets/hgunraj/covidxct).
   - Place images in `data/full/images/` and the labels file in `data/full/`.

2. **Install the setup**
   ```sh
    pip install -e .
    ```
  This will install all libraries required for the workspace to run mixup data builds as well as classifiers training. 
  For the conditional gan training, use the notebook given in `src/conditional_gan.ipynb`, for our work, Google Colab has been used for GAN training.

3. **Configure Subset Generation**
   - Edit `src/configs/data_generation.py` to set the desired number of images for each subset (GAN training, classifier training, testing).
   - An example configuration is provided in the file.

## StyeGan2-ada training 
run all cells from `src/conditional_gan.ipynb` after setting your path to fit your architecture.
Beware, for training the StyleGan2-ada architecture you need to zip your folder. Refer to https://github.com/NVlabs/stylegan2-ada-pytorch for more details.

## Running Scripts

All main commands should be executed from the repository root.

- **Prepare Data Subsets:**
  ```sh
  python src/scripts/data_prepare.py
  ```
- **Generate Mixup Data:**
  ```sh
  python src/scripts/mixup_generation.py
  ```
- **Train Classifier:**
  ```sh
  python src/scripts/train_classifier.py
  ```

## Data Folder Architecture

See [`src/utils/paths.py`](src/utils/paths.py) for the proposed structure and organization of the `data/` folder.

## License

See repository for license information.

## Contact

For questions, please open an issue or contact the repository maintainer.
