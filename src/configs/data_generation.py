from dataclasses import dataclass
import utils.paths as paths


@dataclass
class DataPrepConfig:
    # ——— Label files & data dirs ———
    train_label_file: str = paths.TRAIN_LABEL_FILE
    test_label_file: str = paths.TEST_LABEL_FILE
    image_dir: str = paths.IMAGE_DIR

    # ——— Output dirs ———
    test_output_dir: str = paths.TEST_DIR
    train_gan_output_dir: str = paths.TRAIN_GAN_DIR
    real_output_dir: str = paths.REAL_CLASSIFIER_DIR

    # ——— Subset sizes ———
    test_subset_size: int = 1000
    train_gan_subset_size: int = 10000
    real_subset_size: int = 10000

    # ——— Which steps to run ———
    run_test_subset: bool = True
    run_train_gan_subset: bool = True
    run_real_dataset: bool = True
