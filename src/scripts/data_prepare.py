from configs.data_generation import DataPrepConfig
from src.data_preparation import (
    make_subset,
    make_real_dataset_excluding_gan,
)


def run_data_preparation(cfg: DataPrepConfig) -> None:
    """
    Execute the desired data preparation steps according to cfg flags.
    """
    if cfg.run_test_subset:
        print(f"[STEP] Creating test subset ({cfg.test_subset_size} images/class)…")
        make_subset(
            label_file=cfg.test_label_file,
            output_dir=cfg.test_output_dir,
            subset_size=cfg.test_subset_size,
        )

    if cfg.run_train_gan_subset:
        print(
            f"[STEP] Creating GAN training subset ({cfg.train_gan_subset_size} images/class)…"
        )
        make_subset(
            label_file=cfg.train_label_file,
            output_dir=cfg.train_gan_output_dir,
            subset_size=cfg.train_gan_subset_size,
        )

    if cfg.run_real_dataset:
        print(
            f"[STEP] Creating real classifier dataset ({cfg.real_subset_size} images/class)…"
        )
        make_real_dataset_excluding_gan(
            train_label_file=cfg.train_label_file,
            gan_dir=cfg.train_gan_output_dir,
            output_dir=cfg.real_output_dir,
            subset_size=cfg.real_subset_size,
        )


if __name__ == "__main__":
    cfg = DataPrepConfig()
    run_data_preparation(cfg)
