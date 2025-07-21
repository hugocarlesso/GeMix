from dataclasses import dataclass
import torch
import utils.paths as paths


@dataclass
class DatMixConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ——— Label files & data dirs ———
    train_real_image_dir: str = paths.REAL_CLASSIFIER_DIR
    img_size: tuple = (128, 128)

    # ——— Output dirs ———
    mixup_output_dir: str = paths.MIX_OUTPUT_DIR
    mmix_output_dir: str = paths.MMIX_OUTPUT_DIR
    gemix_output_dir: str = paths.GEMIX_OUTPUT_DIR

    nbr_images_to_generate: int = 1
    same_class_ratio: float = 0.5
    csv_file_name: str = paths.CSV_FILENAME
    maj = 2
    min = 1

    # ——— Which steps to run ———
    run_mixup_generation: bool = True
    run_mmixup_generation: bool = True
    run_gemix_generation: bool = True
