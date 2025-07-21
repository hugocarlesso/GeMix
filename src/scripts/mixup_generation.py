from configs.mixup import DatMixConfig
from mixup_generation import (
    generate_mixup,
    generate_mmixup,
    generate_gemix,
)
import sys
from pathlib import Path

# Add the parent folder of 'torch_utils' to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))


def run_mixup_generation(cfg: DatMixConfig) -> None:
    """
    Run MixUp, Multi-MixUp, and GAN-based MixUp (GeMix) data generation.
    """
    if cfg.run_mixup_generation:
        print(
            f"[STEP] Generating basic MixUp dataset ({cfg.nbr_images_to_generate} images)…"
        )
        generate_mixup(cfg)

    if cfg.run_mmixup_generation:
        print(
            f"[STEP] Generating Multi-MixUp dataset ({cfg.nbr_images_to_generate} images)…"
        )
        generate_mmixup(cfg)

    if cfg.run_gemix_generation:
        print(
            f"[STEP] Generating GAN-based MixUp (GeMix) dataset ({cfg.nbr_images_to_generate} images)…"
        )
        generate_gemix(cfg)


if __name__ == "__main__":
    cfg = DatMixConfig()
    run_mixup_generation(cfg)
