from utils import paths
import os


class Config:
    def __init__(self):
        self.img_size = 128
        self.num_epoch = 5
        self.batch_size = 64
        self.lr = 1e-4
        self.val_ratio = 0.5
        self.model_name = "resnet50"
        self.num_workers = 0
        self.expe_name = "real_only"
        self.best_model_path = os.path.join(
            paths.PATH_WEIGHTS_FOLDER, self.model_name, self.expe_name + ".pth"
        )
