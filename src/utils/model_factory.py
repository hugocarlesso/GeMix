import torch.nn as nn
from torchvision import models


def get_model(config, device):
    if config.model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(device)
    elif config.model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(device)
    elif config.model_name == "effnet":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        model = model.to(device)
    return model
