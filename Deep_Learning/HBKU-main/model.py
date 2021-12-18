# model
from torchvision import models
import numpy as np
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
# unfreeze percent of layers


def unfreeze(model_ft, percent=0.4):
    x = int(np.ceil(len(model_ft._modules.keys()) * percent))
    x = list(model_ft._modules.keys())[-x:]
    print(f"unfreezing these layer {x}",)
    for name in x:
        for params in model_ft._modules[name].parameters():
            params.requires_grad_(True)

    return model_ft


def freeze_layers(model_ft):
    # freeze all layers
    for param in model_ft.parameters():
        param.requires_grad = False

    return model_ft


def unfreeze_percentage(model_ft, percent=0.4):
    # unfreeze 60% of layers
    unfreeze(model_ft, percent)
    # unfreeze 60% of convolutional bloks
    unfreeze(model_ft._blocks, percent)

    return model_ft


def check_freeze(model):
    for name, layer in model._modules.items():
        s = []
        for x in layer.parameters():
            s.append(x.requires_grad)
        # print(name, all(s))


# check_freeze(model_ft)

# ----------------------------------------------------------------------- #
# Initialize model #
# ----------------------------------------------------------------------- #

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    if model_name == 'resnet':
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet':
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained(
                "efficientnet-b2", num_classes=num_classes)
            model_ft = freeze_layers(model_ft)
            check_freeze(model_ft)
            model_ft = unfreeze_percentage(model_ft)
            check_freeze(model_ft)
        else:
            model_ft = EfficientNet.from_name('efficientnet-b2')

    return model_ft


# --------------------------------------------------------------- #
