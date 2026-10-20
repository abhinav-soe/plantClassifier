"""Model construction: pretrained backbone + new classification head."""
import torch.nn as nn
from torchvision import models

ARCHS = ("resnet18", "resnet50", "efficientnet_b0")


def build_model(arch, num_classes, pretrained=True):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"arch must be one of {ARCHS}")
    return m


def head(model, arch):
    return model.fc if arch.startswith("resnet") else model.classifier


def gradcam_layer(model, arch):
    """Last convolutional block: spatially coarse but semantically rich - the standard Grad-CAM target."""
    return model.layer4[-1] if arch.startswith("resnet") else model.features[-1]


def set_backbone_trainable(model, arch, trainable):
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head(model, arch).parameters():
        p.requires_grad = True


def split_params(model, arch):
    head_ids = {id(p) for p in head(model, arch).parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, list(head(model, arch).parameters())
