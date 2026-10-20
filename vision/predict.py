import json
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from vision.gradcam import GradCAM, overlay
from vision.model import build_model, gradcam_layer

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def crop_transform(size):
    return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size)])


def tensor_transform():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


@dataclass
class Bundle:
    model: torch.nn.Module
    classes: list
    arch: str
    img_size: int
    device: torch.device


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model, classes, arch, img_size):
    torch.save(dict(state_dict=model.state_dict(), classes=classes, arch=arch, img_size=img_size), path)


def load_checkpoint(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return Bundle(model.to(device).eval(), ck["classes"], ck["arch"], ck["img_size"], device)


def explain(bundle, pil_image, class_idx=None, topk=5, alpha=0.45):
    """Classify one PIL image and return top-k predictions plus a Grad-CAM overlay for `class_idx`
    (default: the predicted class)."""
    crop = crop_transform(bundle.img_size)(pil_image.convert("RGB"))
    x = tensor_transform()(crop).unsqueeze(0).to(bundle.device)
    cam_fn = GradCAM(bundle.model, gradcam_layer(bundle.model, bundle.arch))
    try:
        cam, probs, idx = cam_fn(x, class_idx)
    finally:
        cam_fn.remove()
    order = np.argsort(-probs)[:topk]
    return dict(top=[(bundle.classes[i], float(probs[i]), int(i)) for i in order], crop=crop, cam=cam,
                overlay=overlay(np.asarray(crop), cam, alpha), explained=idx)
