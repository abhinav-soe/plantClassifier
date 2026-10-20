"""Grad-CAM (Selvaraju et al., 2017), implemented from scratch.
Idea: weight each feature map of the last conv block by how much it increases the target class score
(average of its gradients), sum them, keep the positive part -> a coarse map of 'where the model looked'."""
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.act, self.grad = model, None, None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.act = output.detach()
        output.register_hook(lambda g: setattr(self, "grad", g.detach()))

    def __call__(self, x, class_idx=None):
        """x: (1,3,H,W) normalised tensor. Returns (cam in [0,1] of shape HxW, probabilities, explained class)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        probs = logits.softmax(1)[0].detach().cpu().numpy()
        idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
        logits[0, idx].backward()
        weights = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
        return cam, probs, idx

    def remove(self):
        self._handle.remove()


def overlay(image_rgb, cam, alpha=0.45):
    """Blend a jet-coloured heatmap onto an HxWx3 uint8 image."""
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    out = (1 - alpha) * (np.asarray(image_rgb, dtype=float) / 255) + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype("uint8")
