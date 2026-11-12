"""Grad-CAM on the last residual stage (layer4) of ResNet-18 [paper, Eq. 7]. Used qualitatively only."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model.eval()
        self.acts = self.grads = None
        self._h1 = layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.acts = output
        output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        """x: (1, 3, H, W). Returns (map in [0,1] at H x W, predicted probability)."""
        self.model.zero_grad()
        logit = self.model(x)[0, 0]
        logit.backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # alpha_k
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))  # ReLU(sum_k alpha_k A^k)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, float(torch.sigmoid(logit).item())

    def close(self):
        self._h1.remove()
