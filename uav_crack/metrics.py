"""Streaming binary segmentation metrics."""

import torch


class BinaryMetrics:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> None:
        pred = torch.sigmoid(logits) >= threshold
        truth = target >= 0.5
        self.tp += int((pred & truth).sum().item())
        self.fp += int((pred & ~truth).sum().item())
        self.fn += int((~pred & truth).sum().item())
        self.tn += int((~pred & ~truth).sum().item())

    def compute(self) -> dict[str, float]:
        eps = 1e-9
        crack_iou = self.tp / (self.tp + self.fp + self.fn + eps)
        background_iou = self.tn / (self.tn + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {"mIoU": (crack_iou + background_iou) / 2, "crackIoU": crack_iou,
                "crackPrecision": precision, "crackRecall": recall, "crackF1": f1, "pixelAccuracy": accuracy}
