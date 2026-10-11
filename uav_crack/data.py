"""Dataset discovery and image/mask loading."""

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_SCALES = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")


def _files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def discover_pairs(data_root: Path, scales: Iterable[str] = DEFAULT_SCALES) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    missing: list[str] = []
    for scale in scales:
        image_dir = data_root / "leftImg8bit" / "train" / scale
        mask_dir = data_root / "gtFine" / "train" / scale
        masks = {p.stem: p for p in _files(mask_dir)}
        for image in _files(image_dir):
            mask = masks.get(image.stem)
            if mask is None:
                missing.append(image.name)
            else:
                pairs.append((image, mask, scale))
    if missing:
        raise ValueError(f"Missing masks for {len(missing)} images; examples: {', '.join(missing[:3])}")
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {data_root}; expected leftImg8bit/train and gtFine/train.")
    return pairs


class CrackDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path, str]], augment: bool = False) -> None:
        self.pairs = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path, _ = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            raise ValueError(f"Image/mask size mismatch: {image_path.name} {image.size}, {mask_path.name} {mask.size}")
        if self.augment:
            if torch.rand(()) < 0.5:
                image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if torch.rand(()) < 0.5:
                image, mask = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.uint8)
        # Dataset masks label crack pixels as 1; tolerate either 0/1 or 0/255 storage.
        target = (mask_array > 0).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(target[None].copy())
        return image_tensor, mask_tensor
