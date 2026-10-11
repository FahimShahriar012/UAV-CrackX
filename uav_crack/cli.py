"""Command-line training, benchmark prediction, and visualization."""

import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DEFAULT_SCALES, CrackDataset, discover_pairs
from .metrics import BinaryMetrics
from .model import UNet, segmentation_loss


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
    return torch.device(name)


def _partition(pairs, fraction: float, seed: int):
    rng = random.Random(seed)
    train, valid = [], []
    for scale in sorted({pair[2] for pair in pairs}):
        group = [p for p in pairs if p[2] == scale]
        rng.shuffle(group)
        count = max(1, round(len(group) * fraction))
        valid.extend(group[:count])
        train.extend(group[count:])
    return train, valid


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    pairs = discover_pairs(Path(args.data_root), args.scales)
    train_pairs, valid_pairs = _partition(pairs, args.validation_fraction, args.seed)
    if not train_pairs:
        raise ValueError("No training images remain; reduce --validation-fraction.")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(CrackDataset(train_pairs, augment=True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda",
                              generator=generator)
    valid_loader = DataLoader(CrackDataset(valid_pairs), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = UNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    history = []
    start_epoch = 1
    if args.resume:
        resume_state = torch.load(Path(args.resume), map_location=device, weights_only=False)
        if int(resume_state.get("base_channels", args.base_channels)) != args.base_channels:
            raise ValueError("--base-channels must match the checkpoint when resuming.")
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_iou = float(resume_state.get("best_iou", -1.0))
        history = resume_state.get("history", [])
        print(f"Resuming at epoch {start_epoch} from {args.resume}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)
        model.eval()
        metrics = BinaryMetrics()
        with torch.no_grad():
            for images, masks in valid_loader:
                metrics.update(model(images.to(device)), masks.to(device), threshold=args.threshold)
        scores = metrics.compute()
        record = {"epoch": epoch, "trainLoss": loss_sum / len(train_pairs), **scores}
        history.append(record)
        print(json.dumps(record))
        is_best = scores["crackIoU"] > best_iou
        if is_best:
            best_iou = scores["crackIoU"]
        checkpoint = {"model": model.state_dict(), "base_channels": args.base_channels,
                      "threshold": args.threshold, "epoch": epoch, "best_iou": best_iou,
                      "metrics": scores, "history": history}
        latest_tmp = output / "latest.pt.tmp"
        torch.save({**checkpoint, "optimizer": optimizer.state_dict()}, latest_tmp)
        latest_tmp.replace(output / "latest.pt")
        if is_best:
            best_tmp = output / "best.pt.tmp"
            torch.save(checkpoint, best_tmp)
            best_tmp.replace(output / "best.pt")
        (output / "metrics.json").write_text(json.dumps({"history": history, "trainImages": len(train_pairs),
                                                           "validationImages": len(valid_pairs)}, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {output / 'best.pt'} (holdout crack IoU {best_iou:.4f})")


def _load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = UNet(int(state.get("base_channels", 32))).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, float(state.get("threshold", 0.5))


@torch.no_grad()
def _predict_one(model: UNet, path: Path, device: torch.device, threshold: float) -> Image.Image:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return Image.fromarray((probability >= threshold).astype(np.uint8), mode="L")


def predict(args: argparse.Namespace) -> None:
    device = _device(args.device)
    model, stored_threshold = _load_model(Path(args.checkpoint), device)
    threshold = args.threshold if args.threshold is not None else stored_threshold
    data_root, output_zip = Path(args.data_root), Path(args.output)
    output_dir = output_zip.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for scale in args.scales:
        image_paths.extend(sorted(p for p in (data_root / "leftImg8bit" / "val" / scale).iterdir()
                                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}))
    if not image_paths:
        raise FileNotFoundError(f"No benchmark images found under {data_root / 'leftImg8bit' / 'val'}")
    if len({p.name.lower() for p in image_paths}) != len(image_paths):
        raise ValueError("Duplicate input basenames across scale folders cannot be represented in a flat submission.")
    sizes = set()
    for source in tqdm(image_paths, desc="Predicting masks"):
        mask = _predict_one(model, source, device, threshold)
        sizes.add(mask.size)
        mask.save(output_dir / f"{source.stem}.png", format="PNG")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in image_paths:
            mask_path = output_dir / f"{source.stem}.png"
            archive.write(mask_path, arcname=f"result/{mask_path.name}")
    print(f"Wrote {len(image_paths)} masks ({len(sizes)} distinct dimensions) to {output_zip}")
    if len(image_paths) != 300:
        print("Note: the selected dataset split did not contain 300 images; verify the benchmark submission count.")
    bundle_path = Path(args.bundle_output) if args.bundle_output else None
    if bundle_path is not None:
        _create_complete_bundle(data_root, Path(args.checkpoint), output_zip, bundle_path)


def _create_complete_bundle(data_root: Path, checkpoint: Path, submission_zip: Path, bundle_path: Path) -> None:
    """Collect reproducibility artifacts and the source dataset into one archive."""
    project_root = Path(__file__).resolve().parent.parent
    dataset_files = sorted(p for p in data_root.rglob("*") if p.is_file())
    dataset_bytes = sum(p.stat().st_size for p in dataset_files)
    metrics_file = checkpoint.parent / "metrics.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(checkpoint, arcname="training/best.pt")
        if metrics_file.exists():
            archive.write(metrics_file, arcname="training/metrics.json")
        archive.write(submission_zip, arcname="submission/result.zip")
        for name in ("README.md", "requirements.txt", ".gitignore"):
            path = project_root / name
            if path.exists():
                archive.write(path, arcname=f"project/{name}")
        for path in sorted((project_root / "uav_crack").rglob("*.py")):
            archive.write(path, arcname=f"project/uav_crack/{path.relative_to(project_root / 'uav_crack')}")
        for path in dataset_files:
            archive.write(path, arcname=f"dataset/{data_root.name}/{path.relative_to(data_root)}")
        manifest = {
            "datasetRootName": data_root.name,
            "datasetFiles": len(dataset_files),
            "datasetBytes": dataset_bytes,
            "checkpoint": "training/best.pt",
            "metrics": "training/metrics.json" if metrics_file.exists() else None,
            "benchmarkMasks": "submission/result.zip",
            "projectSource": "project/",
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    print(f"Complete project, dataset, checkpoint, metrics, and submission saved to {bundle_path}")


def visualize(args: argparse.Namespace) -> None:
    path = Path(args.mask)
    mask = Image.open(path).convert("L")
    array = np.asarray(mask)
    foreground = array > 0
    preview = np.asarray(Image.new("RGB", mask.size, (205, 205, 205))).copy()
    preview[foreground] = (220, 30, 30)
    out = Path(args.output) if args.output else path.with_name(path.stem + "_visualized.png")
    Image.fromarray(preview).save(out)
    print(f"Saved visualization: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uav-crack", description="UAV-CrackX crack segmentation baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="Train and select a model using a holdout from labeled training data")
    p.add_argument("--data-root", required=True, help="Path to UAV-CrackX directory containing gtFine and leftImg8bit")
    p.add_argument("--output", default="runs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--resume", default=None, help="Resume training from a latest.pt checkpoint")
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    p.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    p.set_defaults(func=train)
    p = sub.add_parser("predict", help="Predict the unlabeled benchmark split and create result.zip")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="result.zip")
    p.add_argument("--bundle-output", default="UAV-CrackX_complete_results.zip",
                   help="Also bundle project code, source dataset, checkpoint, metrics, and submission; use an empty value to skip")
    p.add_argument("--device", default="auto")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    p.set_defaults(func=predict)
    p = sub.add_parser("visualize", help="Create a red-on-gray preview of a binary mask")
    p.add_argument("mask")
    p.add_argument("--output", default=None)
    p.set_defaults(func=visualize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
