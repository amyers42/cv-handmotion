#!/usr/bin/env python3
"""
Train **HandyNet** on the 7‑class HaGRID subset (call, dislike, fist, like,
OK, one, palm) **using the JSON‑based annotation layout** described by the
user:

hagrid_data/
 ├── call/                 # image folders (one per class)
 ├── dislike/
 ├── ...
 └── annotations/
     ├── train/
     │    ├── call.json    # list of image tags + metadata
     │    ├── ...
     ├── val/
     └── test/

Each JSON contains an *array* where every element has at least a tag that
identifies the image file.  We only need that tag; metadata are ignored.
The script resolves the actual file by looking for `tag`, `tag.jpg`,
`tag.jpeg`, or `tag.png` under the corresponding class folder.

Run:
    python train.py --data-root /path/to/hagrid_data --epochs 100

This replicates the HaGRID paper’s training recipe
(resize‑then‑center‑crop → 224×224, SGD 0.1, ReduceLROnPlateau, early stop).
"""
import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "call", "dislike", "fist", "like", "ok", "one", "palm",
]
NUM_CLASSES = len(CLASS_NAMES)
CARDINALITY_ITEM = 16
INPUT_CHANNELS = 3

# HaGRID per‑channel statistics (paper, suppl. Fig. 3)
MEAN = [0.54, 0.499, 0.473]
STD  = [0.231, 0.232, 0.229]

# ────────────────────────────────────────────────────────────────────────────────
#  Dataset
# ────────────────────────────────────────────────────────────────────────────────
class HagridJsonDataset(Dataset):
    """Dataset that reads class‑specific JSON tag lists and loads images."""

    def __init__(self, root: Path | str, split: str, tfms: transforms.Compose,
                 classes: list[str] | None = None):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.tfms = tfms
        self.classes = classes if classes is not None else CLASS_NAMES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []

        for cls in self.classes:
            jpath = self.root / "annotations" / split / f"{cls}.json"
            if not jpath.exists():
                raise FileNotFoundError(f"Missing annotation file {jpath}")
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            for entry in data:
                tag = entry  # default: raw element is tag
                if isinstance(entry, dict):
                    # common key guesses
                    for k in ("id", "tag", "image", "file_name"):
                        if k in entry:
                            tag = entry[k]
                            break
                img_path = self._resolve_image_path(tag, cls)
                if img_path is not None:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError("No samples found. Check paths and JSON tags.")

    # ---------------------------------------------------------------------
    def _resolve_image_path(self, tag: str, cls: str) -> Path | None:
        """Return an existing image path for *tag* under class folder *cls*."""
        cls_dir = self.root / cls
        # tag may already include extension or relative dir
        candidate = cls_dir / tag if not os.path.isabs(tag) else Path(tag)
        if candidate.exists():
            return candidate
        # try common extensions
        for ext in (".jpg", ".jpeg", ".png"):
            p = cls_dir / f"{tag}{ext}"
            if p.exists():
                return p
        # failed – silently skip (could log)
        return None

    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ---------------------------------------------------------------------
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.tfms:
            img = self.tfms(img)
        return img, label

# ────────────────────────────────────────────────────────────────────────────────
#  Model (HandyNet)
# ────────────────────────────────────────────────────────────────────────────────
class ResidualUnit(nn.Module):
    def __init__(self, channels: int, k: int, dilation: int):
        super().__init__()
        pad = (k - 1) * dilation // 2
        groups = max(1, channels // CARDINALITY_ITEM)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, k, padding=pad,
                               dilation=dilation, groups=groups, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, k, padding=pad,
                               dilation=dilation, groups=groups, bias=False)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        return x + y

class Skip(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x, skip):
        return x, self.conv(x) + skip

class HandyNet(nn.Module):
    def __init__(self, base_width: int = 64):
        super().__init__()
        L = base_width
        self.stem = nn.Conv2d(INPUT_CHANNELS, L, 1, bias=False)
        self.skip1 = Skip(L)

        ks = [11]*8 + [21]*4 + [41]*4
        ds = [1]*4 + [4]*4 + [10]*4 + [25]*4
        blocks = []
        for i, (k, d) in enumerate(zip(ks, ds)):
            blocks.append(ResidualUnit(L, k, d))
            if (i + 1) % 4 == 0:
                blocks.append(Skip(L))
        if (len(ks) + 1) % 4 != 0:
            blocks.append(Skip(L))
        self.blocks = nn.ModuleList(blocks)
        self.last_conv = nn.Conv2d(L, L, 1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(L, NUM_CLASSES)

    def forward(self, x):
        x = self.stem(x)                       # (B, 64, H, W)
        skip = torch.zeros_like(x)             # now also (B, 64, H, W)
        x, skip = self.skip1(x, skip)

        for m in self.blocks:
            if isinstance(m, Skip):
                x, skip = m(x, skip)
            else:
                x = m(x)
        x = self.last_conv(skip)
        x = self.avgpool(x).flatten(1)
        return self.head(x)

# ────────────────────────────────────────────────────────────────────────────────
#  Utilities
# ────────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False


def get_loaders(root: Path, batch_size: int, workers: int):
    tfms_train = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tfms_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = HagridJsonDataset(root, "train", tfms_train)
    val_ds   = HagridJsonDataset(root, "val", tfms_val)
    test_ds  = HagridJsonDataset(root, "test", tfms_val)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=True)
    return train_ld, val_ld, test_ld


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            preds.extend(logits.argmax(1).cpu().numpy())
            gts.extend(y.cpu().numpy())
    return f1_score(gts, preds, average="macro")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

# ────────────────────────────────────────────────────────────────────────────────
#  Entry‑point
# ────────────────────────────────────────────────────────────────────────────────

def main(cfg):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_ld, val_ld, test_ld = get_loaders(Path(cfg.data_root), cfg.batch_size, cfg.workers)

    # model & optimiser
    model = HandyNet(base_width=cfg.width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10, verbose=True)

    best_f1, best_epoch = 0.0, -1
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}  lr={optimizer.param_groups[0]['lr']:.3e}")
        tr_loss = train_epoch(model, train_ld, criterion, optimizer, device)
        val_f1 = evaluate(model, val_ld, device)
        print(f"train loss {tr_loss:.4f} · val F1 {val_f1:.4f}")
        scheduler.step(val_f1)

        if val_f1 > best_f1 + 1e-4:
            best_f1, best_epoch = val_f1, epoch
            ckpt = {"epoch": epoch, "f1": best_f1, "state_dict": model.state_dict()}
            ckpt_path = Path(cfg.out_dir) / "best.pt"
            torch.save(ckpt, ckpt_path)
            print(f"↑ new best (F1={best_f1:.4f}) saved to {ckpt_path}")

        if epoch - best_epoch >= 10:
            print("Early stopping – no val improvement for 10 epochs.")
            break

    # test
    model.load_state_dict(torch.load(Path(cfg.out_dir) / "best.pt", map_location=device)["state_dict"])
    test_f1 = evaluate(model, test_ld, device)
    print(f"\nTest‑set F1: {test_f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=str, required=True, help="path to hagrid_data")
    p.add_argument("--out-dir", type=str, default="runs/hagrid7", help="checkpoint directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=64, help="base channel width L")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
