"""B7 消融: binary (empty vs non-empty) TinyGate — 与 v4 3-class 协议对齐.

差异仅: 标签 1/2 塌缩为 1 (non-empty), NUM_CLASSES=2. 其余同 train_tiny_gate.py --v4:
160 抽取式下采样, 20 epochs, AdamW 3e-4, bs 24, AMP, val top1 存最优.
产物: runs/gate/v4bin/{best.pt, best.onnx, summary.json}
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from train_tiny_gate import TinyGate, export_onnx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GATE_ROOT = Path(__file__).parent / "gate"
OUT_ROOT = Path(__file__).parent / "runs" / "gate" / "v4bin"
SEED = 42
EPOCHS = 20
BATCH = 24
LR = 3e-4
WD = 0.01
WORKERS = 4
NUM_CLASSES = 2  # 0=empty, 1=non-empty


class BinCrops(Dataset):
    """gate_{split}.txt, 标签 1/2 -> 1. 预处理与 v4 逐位一致 (抽取式下采样)."""

    def __init__(self, list_file: Path, train: bool):
        self.root = GATE_ROOT
        self.train = train
        self.items: list[tuple[str, int]] = []
        for ln in list_file.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            p, c = ln.rsplit(" ", 1)
            c = 1 if int(c) > 0 else 0
            self.items.append((p, c))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, c = self.items[i]
        img = Image.open(self.root / p).convert("RGB")
        assert img.size == (640, 640)
        arr = np.asarray(img)[0::4, 0::4, :]
        x = torch.from_numpy(arr.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).contiguous()
        if self.train:
            if random.random() < 0.5:
                x = torch.flip(x, [2])
            if random.random() < 0.5:
                x = torch.flip(x, [1])
        return x, c


def evaluate(model, loader, device):
    model.eval()
    cm = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(
                logits, y, label_smoothing=0.05, reduction="sum").item()
            pred = logits.argmax(1)
            for gt, pr in zip(y.tolist(), pred.tolist()):
                cm[gt][pr] += 1
            n += y.numel()
    total = sum(sum(r) for r in cm)
    top1 = sum(cm[i][i] for i in range(NUM_CLASSES)) / max(1, total)
    gt_cnt = [sum(cm[i]) for i in range(NUM_CLASSES)]
    recall = [cm[i][i] / gt_cnt[i] if gt_cnt[i] else float("nan")
              for i in range(NUM_CLASSES)]
    false_skip = cm[1][0] / gt_cnt[1] if gt_cnt[1] else float("nan")
    return {"top1": top1, "loss": loss_sum / max(1, n), "cm": cm,
            "recall": recall, "false_skip_rate": false_skip}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] torch={torch.__version__} device={device}")

    train_ds = BinCrops(GATE_ROOT / "gate_train.txt", train=True)
    val_ds = BinCrops(GATE_ROOT / "gate_val.txt", train=False)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} (binary: empty vs non-empty)")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=WORKERS, pin_memory=True,
                              persistent_workers=WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = TinyGate(num_classes=NUM_CLASSES, stem_stride=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] TinyGate-binary params={n_params} ({n_params / 1e6:.3f}M)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    best_top1, best_ep = -1.0, -1
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = __import__("time").time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        m = evaluate(model, val_loader, device)
        star = ""
        if m["top1"] > best_top1:
            best_top1, best_ep = m["top1"], ep
            torch.save(model.state_dict(), OUT_ROOT / "best.pt")
            star = " *"
        print(f"[ep {ep:02d}/{EPOCHS}] val top1={m['top1']:.4f} "
              f"empty_recall={m['recall'][0]:.4f} false_skip={m['false_skip_rate']:.4f} "
              f"loss={m['loss']:.4f} ({__import__('time').time() - t0:.0f}s){star}")

    model.load_state_dict(torch.load(OUT_ROOT / "best.pt", map_location=device,
                                     weights_only=True))
    m = evaluate(model, val_loader, device)
    onnx_info = export_onnx(model, OUT_ROOT / "best.onnx", 160)
    summary = {"arch": f"TinyGate binary (empty/non-empty) {n_params} params",
               "train": {"val_top1": m["top1"], "empty_recall": m["recall"][0],
                         "false_skip_rate": m["false_skip_rate"],
                         "epochs": EPOCHS, "best_ep": best_ep, "cm": m["cm"]},
               "onnx": onnx_info}
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] best ep {best_ep} val top1={best_top1:.4f} -> {OUT_ROOT}")


if __name__ == "__main__":
    main()