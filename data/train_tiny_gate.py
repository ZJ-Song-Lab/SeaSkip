"""SeaSkip 轻量 gate v2 训练(微 CNN, ~185k 参数) — 优化迭代.

背景: v1 gate (yolo11n-cls, ~1.5M 参数) 板端 9.08ms = det(14.29ms) 的 64%,
盈亏平衡空占比 gate/det = 63.5%, 测试流 vs 无 gate 基线净亏(HRSID -17.3% /
SSDD -23.8%). v2 目标: gate 延迟 1-2ms, 盈亏平衡降至 ~10%, 准确率不显著低于
v1 (HRSID test gate_acc 0.8891).

与 seaskip_eval.cpp 接口逐位对齐:
  输入  1x3x640x640 float32, RGB/255 (无 ImageNet 归一化, C++ 端同样处理)
  输出  3 个 raw logits (softmax 由 C++ 端完成; v1 ONNX 末尾自带 Softmax,
        板上 C++ 再做一次 -> 双重 softmax, p_empty 分布弥散、阈值悬崖)

架构: stem Conv3->32 k7 s4 + 4 x [DWConv k3 s2 + PWConv] (32->64->128->256->512)
      + GAP + Linear 512->3, 无 dropout (ATC 安全算子), ~189k 参数.
训练: 自然类分布(对齐 v1 协议), 随机水平/垂直翻转, label smoothing 0.05,
      AdamW 3e-4 wd 0.01, cosine, 20 epochs, bs 24, AMP fp16, 按 val top1 存最优.

用法(在 data/ 目录):
    python train_tiny_gate.py              # v2: 640 输入 -> runs/gate/v2
    python train_tiny_gate.py --size 160   # v3: 160 输入(C++ 4x4 box 对齐) -> runs/gate/v3
    python train_tiny_gate.py --v4         # v4: 160 抽取式(C++ 单像素读对齐) -> runs/gate/v4
产物: runs/gate/{v2|v3|v4}/{best.pt, best.onnx, summary.json}
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GATE_ROOT = Path(__file__).parent / "gate"
OUT_ROOT = Path(__file__).parent / "runs" / "gate" / "v2"
SEED = 42
EPOCHS = 20
BATCH = 24
LR = 3e-4
WD = 0.01
WORKERS = 4
NUM_CLASSES = 3  # 0=empty, 1=simple, 2=complex
GATE_CLASSES = ["empty", "simple", "complex"]


class TinyGate(nn.Module):
    """185k 级微 CNN. v2: 640 输入 stem s4 -> 10x10x512; v3: 160 输入 stem s2 -> 5x5x512 -> GAP -> 3 logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, stem_stride: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=stem_stride, padding=3,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            self._dw_block(32, 64),
            self._dw_block(64, 128),
            self._dw_block(128, 256),
            self._dw_block(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)

    @staticmethod
    def _dw_block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=2, padding=1,
                      groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)  # raw logits, softmax 由 C++ 端完成


class GateCrops(Dataset):
    """读 gate_{split}.txt 清单 (相对 gate 根的路径 + 类别), 预处理与 C++ 逐位一致."""

    def __init__(self, list_file: Path, train: bool, size: int = 640,
                 decim: bool = False):
        self.root = GATE_ROOT
        self.train = train
        self.size = size
        self.decim = decim
        self.items: list[tuple[str, int]] = []
        for ln in list_file.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            p, c = ln.rsplit(" ", 1)
            self.items.append((p, int(c)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        p, c = self.items[i]
        img = Image.open(self.root / p).convert("RGB")
        if self.decim:
            # v4(160): 抽取式下采样 = C++ 端每输出像素单次读 (pixel 4gy,4gx), 逐位对齐
            assert img.size == (640, 640), f"decim 需 640 输入, got {img.size}"
            arr = np.asarray(img)[0::4, 0::4, :]
            x = torch.from_numpy(arr.astype(np.float32) / 255.0)
        else:
            if img.size != (self.size, self.size):
                # v3(160): BOX 滤波 = C++ 端 4x4 块平均下采样, 逐位对齐
                filt = Image.BOX if self.size < img.size[0] else Image.BILINEAR
                img = img.resize((self.size, self.size), filt)
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).contiguous()  # HWC -> CHW, RGB/255
        if self.train:
            if random.random() < 0.5:
                x = torch.flip(x, [2])  # 水平翻转
            if random.random() < 0.5:
                x = torch.flip(x, [1])  # 垂直翻转
        return x, c


def evaluate(model: nn.Module, loader: DataLoader, device) -> dict:
    model.eval()
    cm = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]  # cm[gt][pred]
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
    # 误跳率: 非 empty 被判为 empty (直接关联漏检风险)
    non_empty = gt_cnt[1] + gt_cnt[2]
    false_skip = (cm[1][0] + cm[2][0]) / non_empty if non_empty else float("nan")
    return {"top1": top1, "loss": loss_sum / max(1, n),
            "cm": cm, "recall": recall, "false_skip_rate": false_skip}


def export_onnx(model: nn.Module, path: Path, size: int = 640) -> dict:
    """导出 raw-logits ONNX (images 1x3x{size}x{size}, opset 13) 并做 onnxruntime 校验."""
    model = model.to("cpu").eval()
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(model, dummy, str(path), input_names=["images"],
                      output_names=["output0"], opset_version=13)
    info = {"onnx": str(path), "input": f"images [1,3,{size},{size}] float32",
            "output": "output0 [1,3] raw logits", "opset": 13}
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path),
                                    providers=["CPUExecutionProvider"])
        x = torch.rand(1, 3, size, size)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"images": x.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        info["ort_max_abs_diff"] = diff
        assert diff < 1e-3, f"onnxruntime mismatch: {diff}"
    except ImportError:
        info["ort_check"] = "skipped (onnxruntime not installed)"
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-only", action="store_true",
                    help="跳过训练, 只加载 best.pt 导出 ONNX + summary")
    ap.add_argument("--size", type=int, default=640, choices=[640, 160],
                    help="gate 输入分辨率: 640=v2, 160=v3 (C++ 4x4 box 下采样对齐)")
    ap.add_argument("--v4", action="store_true",
                    help="v4: 160 抽取式下采样 (numpy [0::4,0::4] <-> C++ 单像素读), 输出 runs/gate/v4")
    args = ap.parse_args()
    if args.v4:
        args.size = 160
    global OUT_ROOT
    if args.v4:
        OUT_ROOT = OUT_ROOT.parent / "v4"
    elif args.size != 640:
        OUT_ROOT = OUT_ROOT.parent / "v3"
    if args.export_only:
        model = TinyGate(stem_stride=2 if args.size == 160 else 4).to("cpu")
        model.load_state_dict(
            torch.load(OUT_ROOT / "best.pt", map_location="cpu",
                       weights_only=True))
        onnx_info = export_onnx(model, OUT_ROOT / "best.onnx", args.size)
        print(f"[export] {json.dumps(onnx_info, ensure_ascii=False)}")
        (OUT_ROOT / "summary.json").write_text(json.dumps({
            "arch": f"TinyGate 187,587 params, raw logits, "
                    f"images [1,3,{args.size},{args.size}] RGB/255",
            "export_only": True,
            "onnx": onnx_info,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[summary] {OUT_ROOT / 'summary.json'}")
        return
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] torch={torch.__version__} device={device}")

    train_ds = GateCrops(GATE_ROOT / "gate_train.txt", train=True,
                         size=args.size, decim=args.v4)
    val_ds = GateCrops(GATE_ROOT / "gate_val.txt", train=False, size=args.size,
                       decim=args.v4)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} (gate_train/val.txt)")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=WORKERS, pin_memory=True,
                              drop_last=False, persistent_workers=WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = TinyGate(stem_stride=2 if args.size == 160 else 4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] TinyGate size={args.size} params={n_params} "
          f"({n_params / 1e6:.3f}M, 目标 <0.2M)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=EPOCHS)
    scaler = torch.amp.GradScaler(device.type)

    best_top1, best_metrics = -1.0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device,
                                                          non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * y.numel()
            n_seen += y.numel()
        scheduler.step()
        m = evaluate(model, val_loader, device)
        tag = ""
        if m["top1"] > best_top1:
            best_top1, best_metrics = m["top1"], m
            torch.save(model.state_dict(), OUT_ROOT / "best.pt")
            tag = "  <- best"
        print(f"[ep {ep:02d}/{EPOCHS}] train_loss={loss_sum / n_seen:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} val_top1={m['top1']:.4f} "
              f"empty_recall={m['recall'][0]:.4f} "
              f"false_skip={m['false_skip_rate']:.4f}{tag}")
    print(f"[train done] {time.time() - t0:.1f}s best_val_top1={best_top1:.4f}")

    # ---- 导出最优权重为 raw-logits ONNX 并校验 ----
    model.load_state_dict(torch.load(OUT_ROOT / "best.pt",
                                     map_location=device,
                                     weights_only=True))
    onnx_info = export_onnx(model, OUT_ROOT / "best.onnx", args.size)
    print(f"[export] {json.dumps(onnx_info, ensure_ascii=False)}")

    summary = {
        "purpose": "v2 轻量 gate: 延迟 1-2ms, 盈亏平衡跳过率降至 ~10%",
        "arch": "stem Conv3-32 k7 s4 + 4x[DW k3 s2 + PW] (32-64-128-256-512) "
                "+ GAP + Linear512-3, raw logits",
        "params": n_params,
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD,
        "label_smoothing": 0.05, "augment": "hflip/vflip",
        "preprocess": f"RGB/255, resize {args.size} (与 seaskip_eval.cpp 逐位一致)",
        "best_val_top1": best_top1,
        "best_val_metrics": best_metrics,
        "classes": GATE_CLASSES,
        "onnx": onnx_info,
        "next": "pscp 上板 -> ATC 转 gate_tiny.om -> gate-only 延迟测试 "
                "-> full/skip 复跑重算 M3/M6/经济性",
    }
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()