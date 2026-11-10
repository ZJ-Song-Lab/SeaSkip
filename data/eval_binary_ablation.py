"""B7: binary gate vs 3-class gate 在 HRSID 测试流上的 PC 推理对比.

1. 校验: v4 3-class PC 推理决策须与板端 results_v4_hrsid_gate.csv 的 gate_pred 逐位一致
   (验证 PC 推理协议有效, 才可用同一协议评 binary).
2. 评估: v4bin binary gate (empty vs non-empty) PC 推理 -> skip rate / false skips /
   ships on false skips / p_empty 阈值 ROC (tau=0.6).
输出: runs/board_eval/v4/binary_ablation.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_tiny_gate import TinyGate

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
PATCH_DIR = ROOT / "board" / "patches" / "hrsid_test"
OUT = ROOT / "runs" / "board_eval" / "v4" / "binary_ablation.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stream = json.loads((ROOT / "board" / "patch_stream_hrsid_test.json").read_text(encoding="utf-8"))
patches = stream["patches"]
board_pred = {}
with open(ROOT / "board" / "results_v4_hrsid_gate.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        board_pred[int(row["idx"])] = int(row["gate_pred"])


def batch_infer(ckpt: Path, num_classes: int):
    model = TinyGate(num_classes=num_classes, stem_stride=2).to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    preds, pes = [], []
    B = 64
    buf = []
    t0 = time.time()
    for i, p in enumerate(patches):
        img = Image.open(PATCH_DIR / f"{p['idx'] // 4:04d}_{p['idx'] % 4:04d}.jpg").convert("RGB")
        arr = np.asarray(img)[0::4, 0::4, :].astype(np.float32) / 255.0
        buf.append(torch.from_numpy(arr).permute(2, 0, 1))
        if len(buf) == B or i == len(patches) - 1:
            x = torch.stack(buf).to(DEVICE)
            with torch.no_grad():
                prob = F.softmax(model(x), dim=1)
            preds.extend(prob.argmax(1).tolist())
            pes.extend(prob[:, 0].tolist())
            buf = []
        if (i + 1) % 2000 == 0:
            print(f"[infer] {i + 1}/{len(patches)} ({time.time() - t0:.0f}s)")
    return preds, np.array(pes)


def metrics(preds, pes, tau=None):
    skip = [1 if (pe >= tau if tau else pr == 0) else 0
            for pr, pe in zip(preds, pes)]
    tn = fn = 0
    fs_ships = 0
    fs_min_sides = []
    gt_empty = 0
    for p, s in zip(patches, skip):
        has = p["n_boxes"] > 0
        if not has:
            gt_empty += 1
        if s:
            if has:
                fn += 1
                fs_ships += p["n_boxes"]
                fs_min_sides.extend(min(b[2] - b[0], b[3] - b[1]) for b in p["gt_boxes_abs"])
            else:
                tn += 1
    n = len(patches)
    return {
        "skip_rate": (tn + fn) / n,
        "true_skips": tn, "false_skips": fn,
        "gt_ships_on_false_skips": fs_ships,
        "empty_recall": tn / max(1, gt_empty),
        "fs_min_side_median": float(np.median(fs_min_sides)) if fs_min_sides else None,
        "fs_min_side_lt32_frac": (sum(1 for s in fs_min_sides if s < 32)
                                   / max(1, len(fs_min_sides))),
    }


# ---- 1. v4 3-class PC 推理, 与板端对齐校验 ----
pred3, pe3 = batch_infer(ROOT / "runs" / "gate" / "v4" / "best.pt", 3)
diffs = sum(1 for p, (i, bp) in zip(pred3, sorted(board_pred.items())) if p != bp)
print(f"[parity] v4 3-class PC vs board gate_pred diffs = {diffs}")
m3_arg = metrics(pred3, pe3)
m3_tau = metrics(pred3, pe3, tau=0.6)

# ---- 2. binary gate ----
predb, peb = batch_infer(ROOT / "runs" / "gate" / "v4bin" / "best.pt", 2)
mb_arg = metrics(predb, peb)
mb_tau = metrics(predb, peb, tau=0.6)

result = {
    "parity_check": {"v4_3class_pc_vs_board_diffs": diffs,
                      "note": "0 diffs 则 PC 推理协议与板端逐位一致"},
    "three_class_v4": {"argmax": m3_arg, "tau06": m3_tau},
    "binary_v4bin": {"argmax": mb_arg, "tau06": mb_tau},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"[saved] {OUT}")