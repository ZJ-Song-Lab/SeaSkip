"""D13: v4 gate 失败模式分析 — 230 个误跳 patch 的结构与图像统计.

数据: board/patch_stream_hrsid_test.json (GT) + board/results_v4_hrsid_gate.csv (v4 板端 gate 输出).
输出: runs/board_eval/v4/failure_analysis.json + 控制台摘要.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent                   # data/
PATCH_DIR = ROOT / "board" / "patches" / "hrsid_test"
OUT = ROOT / "runs" / "board_eval" / "v4" / "failure_analysis.json"

stream = json.loads((ROOT / "board" / "patch_stream_hrsid_test.json").read_text(encoding="utf-8"))
patches = stream["patches"]

import csv
gate = {}
with open(ROOT / "board" / "results_v4_hrsid_gate.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        gate[int(row["idx"])] = (int(row["gate_pred"]), float(row["p_empty"]))

assert len(gate) == len(patches) == 7848

def ship_stats(gt_boxes):
    """gt_boxes: 绝对坐标窗口内 [x1,y1,x2,y2] -> 个数, 最小边, 面积占比."""
    n = len(gt_boxes)
    if n == 0:
        return n, None, 0.0
    sides = [min(b[2] - b[0], b[3] - b[1]) for b in gt_boxes]
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in gt_boxes)
    return n, float(min(sides)), area / (640 * 640)

# ---- 分类: 按 GT(空/有船) x gate_pred ----
groups = {"true_skip": [], "false_skip": [], "true_run": [], "false_run": []}
for p in patches:
    pred, pe = gate[p["idx"]]
    has_ship = p["n_boxes"] > 0
    key = ("true" if (pred == 0) == (not has_ship) else "false") + ("_skip" if pred == 0 else "_run")
    w = p["window"]
    entry = {"idx": p["idx"], "p_empty": pe, "n_boxes": p["n_boxes"],
             "window": w, "scene": p["scene_id"],
             "min_side": None, "ship_area_frac": None, "border": None}
    if has_ship:
        n, ms, af = ship_stats(p["gt_boxes_abs"])
        entry.update(n_boxes=n, min_side=ms, ship_area_frac=af)
        # 800x800 场景 stride 640 -> 4 窗, 判断窗是否贴场景边(船被裁剪风险)
        entry["border"] = (w[0] > 0 or w[1] > 0)  # 非左上窗 = 贴右/下边
    groups[key].append(entry)

fs, ts, tr = groups["false_skip"], groups["true_skip"], groups["true_run"]
print(f"[groups] true_skip={len(ts)} false_skip={len(fs)} true_run={len(tr)} false_run={len(groups['false_run'])}")

# ---- 结构统计 ----
def quant(xs, q):
    return float(np.quantile(xs, q)) if xs else None

fs_ships = [e["n_boxes"] for e in fs]
tr_ships = [e["n_boxes"] for e in tr]
fs_min = [e["min_side"] for e in fs if e["min_side"] is not None]
tr_min = [e["min_side"] for e in tr if e["min_side"] is not None]
fs_area = [e["ship_area_frac"] for e in fs if e["ship_area_frac"] is not None]
tr_area = [e["ship_area_frac"] for e in tr if e["ship_area_frac"] is not None]

# 小船占比 (<32px 即 complex 判据)
small_fs = sum(1 for m in fs_min if m < 32) / max(1, len(fs_min))
small_tr = sum(1 for m in tr_min if m < 32) / max(1, len(tr_min))

# 窗位置(贴边 vs 左上)
border_fs = sum(1 for e in fs if e["border"]) / max(1, len(fs))
border_tr = sum(1 for e in tr if e["border"]) / max(1, len(tr))

# p_empty 分布
fs_pe = [e["p_empty"] for e in fs]

# τ=0.6 阈值下还剩多少误跳 (p_empty < 0.4 才跳 -> 即 p_nonempty>0.6 才跑)
fs_tau06 = sum(1 for e in fs if e["p_empty"] >= 0.6)

# GT 类别分布 (误跳里 simple vs complex 需按挖掘协议: 船数>5 或 最小边<32)
def gt_diff(e):
    if e["n_boxes"] > 5 or (e["min_side"] is not None and e["min_side"] < 32):
        return "complex"
    return "simple"
fs_cls = Counter(gt_diff(e) for e in fs)
tr_cls = Counter(gt_diff(e) for e in tr)

# ---- 图像统计: 误跳 vs 真跳 (亮度/对比度/亮像素占比) ----
def img_stats(idx):
    img = np.asarray(Image.open(PATCH_DIR / f"{idx // 4:04d}_{idx % 4:04d}.jpg").convert("L"), dtype=np.float32)
    return float(img.mean()), float(img.std()), float((img > 200).mean())

rng = np.random.default_rng(42)
ts_sample = rng.choice(len(ts), size=min(300, len(ts)), replace=False)
fs_img, ts_img = [], []
for e in fs:
    fs_img.append(img_stats(e["idx"]))
for j in ts_sample:
    ts_img.append(img_stats(ts[j]["idx"]))
fs_img, ts_img = np.array(fs_img), np.array(ts_img)

summary = {
    "n": {"true_skip": len(ts), "false_skip": len(fs), "true_run": len(tr),
          "false_run": len(groups["false_run"])},
    "false_skip_structure": {
        "ships_per_patch_mean": float(np.mean(fs_ships)),
        "ships_per_patch_median": float(np.median(fs_ships)),
        "min_ship_side_px": {"median": quant(fs_min, 0.5), "q25": quant(fs_min, 0.25),
                              "q75": quant(fs_min, 0.75), "min": min(fs_min) if fs_min else None},
        "pct_min_side_lt32": small_fs,
        "ship_area_frac": {"median": quant(fs_area, 0.5), "q75": quant(fs_area, 0.75),
                            "max": max(fs_area) if fs_area else None},
        "pct_border_window": border_fs,
        "gt_class": dict(fs_cls),
        "p_empty": {"median": quant(fs_pe, 0.5), "q25": quant(fs_pe, 0.25), "q75": quant(fs_pe, 0.75),
                     "n_with_pe_ge_0.6 (kept at tau=0.6)": fs_tau06},
    },
    "true_run_reference": {
        "ships_per_patch_mean": float(np.mean(tr_ships)),
        "min_ship_side_px": {"median": quant(tr_min, 0.5), "q25": quant(tr_min, 0.25)},
        "pct_min_side_lt32": small_tr,
        "ship_area_frac_median": quant(tr_area, 0.5),
        "pct_border_window": border_tr,
        "gt_class": dict(tr_cls),
    },
    "image_stats_mean_std": {
        "false_skip_brightness": [float(fs_img[:, 0].mean()), float(fs_img[:, 0].std())],
        "true_skip_brightness": [float(ts_img[:, 0].mean()), float(ts_img[:, 0].std())],
        "false_skip_contrast": [float(fs_img[:, 1].mean()), float(fs_img[:, 1].std())],
        "true_skip_contrast": [float(ts_img[:, 1].mean()), float(ts_img[:, 1].std())],
        "false_skip_bright_px_frac": [float(fs_img[:, 2].mean()), float(fs_img[:, 2].std())],
        "true_skip_bright_px_frac": [float(ts_img[:, 2].mean()), float(ts_img[:, 2].std())],
    },
    "false_skip_list": fs,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "false_skip_list"},
                 ensure_ascii=False, indent=2))
print(f"[saved] {OUT}")