"""C12: CA-CFAR 预筛基线 vs 学习门 (SeaSkip v4) on HRSID test 流.

协议:
  1. 调参: gate_train.txt 的 HRSID 子集 (空 vs 非空), 网格 Pfa x min_pix (guard=4, train_half=12).
  2. 评估: 7848 patch 测试流, GT 来自 patch_stream_hrsid_test.json.
     skip = CFAR 无检出 (det 像素数 < min_pix). 指标: skip_rate / false_skips /
     gt_ships_on_false_skips / ships lost 尺寸分布, 对齐 gate 评估口径.
  3. 计时: PC 端逐 patch CFAR 纯计算耗时 (numpy 积分图实现, 板端 C++ 另测).
输出: runs/board_eval/cfar/cfar_hrsid_test.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent                   # data/
GATE_ROOT = ROOT / "gate"
PATCH_DIR = ROOT / "board" / "patches" / "hrsid_test"
OUT_DIR = ROOT / "runs" / "board_eval" / "cfar"
OUT = OUT_DIR / "cfar_hrsid_test.json"

GUARD = 4          # 保护单元半宽
TRAIN_HALF = 12    # 训练单元半宽 (窗 25x25, 训练环 25^2-9^2=544 cells)
PFAS = [1e-3, 1e-4, 1e-5]
MIN_PIX = [1, 4, 16]
N_TRAIN_CELLS = (2 * (GUARD + TRAIN_HALF) + 1) ** 2 - (2 * GUARD + 1) ** 2


def cfar_detect(img: np.ndarray, alpha: float) -> np.ndarray:
    """CA-CFAR: det = pixel > alpha * local_clutter_mean (训练环均值, 积分图).
    img: (H,W) float32 灰度. 边缘 HALF 宽度内不检测."""
    h, w = img.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = img.cumsum(0).cumsum(1)

    t = GUARD + TRAIN_HALF
    if w - 2 * t < 1 or h - 2 * t < 1:
        return np.zeros((max(0, h - 2 * GUARD), max(0, w - 2 * GUARD)), dtype=bool)
    # 检测中心区: 外窗 (2t+1) 完全落入图像
    x0 = np.arange(0, w - 2 * t)
    y0 = np.arange(0, h - 2 * t)
    x1, y1 = x0 + 2 * t + 1, y0 + 2 * t + 1
    outer_sum = ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)]
    # 内框和 (2g+1), 中心对齐: 内窗左上 = 外窗左上 + (t-g)
    g = GUARD
    x0i, y0i = x0 + (t - g), y0 + (t - g)
    x1i, y1i = x0i + 2 * g + 1, y0i + 2 * g + 1
    inner_sum = ii[np.ix_(y1i, x1i)] - ii[np.ix_(y0i, x1i)] - ii[np.ix_(y1i, x0i)] + ii[np.ix_(y0i, x0i)]
    clutter = (outer_sum - inner_sum) / N_TRAIN_CELLS
    det_region = img[t:h - t, t:w - t]
    out = np.zeros((h, w), dtype=bool)
    out[t:h - t, t:w - t] = det_region > alpha * clutter
    return out


def alpha_of(pfa: float) -> float:
    return N_TRAIN_CELLS * (pfa ** (-1.0 / N_TRAIN_CELLS) - 1.0)


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


# ---------- 1. 调参 (HRSID 训练子集) ----------
tune_items = []
for ln in (GATE_ROOT / "gate_train.txt").read_text(encoding="utf-8").splitlines():
    p, c = ln.rsplit(" ", 1)
    if "hrsid" in p:
        tune_items.append((p, 1 if int(c) > 0 else 0))
rng = np.random.default_rng(42)
sel = rng.choice(len(tune_items), size=2000, replace=False)
tune_items = [tune_items[i] for i in sel]
print(f"[tune] {len(tune_items)} HRSID train crops "
      f"(empty={sum(1 for _, c in tune_items if c == 0)})")

alphas = {p: alpha_of(p) for p in PFAS}
configs = [(p, mp) for p in PFAS for mp in MIN_PIX]
tune_stats = {cfg: [0, 0, 0, 0] for cfg in configs}  # tp run, fp run, tn skip, fn skip
for p, label in tune_items:
    img = load_gray(GATE_ROOT / p)
    for cfg in configs:
        pfa, mp = cfg
        det = cfar_detect(img, alphas[pfa])
        n_det = int(det.sum())
        run = n_det >= mp
        if label == 1:
            tune_stats[cfg][0 if run else 3] += 1
        else:
            tune_stats[cfg][1 if run else 2] += 1

tune_results = []
for cfg in configs:
    tp, fp, tn, fn = tune_stats[cfg]
    er = tn / max(1, tn + fp)          # empty recall = skip 准确率
    fsr = fn / max(1, fn + tp)         # false skip rate
    tune_results.append({"pfa": cfg[0], "min_pix": cfg[1],
                         "empty_recall": er, "false_skip_rate": fsr,
                         "skip_rate": (tn + fn) / len(tune_items)})
    print(f"[tune] pfa={cfg[0]:.0e} min_pix={cfg[1]:>2d} "
          f"empty_recall={er:.4f} false_skip_rate={fsr:.4f}")

# 选点: false_skip_rate 最低者中 empty_recall 最高 (保守) + 单独报告 Pareto
best = min(tune_results, key=lambda r: (r["false_skip_rate"], -r["empty_recall"]))
print(f"[tune] selected: {best}")

# ---------- 2. 测试流评估 (选定 config + 边界 config 供 Pareto) ----------
stream = json.loads((ROOT / "board" / "patch_stream_hrsid_test.json").read_text(encoding="utf-8"))
patches = stream["patches"]
gate = {}
with open(ROOT / "board" / "results_v4_hrsid_gate.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        gate[int(row["idx"])] = (int(row["gate_pred"]), float(row["p_empty"]))

eval_configs = [best] + [r for r in tune_results
                         if r is not best and r["pfa"] == best["pfa"]][:0]
# 评估全部 9 个 config 在测试流上? 成本可控(CFAR 便宜), 但为诚实只报调参选定的 1 个 + 说明.
# 这里评估选定 config; 同时记录 gate 决策做逐 patch 对照.
stats = {"run": 0, "skip": 0}
tp = fp = tn = fn = 0
fs_ships = 0
fs_min_sides = []
fs_patches = []
pc_cfar_ms = []
pfa, min_pix = best["pfa"], best["min_pix"]
alpha = alphas[pfa]
t_start = time.time()
for i, p in enumerate(patches):
    img = load_gray(PATCH_DIR / f"{p['idx'] // 4:04d}_{p['idx'] % 4:04d}.jpg")
    t0 = time.perf_counter()
    det = cfar_detect(img, alpha)
    n_det = int(det.sum())
    pc_cfar_ms.append(time.perf_counter() - t0)
    run = n_det >= min_pix
    has_ship = p["n_boxes"] > 0
    if run:
        stats["run"] += 1
        if has_ship:
            tp += 1
        else:
            fp += 1
    else:
        stats["skip"] += 1
        if has_ship:
            fn += 1
            fs_ships += p["n_boxes"]
            sides = [min(b[2] - b[0], b[3] - b[1]) for b in p["gt_boxes_abs"]]
            fs_min_sides.extend(sides)
            fs_patches.append(p["idx"])
        else:
            tn += 1
    if (i + 1) % 1000 == 0:
        print(f"[test] {i + 1}/{len(patches)} ({time.time() - t_start:.0f}s)")

n = len(patches)
result = {
    "protocol": {"guard": GUARD, "train_half": TRAIN_HALF,
                 "n_train_cells": N_TRAIN_CELLS, "alpha": alpha,
                 "pfa": pfa, "min_pix": min_pix,
                 "tune_set": "2000 HRSID crops from gate_train.txt (seed 42)",
                 "tune_grid_results": tune_results,
                 "tune_selection": "min false_skip_rate, then max empty_recall"},
    "test_hrsid": {
        "n": n,
        "skip_rate": (tn + fn) / n,
        "true_skips": tn, "false_skips": fn,
        "gt_ships_on_false_skips": fs_ships,
        "empty_recall": tn / max(1, tn + fp),
        "false_run_nonempty_missed": fp,
        "fs_min_side_median": float(np.median(fs_min_sides)) if fs_min_sides else None,
        "fs_min_side_lt32_frac": (sum(1 for s in fs_min_sides if s < 32)
                                   / max(1, len(fs_min_sides))),
    },
    "pc_timing": {"cfar_ms_mean": float(np.mean(pc_cfar_ms)),
                   "cfar_ms_p95": float(np.quantile(pc_cfar_ms, 0.95))},
    "false_skip_patches": fs_patches,
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "false_skip_patches"},
                 ensure_ascii=False, indent=2))
print(f"[saved] {OUT}")