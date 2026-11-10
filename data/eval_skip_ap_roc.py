"""SeaSkip 论文补强分析(本地, 无需板端):
1) skip 模式端到端 mAP50 (相对 full 的精度代价, 含误跳 GT 船损失)
2) gate 阈值 ROC: p_empty 阈值跳过策略 vs argmax 策略, 找零误跳阈值点
3) 无 gate 基线对比 + 稀疏度收益模拟(逐 patch 真实实测耗时混合重放)

用法(在 data/ 目录下执行):
    python eval_skip_ap_roc.py --dataset hrsid --split test \
        --full board/results_hrsid_full.csv --skip board/results_hrsid_skip.csv

输入: 板端 full/skip CSV(boxes 列: x1,y1,x2,y2,conf|...) + patch 流清单 GT
输出: runs/board_eval/skip_ap_roc_<ds>_<split>.json

说明:
  - GT 框为原始图坐标(未裁剪到窗口内), 板端检出框已裁剪到 patch 内,
    跨窗目标 IoU 会被低估, 故绝对 mAP 可能略低于 PC 参考, 但 full/skip
    两模式用同一 GT, 其差值(精度代价)不受影响。
  - 稀疏度模拟: 用板端逐 patch 实测耗时与真实 gate 决策, 分层抽样混合
    empty/ship patch 构造不同空占比的流, 属真实测量的重放(仅混合是合成)。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from common import BOARD_ROOT, utf8_console


def parse_boxes(s: str) -> list[list[float]]:
    if not s or s in ("-", ""):
        return []
    out = []
    for part in s.split("|"):
        v = [float(x) for x in part.split(",")]
        if len(v) == 5:
            out.append(v)
    return out


def load_csv_boxes(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[int(r["idx"])] = {
                "gate_pred": int(r["gate_pred"]),
                "p_empty": float(r["p_empty"]),
                "ran_det": int(r["ran_det"]),
                "prep_ms": float(r["prep_ms"]),
                "gate_ms": float(r["gate_ms"]),
                "det_ms": float(r["det_ms"]) if r.get("det_ms") not in (None, "", "-") else 0.0,
                "decode_ms": float(r["decode_ms"]) if r.get("decode_ms") not in (None, "", "-") else 0.0,
                "gprep_ms": float(r["gprep_ms"]) if r.get("gprep_ms") not in (None, "", "-") else 0.0,
                "boxes": parse_boxes(r["boxes"]),
            }
    return rows


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def ap50(patches: list[dict], dets: dict[int, list[list[float]]],
         iou_thr: float = 0.5) -> tuple[float, int, int]:
    """单类 AP50(VOC all-point). 返回 (ap, n_dets, n_matched).

    GT 使用 gt_boxes_win(已平移到窗口坐标并裁剪到窗口边界, 与板端检出同系).
    """
    all_d: list[tuple[float, int, list[float]]] = []
    for i, p in enumerate(patches):
        for b in dets.get(i, []):
            all_d.append((b[4], i, b))
    all_d.sort(key=lambda t: -t[0])
    n_gt = sum(p["n_boxes"] for p in patches)
    if n_gt == 0:
        return float("nan"), len(all_d), 0
    matched: set[tuple[int, int]] = set()
    tp_flags: list[int] = []
    for _conf, i, b in all_d:
        gts = patches[i]["gt_boxes_win"]
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if (i, j) in matched:
                continue
            v = iou(b, g)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            matched.add((i, best_j))
            tp_flags.append(1)
        else:
            tp_flags.append(0)
    m = len(tp_flags)
    if m == 0:
        return 0.0, 0, 0
    ctp = 0
    precs, recs = [], []
    for k in range(m):
        ctp += tp_flags[k]
        precs.append(ctp / (k + 1))
        recs.append(ctp / n_gt)
    for k in range(m - 2, -1, -1):
        precs[k] = max(precs[k], precs[k + 1])
    ap, prev_r = 0.0, 0.0
    for k in range(m):
        if recs[k] > prev_r:
            ap += (recs[k] - prev_r) * precs[k]
            prev_r = recs[k]
    return ap, m, len(matched)


def main() -> None:
    utf8_console()
    ap_ = argparse.ArgumentParser(description="SeaSkip skip-mAP/阈值ROC/稀疏度模拟")
    ap_.add_argument("--dataset", default="hrsid", choices=["ssdd", "hrsid"])
    ap_.add_argument("--split", default="test")
    ap_.add_argument("--full", required=True, type=Path)
    ap_.add_argument("--skip", type=Path)
    ap_.add_argument("--iou", type=float, default=0.5)
    ap_.add_argument("--ratios", nargs="+", type=float,
                     default=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
    ap_.add_argument("--seed", type=int, default=42)
    ap_.add_argument("--out", default=str(Path("runs") / "board_eval"))
    args = ap_.parse_args()

    stream = json.loads(
        (BOARD_ROOT / f"patch_stream_{args.dataset}_{args.split}.json")
        .read_text(encoding="utf-8"))
    patches = stream["patches"]
    n = len(patches)
    # GT 框平移到窗口坐标并裁剪到窗口边界(与板端检出同系; 跨窗目标取窗口内可见部分)
    for p in patches:
        x0, y0, x1, y1 = p["window"]
        w, h = x1 - x0, y1 - y0
        p["gt_boxes_win"] = [
            [max(0.0, b[0] - x0), max(0.0, b[1] - y0),
             min(float(w), b[2] - x0), min(float(h), b[3] - y0)]
            for b in p["gt_boxes_abs"]
        ]
    full = load_csv_boxes(args.full)
    assert set(full) == set(range(n)), "full CSV idx 与清单不一致"
    with open(args.full, newline="", encoding="utf-8") as f:
        v3 = "decode_ms" in next(csv.reader(f))  # v3 评估器分段计时列
    skip = load_csv_boxes(args.skip) if args.skip and Path(args.skip).exists() else None
    gt_empty = [p["n_boxes"] == 0 for p in patches]
    n_empty = sum(gt_empty)

    # ---------- 1) skip 模式端到端 mAP50 ----------
    print("===== 1) 端到端 mAP50: full vs skip =====")
    dets_full = {i: full[i]["boxes"] for i in range(n)}
    ap_full, nd_full, nm_full = ap50(patches, dets_full, args.iou)
    print(f"full 模式: AP50={ap_full:.4f}  dets={nd_full}  matched={nm_full}"
          f"  GT_ships={sum(p['n_boxes'] for p in patches)}")
    ap_skip, nd_skip, nm_skip = (float("nan"), 0, 0)
    if skip is not None:
        dets_skip = {i: skip[i]["boxes"] for i in range(n)}
        ap_skip, nd_skip, nm_skip = ap50(patches, dets_skip, args.iou)
        d_drop = ap_full - ap_skip
        print(f"skip 模式: AP50={ap_skip:.4f}  dets={nd_skip}  matched={nm_skip}")
        print(f"精度代价: ΔAP50={d_drop:+.4f}  检出框损失 {nd_full - nd_skip}"
              f"  GT 匹配损失 {nm_full - nm_skip}")

    # ---------- 2) gate 阈值 ROC ----------
    print("\n===== 2) gate 阈值 ROC (策略: p_empty>=t 则跳过) =====")
    # argmax 策略(板端现状): gate_pred==0 跳过
    am_sk = sum(1 for i in range(n) if full[i]["gate_pred"] == 0)
    am_ts = sum(1 for i in range(n) if full[i]["gate_pred"] == 0 and gt_empty[i])
    am_fs = am_sk - am_ts
    am_gt_lost = sum(patches[i]["n_boxes"] for i in range(n)
                     if full[i]["gate_pred"] == 0 and not gt_empty[i])
    am_det_lost = sum(len(full[i]["boxes"]) for i in range(n)
                      if full[i]["gate_pred"] == 0 and not gt_empty[i])
    print(f"[argmax 现状] skip_rate={am_sk / n:.4f} true={am_ts} false={am_fs} "
          f"GT_船损={am_gt_lost} 检出损={am_det_lost}")
    roc: list[dict] = []
    for t in (0.34, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.5, 0.6, 0.9):
        sk = ts = fs = 0
        gt_lost = det_lost = 0
        for i in range(n):
            if full[i]["p_empty"] >= t:
                sk += 1
                if gt_empty[i]:
                    ts += 1
                else:
                    fs += 1
                    gt_lost += patches[i]["n_boxes"]
                    det_lost += len(full[i]["boxes"])
        roc.append({"thr": t, "skip_rate": round(sk / n, 4), "skipped": sk,
                    "true_skips": ts, "false_skips": fs,
                    "gt_ships_lost": gt_lost, "dets_lost": det_lost})
        print(f"t={t:<5} skip_rate={sk / n:6.2%} true={ts:<5} false={fs:<4} "
              f"GT_船损={gt_lost:<4} 检出损={det_lost}")
    safe = [r for r in roc if r["false_skips"] == 0]
    safe_pt = max(safe, key=lambda r: r["skip_rate"]) if safe else None
    if safe_pt:
        print(f"零误跳推荐点: t={safe_pt['thr']} skip_rate="
              f"{safe_pt['skip_rate']:.2%} (false_skips=0)")
    else:
        print("阈值扫描内无零误跳点, 需更高阈值")

    # ---------- 3) 基线对比 + 稀疏度收益模拟 ----------
    print("\n===== 3) 基线对比与稀疏度模拟(逐 patch 实测重放) =====")
    prep_m = statistics.mean(full[i]["prep_ms"] for i in range(n))
    gate_m = statistics.mean(full[i]["gate_ms"] for i in range(n))
    det_m = statistics.mean(full[i]["det_ms"] for i in range(n))
    if v3:
        # v3 分段: decode/gprep 逐 patch 必付; prep/det 仅 det 需执行时支付
        dec_m = statistics.mean(full[i]["decode_ms"] for i in range(n))
        gpr_m = statistics.mean(full[i]["gprep_ms"] for i in range(n))
        full_pipe = dec_m + gpr_m + gate_m + prep_m + det_m
        base_pipe = dec_m + prep_m + det_m  # 无 gate 直接检测
        be = (gpr_m + gate_m) / (prep_m + det_m)
        print(f"均值: decode={dec_m:.2f} gprep={gpr_m:.2f} gate={gate_m:.2f} "
              f"prep={prep_m:.2f} det={det_m:.2f}ms (v3 分段)")
    else:
        full_pipe = prep_m + gate_m + det_m
        base_pipe = prep_m + det_m  # 无 gate 直接检测
        be = gate_m / det_m
        print(f"均值: prep={prep_m:.2f} gate={gate_m:.2f} det={det_m:.2f}ms")
    print(f"full 流水线={full_pipe:.2f}ms | 无 gate 基线={base_pipe:.2f}ms")
    print(f"盈亏平衡跳过率(SeaSkip 优于基线): r* = 门控开销/可跳成本 = {be:.3f}")

    def replay(idxs: list[int]) -> dict:
        # skip 决策来自真实 gate 输出; v1/v2: prep/gate 必付仅 det 可跳;
        # v3: decode/gprep/gate 必付, prep+det 在跳过 patch 上均免
        nf = len(idxs)
        t_full = t_skip = t_base = 0.0
        nsk = gt_lost = 0
        for i in idxs:
            r = full[i]
            if v3:
                t_full += (r["decode_ms"] + r["gprep_ms"] + r["gate_ms"]
                           + r["prep_ms"] + r["det_ms"])
                t_base += r["decode_ms"] + r["prep_ms"] + r["det_ms"]
                t_skip += r["decode_ms"] + r["gprep_ms"] + r["gate_ms"]
                run_cost = r["prep_ms"] + r["det_ms"]
            else:
                t_full += r["prep_ms"] + r["gate_ms"] + r["det_ms"]
                t_base += r["prep_ms"] + r["det_ms"]
                t_skip += r["prep_ms"] + r["gate_ms"]
                run_cost = r["det_ms"]
            if r["gate_pred"] == 0:  # skip 决策(真实 gate 输出)
                nsk += 1
                if not gt_empty[i]:
                    gt_lost += patches[i]["n_boxes"]
            else:
                t_skip += run_cost
        return {"n": nf, "skip_rate": round(nsk / nf, 4),
                "full_ms": round(t_full / nf, 3),
                "seaskip_ms": round(t_skip / nf, 3),
                "baseline_ms": round(t_base / nf, 3),
                "save_vs_full": round(1 - (t_skip / t_full), 4),
                "save_vs_base": round(1 - (t_skip / t_base), 4),
                "gt_ships_lost": gt_lost}

    empty_idx = [i for i in range(n) if gt_empty[i]]
    ship_idx = [i for i in range(n) if not gt_empty[i]]
    sim_rows: list[dict] = []
    sim_rows.append({"target_ratio": round(n_empty / n, 4),
                     "note": "实测 test 流(无抽样)", **replay(range(n))})
    for ratio in args.ratios:
        k_ship = min(len(ship_idx),
                     max(1, round(len(empty_idx) * (1 - ratio) / ratio)))
        rng = random.Random(args.seed + int(ratio * 1000))
        ships = rng.sample(ship_idx, k_ship)
        sim_rows.append({"target_ratio": ratio,
                         "note": f"全部 {len(empty_idx)} 空 patch + {k_ship} 船 patch 抽样",
                         **replay(sorted(empty_idx + ships))})
    print(f"{'空占比':>8} {'skip率':>8} {'full':>8} {'SeaSkip':>8} {'基线':>8} "
          f"{'省/full':>8} {'省/基线':>9} {'GT船损':>6}")
    for r in sim_rows:
        print(f"{r['target_ratio']:>8.2%} {r['skip_rate']:>8.2%} "
              f"{r['full_ms']:>8.2f} {r['seaskip_ms']:>8.2f} {r['baseline_ms']:>8.2f} "
              f"{r['save_vs_full']:>8.2%} {r['save_vs_base']:>9.2%} "
              f"{r['gt_ships_lost']:>6}")

    report = {
        "dataset": f"{args.dataset}/{args.split}", "n_patches": n,
        "n_gt_empty": n_empty,
        "ap50": {"full": round(ap_full, 4), "skip": round(ap_skip, 4),
                 "delta": round(ap_full - ap_skip, 4),
                 "n_dets": {"full": nd_full, "skip": nd_skip},
                 "n_matched": {"full": nm_full, "skip": nm_skip},
                 "note": "GT 框未裁剪至窗口, 跨窗目标 IoU 低估, 绝对值偏保守; "
                         "full/skip 差值不受影响"},
        "argmax_policy": {"skip_rate": round(am_sk / n, 4), "skipped": am_sk,
                           "true_skips": am_ts, "false_skips": am_fs,
                           "gt_ships_lost": am_gt_lost,
                           "dets_lost": am_det_lost},
        "threshold_roc": roc,
        "zero_risk_point": safe_pt,
        "latency_mean_ms": {
            **({"decode": round(dec_m, 3), "gprep": round(gpr_m, 3)} if v3 else {}),
            "prep": round(prep_m, 3), "gate": round(gate_m, 3),
            "det": round(det_m, 3), "full_pipe": round(full_pipe, 3),
            "baseline_no_gate": round(base_pipe, 3)},
        "break_even_empty_ratio": round(be, 4),
        "sparsity_simulation": sim_rows,
        "sim_note": "逐 patch 板端实测耗时与真实 gate 决策重放; "
                    "仅 empty/ship 分层混合为构造(种子抽样), 无任何合成测量",
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"skip_ap_roc_{args.dataset}_{args.split}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[report] {out_json}")


if __name__ == "__main__":
    main()