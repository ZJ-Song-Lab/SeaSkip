"""SeaSkip 板端实验 M3/M6 评估: 对照 patch 流清单 GT 评估 gate 跳过质量与延迟收益.

用法(在 data/ 目录下执行):
    python eval_board_m3m6.py --dataset ssdd --full board_results_ssdd_full.csv \
        --skip board_results_ssdd_skip.csv [--gate board_results_ssdd_gate.csv]

输入 CSV 为板端 seaskip_eval 的输出:
    idx,file,gate_pred,p_empty,p_simple,p_complex,ran_det,n_det,boxes,prep_ms,gate_ms,det_ms

产出:
    M3: gate_pred vs GT 三态混淆矩阵; 假跳过(GT 非空却跳过)清单与漏检代价
    M6: prep/gate/det 延迟分解; full vs skip 每 patch 流水线延迟与节省率
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from common import BOARD_ROOT, PROCESSED_ROOT, utf8_console

CLASSES = ["empty", "simple", "complex"]


def gt_label(patch: dict, scene: str | None, k_simple: int, small_px: float) -> int:
    """复现 common.label_patch 的三态规则(窗口内 GT 已在清单中)."""
    n = patch["n_boxes"]
    if n == 0:
        return 0
    boxes = patch["gt_boxes_abs"]
    sides = [min(b[2] - b[0], b[3] - b[1]) for b in boxes]
    if n > k_simple or min(sides) < small_px or scene == "inshore":
        return 2
    return 1


def load_board_csv(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[int(r["idx"])] = {
                "gate_pred": int(r["gate_pred"]),
                "p_empty": float(r["p_empty"]),
                "p_simple": float(r["p_simple"]),
                "p_complex": float(r["p_complex"]),
                "ran_det": int(r["ran_det"]),
                "n_det": int(r["n_det"]),
                "prep_ms": float(r["prep_ms"]),
                "gate_ms": float(r["gate_ms"]),
                "det_ms": float(r["det_ms"]),
                "file": r["file"],
            }
    return rows


def pct(x: float) -> str:
    return f"{x:.1%}"


def main() -> None:
    utf8_console()
    ap = argparse.ArgumentParser(description="SeaSkip M3/M6 板端结果评估")
    ap.add_argument("--dataset", default="ssdd", choices=["ssdd", "hrsid"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--full", required=True, type=Path, help="板端 full 模式 CSV")
    ap.add_argument("--skip", type=Path, help="板端 skip 模式 CSV")
    ap.add_argument("--gate", type=Path, help="板端 gate 模式 CSV (可选)")
    ap.add_argument("--k-simple", type=int, default=5)
    ap.add_argument("--small-px", type=float, default=32.0)
    ap.add_argument("--out", default=str(Path("runs") / "board_eval"),
                    help="评估结果输出目录")
    args = ap.parse_args()

    stream_path = BOARD_ROOT / f"patch_stream_{args.dataset}_{args.split}.json"
    stream = json.loads(stream_path.read_text(encoding="utf-8"))
    patches = stream["patches"]
    meta = json.loads(
        (PROCESSED_ROOT / args.dataset / f"{args.dataset}_meta.json")
        .read_text(encoding="utf-8"))
    scene_of = {e["id"]: e.get("scene") for e in meta["images"]}

    gt = [gt_label(p, scene_of.get(p["scene_id"]), args.k_simple, args.small_px)
          for p in patches]
    n_gt = {c: gt.count(i) for i, c in enumerate(CLASSES)}
    print(f"[stream] {args.dataset}/{args.split}: {len(patches)} patches, "
          f"GT 分布 {n_gt}")

    full = load_board_csv(args.full)
    assert set(full) == set(range(len(patches))), \
        "full CSV idx 与清单 patch 数不一致"
    preds = [full[i]["gate_pred"] for i in range(len(patches))]

    # ---------- M3: gate 跳过质量 ----------
    print("\n===== M3: gate 跳过质量 =====")
    cm = [[0] * 3 for _ in range(3)]  # [gt][pred]
    for i, g in enumerate(gt):
        cm[g][preds[i]] += 1
    print("混淆矩阵 (行=GT, 列=gate_pred):")
    print(f"{'GT\\pred':>10} {'empty':>8} {'simple':>8} {'complex':>8} {'recall':>8}")
    for gi, cname in enumerate(CLASSES):
        row = cm[gi]
        rec = row[gi] / sum(row) if sum(row) else float("nan")
        print(f"{cname:>10} {row[0]:>8} {row[1]:>8} {row[2]:>8} {rec:>8.3f}")
    acc = sum(cm[i][i] for i in range(3)) / len(gt)
    print(f"总体准确率: {acc:.4f}")

    false_skips, true_skips = [], 0
    for i, g in enumerate(gt):
        if preds[i] == 0:
            if g == 0:
                true_skips += 1
            else:
                false_skips.append(i)
    print(f"\ntrue_skip  (GT=empty 且跳过): {true_skips}")
    print(f"false_skip (GT 非空却跳过): {len(false_skips)}")
    if false_skips:
        fs_boxes = sum(patches[i]["n_boxes"] for i in false_skips)
        fs_dets = sum(full[i]["n_det"] for i in false_skips)
        print("  [风险清单] idx, scene, GT类, GT船数, full模式检出数, "
              "gate(empty,simple,complex):")
        for i in false_skips[:20]:
            r = full[i]
            print(f"    {i:>4} {patches[i]['scene_id']} {CLASSES[gt[i]]:>7} "
                  f"GT船={patches[i]['n_boxes']:<3} det={r['n_det']:<3} "
                  f"p=({r['p_empty']:.3f},{r['p_simple']:.3f},{r['p_complex']:.3f})")
        print(f"  若跳过将损失的 GT 船数合计: {fs_boxes}, "
              f"full 模式在这些 patch 的检出框合计: {fs_dets}")
    if n_gt.get("empty", 0):
        print(f"GT empty 中被正确跳过: {true_skips}/{gt.count(0)} "
              f"({pct(true_skips / gt.count(0))})")

    # ---------- M6: 延迟与计算节省 ----------
    print("\n===== M6: 板端延迟与计算节省 =====")
    F = [full[i] for i in range(len(patches))]
    stats = lambda xs: (statistics.mean(xs), statistics.median(xs),
                        sorted(xs)[int(0.9 * (len(xs) - 1))])
    for key in ("prep_ms", "gate_ms", "det_ms"):
        xs = [r[key] for r in F if r[key] > 0 or key == "prep_ms"]
        if not xs:
            print(f"{key:>8}: (no samples)")
            continue
        m, md, p90 = stats(xs)
        print(f"{key:>8}: mean={m:7.2f}ms  median={md:7.2f}ms  p90={p90:7.2f}ms  "
              f"n={len(xs)}")

    prep_m = statistics.mean(r["prep_ms"] for r in F)
    gate_m = statistics.mean(r["gate_ms"] for r in F)
    det_xs = [r["det_ms"] for r in F if r["ran_det"]]
    det_m = statistics.mean(det_xs) if det_xs else 0.0
    full_pipe = prep_m + gate_m + det_m
    print(f"\nfull  流水线/patch: prep+gate+det = {full_pipe:.2f}ms "
          f"(throughput {1000 / full_pipe:.2f} patch/s)" if full_pipe else "")

    if args.skip and Path(args.skip).exists():
        skip = load_board_csv(args.skip)
        ran = sum(1 for i in skip if skip[i]["ran_det"])
        skipped = len(skip) - ran
        prep_s = statistics.mean(skip[i]["prep_ms"] for i in skip)
        gate_s = statistics.mean(skip[i]["gate_ms"] for i in skip)
        det_s_xs = [skip[i]["det_ms"] for i in skip if skip[i]["ran_det"]]
        det_s = statistics.mean(det_s_xs) if det_s_xs else 0.0
        # prep/gate 两种模式均逐 patch 执行, 与模式无关; 板端 CPU prep 跨运行
        # 受并发/热状态影响波动大(实测 25~69ms), 而 NPU 段(gate/det)稳定。
        # 故 skip 流水线采用反事实构造: full 各段均值 + det×skip 实测执行率,
        # 保证 full/skip 可比; skip 运行自身实测分段单独报告供参考。
        run_rate = ran / len(skip)
        skip_pipe = prep_m + gate_m + det_m * run_rate
        saving = (1 - skip_pipe / full_pipe) if (full_pipe and skipped) else 0.0
        print(f"skip  模式: 跳过 {skipped}/{len(skip)} ({pct(skipped / len(skip))})")
        print(f"skip  流水线/patch(反事实: full 各段 + det×{run_rate:.3f}): "
              f"{skip_pipe:.2f}ms (throughput {1000 / skip_pipe:.2f} patch/s)")
        print(f"计算节省: {pct(saving)} (相对 full; = det×跳过率/full_pipe)")
        print(f"[skip 运行实测] prep={prep_s:.2f}ms gate={gate_s:.2f}ms "
              f"det={det_s:.2f}ms (n_det={len(det_s_xs)})")
        mismatch = [i for i in skip if skip[i]["ran_det"]
                    and skip[i]["n_det"] != full[i]["n_det"]]
        print(f"[一致性] skip/full 同 patch n_det 不一致数: {len(mismatch)}")
        # gate-only 延迟(若提供)
        if args.gate and Path(args.gate).exists():
            g = load_board_csv(args.gate)
            gms = [g[i]["gate_ms"] for i in g]
            print(f"gate-only 模式: gate mean={statistics.mean(gms):.2f}ms")
        report = {
            "dataset": f"{args.dataset}/{args.split}", "n_patches": len(patches),
            "gt_dist": n_gt, "gate_acc": round(acc, 4), "confusion": cm,
            "true_skips": true_skips, "false_skips": len(false_skips),
            "latency_ms": {"prep": round(prep_m, 3), "gate": round(gate_m, 3),
                           "det": round(det_m, 3), "full_pipe": round(full_pipe, 3),
                           "skip_pipe": round(skip_pipe, 3)},
            "skip_run_ms": {"prep": round(prep_s, 3), "gate": round(gate_s, 3),
                            "det": round(det_s, 3), "det_runs": len(det_s_xs)},
            "skip_rate": round(skipped / len(skip), 4) if skip else None,
            "saving": round(saving, 4) if full_pipe else None,
        }
    else:
        report = {
            "dataset": f"{args.dataset}/{args.split}", "n_patches": len(patches),
            "gt_dist": n_gt, "gate_acc": round(acc, 4), "confusion": cm,
            "true_skips": true_skips, "false_skips": len(false_skips),
            "latency_ms": {"prep": round(prep_m, 3), "gate": round(gate_m, 3),
                           "det": round(det_m, 3), "full_pipe": round(full_pipe, 3)},
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"m3m6_{args.dataset}_{args.split}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[report] {out_json}")


if __name__ == "__main__":
    main()