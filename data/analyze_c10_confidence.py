"""C10: 板端重复实验置信区间分析 — 5x full + 5x skip (v4, HRSID test 流).

输入: runs/board_eval/v4/c10/results_v4_c10_{full,skip}_{1..5}.csv (板端 pscp 下载).
经济学协议与主结果一致(同 run 反事实): baseline = decode+prep+det (full 段均值);
SeaSkip = decode+gprep+gate + run_rate x (prep+det); run_rate 取配对 skip run 实测.
输出: runs/board_eval/v4/c10_confidence.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
C10 = ROOT / "runs" / "board_eval" / "v4" / "c10"
OUT = ROOT / "runs" / "board_eval" / "v4" / "c10_confidence.json"


def load_run(path: Path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    dec = np.array([float(r["decode_ms"]) for r in rows])
    gpr = np.array([float(r["gprep_ms"]) for r in rows])
    gat = np.array([float(r["gate_ms"]) for r in rows])
    pre = np.array([float(r["prep_ms"]) for r in rows if r["ran_det"] == "1"])
    det = np.array([float(r["det_ms"]) for r in rows if r["ran_det"] == "1"])
    pred = [int(r["gate_pred"]) for r in rows]
    pe = np.array([float(r["p_empty"]) for r in rows])
    return {
        "n": len(rows),
        "decode": float(dec.mean()), "gprep": float(gpr.mean()),
        "gate": float(gat.mean()), "prep": float(pre.mean()),
        "det": float(det.mean()),
        "pred": pred, "p_empty": pe,
        "skip_rate_argmax": 1.0 - sum(1 for p in pred if p > 0) / len(pred),
        "skip_rate_tau06": float((pe >= 0.6).mean()),
    }


fulls = [load_run(C10 / f"results_v4_c10_full_{i}.csv") for i in range(1, 6)]
skips = [load_run(C10 / f"results_v4_c10_skip_{i}.csv") for i in range(1, 6)]

# skip 决策跨 run 一致性 (确定性)
ref = skips[0]["pred"]
consist = all(s["pred"] == ref for s in skips)
n_skip = sum(1 for p in ref if p == 0)

pairs = []
for f, s in zip(fulls, skips):
    base = f["decode"] + f["prep"] + f["det"]
    gp = f["gprep"] + f["gate"]
    per_skip = f["prep"] + f["det"]
    for policy, sr in (("argmax", s["skip_rate_argmax"]), ("tau06", s["skip_rate_tau06"])):
        ss = f["decode"] + gp + (1 - sr) * per_skip
        pairs.append({
            "policy": policy,
            "decode": f["decode"], "gprep": f["gprep"], "gate": f["gate"],
            "prep": f["prep"], "det": f["det"],
            "baseline_ms": base, "seaskip_ms": ss,
            "save_vs_baseline": (base - ss) / base,
            "r_star": gp / per_skip,
            "skip_rate": sr,
        })


def agg(policy):
    sel = [p for p in pairs if p["policy"] == policy]
    keys = ["decode", "gprep", "gate", "prep", "det", "baseline_ms",
            "seaskip_ms", "save_vs_baseline", "r_star", "skip_rate"]
    out = {}
    for k in keys:
        v = np.array([p[k] for p in sel])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                  "min": float(v.min()), "max": float(v.max())}
    return out


result = {
    "protocol": "5x full + 5x skip, same-run counterfactual pairing (full_i, skip_i)",
    "skip_decision_consistency_across_5_skip_runs": consist,
    "n_skips": n_skip,
    "argmax": agg("argmax"),
    "tau06": agg("tau06"),
    "per_pair": pairs,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "per_pair"},
                 ensure_ascii=False, indent=2))
print(f"[saved] {OUT}")