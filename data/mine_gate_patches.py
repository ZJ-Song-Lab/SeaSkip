"""从已转换数据集挖掘 gate 三态训练 patch(论文公式 1、2 的代码实现).

用法(在 data/ 目录下执行):
    python mine_gate_patches.py --datasets ssdd hrsid --patch 640 --stride 160 --save-crops

输入: processed/<ds>/<ds>_meta.json 与 labels/ (voc2yolo.py / coco2yolo.py 产物)
输出(gate/):
    gate_train.txt / gate_val.txt     每行: <crops 相对路径> <class_idx 0/1/2>
    crops/{train,val}/<ds>_<id>_x.._y...jpg
    manifest.jsonl                   每个 patch: 来源/窗口/类别/n/min_side/scene
    gate_distribution.json          分布报告(train 空 patch 比例 < 15% 触发 R1 预警)

说明:
  - SSDD 小图(<=patch)整图为单窗且必含船 -> 主要提供难度分层(inshore=complex);
  - 空 patch 主要来自 HRSID 800x800 的边缘/角落窗与全景图;
  - gate 训练用图像清单可直接接普通分类训练器(格式: 路径 + 类别索引).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from common import (GATE_CLASSES, GATE_ROOT, PROCESSED_ROOT, crop_patch,
                    ensure_dir, label_patch, load_yolo_labels, open_image,
                    read_json, sliding_windows, utf8_console, write_json,
                    write_lines)


def mine_dataset(ds: str, processed_root: Path, splits: List[str], patch: int,
                 stride: int, k_simple: int, small_px: float, save_crops: bool,
                 out_root: Path, manifest_path: Path, lines_by_split: Dict[str, List[str]],
                 stats: Dict[str, Counter]) -> None:
    meta_path = processed_root / ds / f"{ds}_meta.json"
    if not meta_path.exists():
        print(f"[警告] 未找到 {meta_path}, 跳过 {ds} (先运行 voc2yolo/coco2yolo)")
        return
    meta = read_json(meta_path)
    n_no_gt = 0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for e in meta["images"]:
            if e["split"] not in splits:
                continue
            lbl = processed_root / ds / "labels" / e["split"] / f"{e['id']}.txt"
            img_path = processed_root / ds / e["file"]
            if e.get("n_boxes") is None and not lbl.exists():
                n_no_gt += 1  # 无 GT 的图(如未配标注的全景图)不能安全打标, 跳过
                continue
            w = e["width"] or 0
            h = e["height"] or 0
            if save_crops and (w <= 0 or h <= 0):
                img = open_image(img_path)
                w, h = img.size
            boxes = load_yolo_labels(lbl, w, h)[:, 1:5]
            img = open_image(img_path) if save_crops else None

            for win in sliding_windows(w, h, patch, stride):
                cls, info = label_patch(win, boxes, e.get("scene"), k_simple, small_px)
                stats[(ds, e["split"])][cls] += 1
                name = f"{ds}_{e['id']}_x{win[0]}_y{win[1]}.jpg"
                if save_crops:
                    crop_patch(img, win, patch, out_root / "crops" / e["split"] / name)
                lines_by_split[e["split"]].append(f"crops/{e['split']}/{name} {cls}")
                mf.write(json.dumps({
                    "dataset": ds, "id": e["id"], "split": e["split"],
                    "scene": e.get("scene"), "window": list(win),
                    "class_idx": cls, "class": GATE_CLASSES[cls],
                    "n_boxes": info["n"], "min_side": info["min_side"],
                }, ensure_ascii=False) + "\n")
    if n_no_gt:
        print(f"[提示] {ds}: {n_no_gt} 张无 GT 的图(如未配标注的全景图)未参与打标")


def main() -> None:
    utf8_console()
    ap = argparse.ArgumentParser(description="gate 三态 patch 挖掘")
    ap.add_argument("--datasets", nargs="+", default=["ssdd", "hrsid"],
                    choices=["ssdd", "hrsid"])
    ap.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    ap.add_argument("--out", default=str(GATE_ROOT))
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--patch", type=int, default=640, help="patch 边长")
    ap.add_argument("--stride", type=int, default=160, help="滑窗步长")
    ap.add_argument("--k-simple", type=int, default=5, help="simple 档最大船数(论文 k)")
    ap.add_argument("--small-px", type=float, default=32.0, help="小目标阈值(px)")
    ap.add_argument("--cap-per-class", type=int, default=0,
                    help="每类每 split 最多保留 N 个(0=不限制), 用于压制 empty 主导")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-crops", action="store_true", help="只生成清单不落图")
    args = ap.parse_args()

    out_root = Path(args.out)
    processed_root = Path(args.processed_root)
    save_crops = not args.no_crops
    if args.stride > args.patch:
        print(f"[警告] stride({args.stride}) > patch({args.patch}) 会漏覆盖, 建议小于等于")

    lines_by_split: Dict[str, List[str]] = {s: [] for s in args.splits}
    stats: Dict[tuple, Counter] = {}
    manifest_path = out_root / "manifest.jsonl"
    ensure_dir(out_root)
    manifest_path.write_text("", encoding="utf-8")

    for ds in args.datasets:
        for sp in args.splits:
            stats.setdefault((ds, sp), Counter())
        mine_dataset(ds, processed_root, args.splits, args.patch, args.stride,
                     args.k_simple, args.small_px, save_crops, out_root,
                     manifest_path, lines_by_split, stats)

    # ---- 可选类别下采样 ----
    if args.cap_per_class > 0:
        rng = random.Random(args.seed)
        for sp, lines in lines_by_split.items():
            by_cls: Dict[int, List[str]] = {}
            for ln in lines:
                by_cls.setdefault(int(ln.rsplit(" ", 1)[1]), []).append(ln)
            kept: List[str] = []
            for cls, ls in by_cls.items():
                rng.shuffle(ls)
                kept += ls[: args.cap_per_class]
            lines_by_split[sp] = sorted(kept)
            print(f"[下采样] {sp}: 保留 {len(kept)}/{len(lines)} (cap={args.cap_per_class}/类)")

    # ---- 清单 ----
    for sp, lines in lines_by_split.items():
        write_lines(out_root / f"gate_{sp}.txt", lines)

    # ---- 分布报告 ----
    dist = {"patch": args.patch, "stride": args.stride,
            "k_simple": args.k_simple, "small_px": args.small_px,
            "counts": {}, "empty_ratio_train": None}
    for (ds, sp), cnt in stats.items():
        dist["counts"][f"{ds}/{sp}"] = {GATE_CLASSES[i]: cnt.get(i, 0) for i in range(3)}
    # train 空 patch 比例(用于 R1 预警)
    empty_tr = sum(stats[(ds, "train")].get(0, 0) for ds in args.datasets
                   if (ds, "train") in stats)
    total_tr = sum(sum(stats[(ds, "train")].values()) for ds in args.datasets
                   if (ds, "train") in stats)
    if total_tr:
        dist["empty_ratio_train"] = round(empty_tr / total_tr, 6)
    write_json(out_root / "gate_distribution.json", dist)

    print("[gate 挖掘完成] 分布:")
    for (ds, sp), cnt in stats.items():
        total = sum(cnt.values())
        if total == 0:
            continue
        print(f"  {ds}/{sp}: total={total}, empty={cnt.get(0,0)} "
              f"({cnt.get(0,0)/total:.1%}), simple={cnt.get(1,0)} "
              f"({cnt.get(1,0)/total:.1%}), complex={cnt.get(2,0)} "
              f"({cnt.get(2,0)/total:.1%})")
    if total_tr and empty_tr / total_tr < 0.15:
        print("[R1 预警] train 空 patch 比例 < 15%: 建议启用难度感知两级推理退路, "
              "或检查全景图标注是否已导入(见 02 文档 §8 R1)")
    print(f"  输出: {out_root}")
    print("下一步: python check_dataset.py --visualize 20")
    print("        python build_patch_stream.py --dataset hrsid --split test")


if __name__ == "__main__":
    main()