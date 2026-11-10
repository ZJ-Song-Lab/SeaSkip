"""生成 Atlas 200I DK 板端实验的 patch 流清单与预切块(实验 M3/M6 的输入).

用法(在 data/ 目录下执行):
    python build_patch_stream.py --dataset hrsid --split test --stride 640
    python build_patch_stream.py --dataset hrsid --split panorama --stride 320
    python build_patch_stream.py --dataset ssdd --split test

输出(board/):
    patch_stream_<ds>_<split>.json   清单: 元信息 + 每个 patch(场景/坐标/GT/文件)
    patches/<ds>_<split>/%04d_%04d.jpg  预切块(按 idx 顺序命名)

板端对接方式:
    C++ 编排器按 idx 顺序读取 patches/*.jpg -> gate.om 推理 -> 阈值决策 ->
    (非空)det.om 推理 + CPU NMS; 结果按 idx 回传 PC, 用清单中的 gt_boxes 合并评估.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List

from common import (BOARD_ROOT, GATE_CLASSES, PROCESSED_ROOT, boxes_in_window,
                    crop_patch, ensure_dir, label_patch, load_yolo_labels,
                    open_image, read_json, sliding_windows, utf8_console,
                    write_json)


def main() -> None:
    utf8_console()
    ap = argparse.ArgumentParser(description="板端 patch 流构建")
    ap.add_argument("--dataset", required=True, choices=["ssdd", "hrsid"])
    ap.add_argument("--split", default="test",
                    help="test / val / panorama(需 coco2yolo --panorama-dir 已登记)")
    ap.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    ap.add_argument("--out", default=str(BOARD_ROOT))
    ap.add_argument("--patch", type=int, default=640)
    ap.add_argument("--stride", type=int, default=640,
                    help="640=无重叠; 320=50% 重叠(全覆盖评估用)")
    ap.add_argument("--k-simple", type=int, default=5)
    ap.add_argument("--small-px", type=float, default=32.0)
    ap.add_argument("--no-crops", action="store_true", help="只写清单不落图")
    args = ap.parse_args()

    processed_root = Path(args.processed_root)
    out_root = Path(args.out)
    meta = read_json(processed_root / args.dataset / f"{args.dataset}_meta.json")
    entries = [e for e in meta["images"] if e["split"] == args.split]
    if not entries:
        raise SystemExit(f"[错误] {args.dataset} 中无 split={args.split} 的样本 "
                         f"(panorama 需先在 coco2yolo.py 用 --panorama-dir 登记并补 GT)")

    save_crops = not args.no_crops
    crop_dir = ensure_dir(out_root / "patches" / f"{args.dataset}_{args.split}")
    patches: List[dict] = []
    exp_cnt: Counter = Counter()

    for scene_idx, e in enumerate(entries):
        img_path = processed_root / args.dataset / e["file"]
        w, h = e.get("width") or 0, e.get("height") or 0
        if save_crops or not w or not h:
            img = open_image(img_path)
            w, h = img.size
        else:
            img = None
        lbl = processed_root / args.dataset / "labels" / e["split"] / f"{e['id']}.txt"
        boxes = load_yolo_labels(lbl, w, h)[:, 1:5]

        for p_idx, win in enumerate(sliding_windows(w, h, args.patch, args.stride)):
            cls_exp, _ = label_patch(win, boxes, e.get("scene"),
                                     args.k_simple, args.small_px)
            exp_cnt[cls_exp] += 1
            rel = boxes_in_window(win, boxes)
            n = int(rel.sum())
            gt = boxes[rel].tolist() if n else []
            rec = {
                "idx": len(patches), "scene_id": e["id"], "scene_idx": scene_idx,
                "window": list(win), "n_boxes": n,
                "gt_boxes_abs": gt,
            }
            if save_crops:
                fname = f"{scene_idx:04d}_{p_idx:04d}.jpg"
                crop_patch(img, win, args.patch, crop_dir / fname)
                rec["file"] = f"patches/{args.dataset}_{args.split}/{fname}"
            patches.append(rec)

    n_empty_expect = exp_cnt.get(0, 0)
    n_patches_total = len(patches)
    manifest = {
        "meta": {
            "dataset": args.dataset, "split": args.split,
            "patch": args.patch, "stride": args.stride,
            "n_scenes": len(entries), "n_patches": n_patches_total,
            "expected_class_dist": {GATE_CLASSES[i]: exp_cnt.get(i, 0)
                                    for i in range(3)},
            "n_empty_expected": n_empty_expect,
            "empty_ratio_expected": round(n_empty_expect / n_patches_total, 6)
            if n_patches_total else None,
            "note": "expected_class_dist 为按 GT 与论文公式(1)(2)统计的理论分布, "
                    "用于评估 gate 跳过质量与理论收益上限",
        },
        "patches": patches,
    }
    out_json = out_root / f"patch_stream_{args.dataset}_{args.split}.json"
    write_json(out_json, manifest)

    dist_txt = " ".join(f"{GATE_CLASSES[i]}={exp_cnt.get(i, 0)}"
                        f"({exp_cnt.get(i, 0) / n_patches_total:.1%})"
                        for i in range(3)) if n_patches_total else "-"
    print(f"[patch 流] {args.dataset}/{args.split}: "
          f"{len(entries)} 景 -> {n_patches_total} patch | 理论分布 {dist_txt}")
    print(f"  清单: {out_json}")
    if save_crops:
        print(f"  预切块: {crop_dir}")
    print("板端: C++ 按 idx 顺序读图 -> gate.om -> 决策 -> det.om, 结果按 idx 回传合并评估")


if __name__ == "__main__":
    main()