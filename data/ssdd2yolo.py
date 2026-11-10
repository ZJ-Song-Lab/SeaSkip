"""SSDD (本地 YOLO-HBB 布局) -> SeaSkip 统一 processed 格式(官方划分 + test 场景).

用法(在 data/ 目录下执行):
    python ssdd2yolo.py --ssdd-root "D:/SAR datasetS/SSDD" --out processed/ssdd

适配本地 SSDD 布局(与官方 VOC 版不同, 无 XML):
    images/train/*.jpg           1160 张全量图
    labels_hbb/train/*.txt       1160 个 HBB YOLO 标签 (cls cx cy w h, 归一化)
    labels/{train,test}/*.txt    官方划分定义 (train 928 / test 232, val=test 为拷贝)
    images/JPEGImages_test_{inshore,offshore}/  test 集场景信息 (46 + 186)

划分策略(避免测试集泄漏):
    train 928 内部按 --val-ratio 切出 val (默认 10%), test 232 独立;
    检测器与 gate 的训练/验证都只碰 train+val, test 仅评估.

产物(与 voc2yolo.py / coco2yolo.py 同 schema):
    processed/ssdd/images/{train,val,test}/xxx.jpg
    processed/ssdd/labels/{train,val,test}/xxx.txt
    processed/ssdd/ssdd_meta.json   (id/file/split/scene/width/height/n_boxes)
    processed/ssdd/ssdd.yaml        ultralytics data 配置
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Optional

from common import (IMG_EXTS, RAW_ROOT, PROCESSED_ROOT, ensure_dir, find_image,
                    require_pillow, utf8_console, write_json)
from PIL import Image


def read_label(path: Path):
    """读取 HBB YOLO 标签 -> (合法行, 原始总行数).
    合法行: 5 列, 类别 0, 数值在 [0,1], w/h > 0."""
    lines, n_raw = [], 0
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        n_raw += 1
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        if cls != 0 or any(v < -1e-6 or v > 1 + 1e-6 for v in vals) \
                or vals[2] <= 0 or vals[3] <= 0:
            continue
        lines.append("0 " + " ".join(f"{v:.6f}" for v in vals))
    return lines, n_raw


def main() -> None:
    utf8_console()
    require_pillow()
    ap = argparse.ArgumentParser(description="SSDD(本地 YOLO-HBB 布局) -> 统一格式")
    ap.add_argument("--ssdd-root", default=str(RAW_ROOT / "SSDD"))
    ap.add_argument("--out", default=str(PROCESSED_ROOT / "ssdd"))
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="从官方 train 内部切 val 的比例(防 test 泄漏)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.ssdd_root)
    out = Path(args.out)
    img_dir = root / "images" / "train"
    hbb_dir = root / "labels_hbb" / "train"
    split_dirs = {sp: root / "labels" / sp for sp in ("train", "test")}
    for p in (img_dir, hbb_dir, *split_dirs.values()):
        if not p.is_dir():
            raise SystemExit(f"[错误] 目录不存在: {p}\n  请检查 --ssdd-root 指向的 SSDD 布局")
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("[错误] --val-ratio 需在 [0,1) 内")

    # ---- 场景信息(仅 test 提供) ----
    scene_of: Dict[str, str] = {}
    for scene in ("inshore", "offshore"):
        d = root / "images" / f"JPEGImages_test_{scene}"
        if d.is_dir():
            for p in d.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    scene_of[p.stem] = scene

    # ---- 官方划分 + HBB 标签 ----
    official = {sp: {p.stem for p in sd.glob("*.txt")}
                for sp, sd in split_dirs.items()}
    stems = sorted({p.stem for p in img_dir.iterdir()
                    if p.suffix.lower() in IMG_EXTS})
    missing_lbl = [s for s in stems if not (hbb_dir / f"{s}.txt").exists()]
    if missing_lbl:
        raise SystemExit(f"[错误] {len(missing_lbl)} 张图缺 HBB 标签, 例: {missing_lbl[:3]}")
    off_unknown = [s for s in stems
                   if s not in official["train"] and s not in official["test"]]
    if off_unknown:
        print(f"[警告] {len(off_unknown)} 张图不在官方 train/test 中, 例: {off_unknown[:3]}")

    # train 池内部切 val
    train_pool = sorted(s for s in stems if s in official["train"])
    test_pool = sorted(s for s in stems if s in official["test"])
    rng = random.Random(args.seed)
    rng.shuffle(train_pool)
    n_va = round(len(train_pool) * args.val_ratio)
    val_pool, train_pool = train_pool[:n_va], train_pool[n_va:]

    # ---- 写出 ----
    meta_images, bad_lines = [], 0
    for split, pool in (("train", train_pool), ("val", val_pool), ("test", test_pool)):
        for stem in pool:
            src = find_image(img_dir, stem)
            if src is None:
                raise SystemExit(f"[错误] 找不到图片: {img_dir / stem}")
            img_dst = ensure_dir(out / "images" / split) / (stem + src.suffix)
            lbl_dst = ensure_dir(out / "labels" / split) / f"{stem}.txt"
            shutil.copy2(src, img_dst)
            lines, n_raw = read_label(hbb_dir / f"{stem}.txt")
            bad_lines += n_raw - len(lines)
            lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""),
                               encoding="utf-8")
            with Image.open(src) as im:
                w, h = im.size
            scene: Optional[str] = scene_of.get(stem)
            meta_images.append({"id": stem, "file": f"images/{split}/{img_dst.name}",
                                "split": split, "scene": scene,
                                "width": w, "height": h, "n_boxes": len(lines)})

    meta = {"dataset": "ssdd", "source": str(root), "seed": args.seed,
            "split_policy": f"官方 train/test + train 内部切 val(ratio={args.val_ratio}); "
                            "原始 labels/val 与 test 完全相同, 不采用",
            "val_ratio": args.val_ratio,
            "n_images": len(meta_images), "images": meta_images}
    write_json(out / "ssdd_meta.json", meta)
    (out / "ssdd.yaml").write_text(
        f"# 由 ssdd2yolo.py 自动生成\n"
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: ship\n", encoding="utf-8")

    # ---- 报告 ----
    print(f"[SSDD] 共 {len(meta_images)} 张, 无效标签行 {bad_lines}")
    for split in ("train", "val", "test"):
        items = [m for m in meta_images if m["split"] == split]
        n_box = sum(m["n_boxes"] for m in items)
        n_in = sum(1 for m in items if m["scene"] == "inshore")
        n_off = sum(1 for m in items if m["scene"] == "offshore")
        print(f"  {split:5s}: {len(items):4d} 张 | 框 {n_box:5d} | "
              f"inshore {n_in}, offshore {n_off}, "
              f"未知 {len(items) - n_in - n_off}")
    print(f"  输出: {out}")
    print("下一步: python mine_gate_patches.py --datasets ssdd hrsid")


if __name__ == "__main__":
    main()