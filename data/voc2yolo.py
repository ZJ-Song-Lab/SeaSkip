"""SSDD (VOC 格式) -> YOLO 格式转换 + inshore/offshore 元数据 + 分层划分.

用法(在 data/ 目录下执行):
    python voc2yolo.py --voc_root raw/SSDD --out processed/ssdd

SSDD 目录需包含: Annotations/*.xml 与 JPEGImages/ (或 JPGImages/) 图片.
inshore/offshore 判定顺序: 图片父目录名 -> 文件名关键词 -> 未知(null).

产物:
    processed/ssdd/images/{train,val,test}/xxx.jpg
    processed/ssdd/labels/{train,val,test}/xxx.txt   YOLO 归一化, 单类 0=ship
    processed/ssdd/ssdd_meta.json                    场景/划分/尺寸/框数元数据
    processed/ssdd/ssdd.yaml                         ultralytics data 配置
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from common import (RAW_ROOT, PROCESSED_ROOT, clip_box, die, ensure_dir,
                    find_image, guess_scene, utf8_console, write_json,
                    xyxy_to_yolo)


def find_subdir(root: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = root / n
        if p.is_dir():
            return p
    return None


def parse_voc(xml_path: Path):
    """解析 VOC XML -> (w, h, [(x0,y0,x1,y1), ...], 类名列表)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    boxes, names = [], []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            box = (float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                   float(bb.findtext("xmax")), float(bb.findtext("ymax")))
        except (TypeError, ValueError):
            continue
        boxes.append(box)
        names.append(obj.findtext("name", "ship"))
    return w, h, boxes, names


def main() -> None:
    utf8_console()
    ap = argparse.ArgumentParser(description="SSDD(VOC) -> YOLO 转换")
    ap.add_argument("--voc-root", default=str(RAW_ROOT / "SSDD"), help="SSDD 解压根目录")
    ap.add_argument("--out", default=str(PROCESSED_ROOT / "ssdd"), help="输出目录")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.2, help="test 为剩余比例")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    voc_root = Path(args.voc_root)
    out = Path(args.out)
    if not voc_root.is_dir():
        die(f"SSDD 根目录不存在: {voc_root}\n"
            "  请将 SSDD 解压至 data/raw/SSDD (内含 Annotations/ 与 JPEGImages/)")
    ann_dir = find_subdir(voc_root, ["Annotations", "annotations"])
    img_dir = find_subdir(voc_root, ["JPEGImages", "JPGImages", "images", "Images"])
    if ann_dir is None or img_dir is None:
        die(f"未找到 Annotations/ 或 JPEGImages/ 目录: {voc_root}")

    xmls = sorted(ann_dir.glob("*.xml"))
    if not xmls:
        die(f"Annotations 下无 XML: {ann_dir}")

    # ---- 解析全部样本 ----
    entries = []  # {id, src_img, scene, w, h, boxes}
    skipped = 0
    for xp in xmls:
        stem = xp.stem
        img_path = find_image(img_dir, stem)
        if img_path is None:
            skipped += 1
            continue
        w, h, boxes, _names = parse_voc(xp)
        if w <= 0 or h <= 0:
            from PIL import Image
            with Image.open(img_path) as im:
                w, h = im.size
        scene = guess_scene(img_path)
        entries.append({"id": stem, "src_img": str(img_path), "scene": scene,
                        "w": w, "h": h, "boxes": boxes})
    if skipped:
        print(f"[警告] {skipped} 个 XML 未找到对应图片, 已跳过")
    if not entries:
        die("没有可用的 (XML, 图片) 样本对")

    # ---- 按场景分层划分 ----
    buckets: Dict[Optional[str], List[dict]] = {}
    for e in entries:
        buckets.setdefault(e["scene"], []).append(e)

    rng = random.Random(args.seed)
    split_of: Dict[str, str] = {}
    for scene, items in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        rng.shuffle(items)
        n = len(items)
        n_tr = round(n * args.train_ratio)
        n_va = round(n * args.val_ratio)
        for i, e in enumerate(items):
            split_of[e["id"]] = ("train" if i < n_tr else
                                 "val" if i < n_tr + n_va else "test")

    # ---- 写出 ----
    meta_images, degenerated = [], 0
    for e in entries:
        split = split_of[e["id"]]
        img_dst = ensure_dir(out / "images" / split) / (e["id"] + Path(e["src_img"]).suffix)
        lbl_dst = ensure_dir(out / "labels" / split) / f"{e['id']}.txt"
        shutil.copy2(e["src_img"], img_dst)

        lines = []
        for (x0, y0, x1, y1) in e["boxes"]:
            cb = clip_box(x0, y0, x1, y1, e["w"], e["h"])
            if cb is None:
                degenerated += 1
                continue
            lines.append("0 " + " ".join(f"{v:.6f}" for v in
                         xyxy_to_yolo(*cb, e["w"], e["h"])))
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        meta_images.append({
            "id": e["id"], "file": f"images/{split}/{img_dst.name}",
            "split": split, "scene": e["scene"],
            "width": e["w"], "height": e["h"], "n_boxes": len(lines),
        })

    meta = {"dataset": "ssdd", "source": str(voc_root), "seed": args.seed,
            "ratios": [args.train_ratio, args.val_ratio,
                       round(1 - args.train_ratio - args.val_ratio, 6)],
            "n_images": len(meta_images), "images": meta_images}
    write_json(out / "ssdd_meta.json", meta)

    yaml_text = (f"# 由 voc2yolo.py 自动生成\n"
                 f"path: {out.resolve().as_posix()}\n"
                 f"train: images/train\nval: images/val\ntest: images/test\n"
                 f"names:\n  0: ship\n")
    (out / "ssdd.yaml").write_text(yaml_text, encoding="utf-8")

    # ---- 报告 ----
    print(f"[SSDD] 共 {len(meta_images)} 张 (跳过 XML {skipped}), 退化框 {degenerated}")
    for split in ("train", "val", "test"):
        items = [m for m in meta_images if m["split"] == split]
        n_box = sum(m["n_boxes"] for m in items)
        n_in = sum(1 for m in items if m["scene"] == "inshore")
        print(f"  {split:5s}: {len(items):4d} 张 | 框 {n_box:5d} | inshore {n_in}, "
              f"offshore {sum(1 for m in items if m['scene'] == 'offshore')}, "
              f"未知 {sum(1 for m in items if m['scene'] is None)}")
    print(f"  输出: {out}")
    print("下一步: python mine_gate_patches.py --datasets ssdd hrsid")


if __name__ == "__main__":
    main()