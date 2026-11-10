"""HRSID (COCO 格式) -> YOLO 转换(沿用官方 6.5:3.5 划分) + 全景图登记.

用法(在 data/ 目录下执行):
    python coco2yolo.py --coco_root raw/HRSID --out processed/hrsid [--panorama-dir raw/HRSID/panoramas]

自动发现根目录下所有 COCO json(含 images/annotations 键), 按文件名含 train/val/test
定划分; 若无 val 集则从 train 按比例切出(种子固定). 单类映射: 所有类别 -> 0(ship).

--panorama-dir: 将 HRSID 全景大图登记为 split=panorama(无标签), 供 patch 流/空样本挖掘.

产物:
    processed/hrsid/images/{train,val,test}[/panorama]/...
    processed/hrsid/labels/{train,val,test}/...
    processed/hrsid/hrsid_meta.json
    processed/hrsid/hrsid.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from common import (IMG_EXTS, RAW_ROOT, PROCESSED_ROOT, clip_box, die, ensure_dir,
                    utf8_console, write_json, xyxy_to_yolo)


def resolve_image(root: Path, json_path: Path, file_name: str) -> Optional[Path]:
    """按候选顺序解析 COCO file_name 对应的图片路径."""
    fn = file_name.replace("\\", "/")
    base = PurePosixPath(fn).parts[-1]
    cands = [root / fn, json_path.parent / fn, root / base, json_path.parent / base]
    for sub in ("HRSID_JPG", "images", "JPEGImages", "JPEG", "train2016",
                "test2016", "val2016"):
        cands += [root / sub / fn, root / sub / base]
    for c in cands:
        if c.is_file():
            return c
    hits = [p for p in root.rglob(base) if p.is_file()]  # 兜底按文件名搜索
    return hits[0] if hits else None


def load_coco_jsons(root: Path) -> List[Path]:
    js = []
    for p in sorted(root.rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "images" in d and "annotations" in d:
                js.append(p)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return js


def main() -> None:
    utf8_console()
    ap = argparse.ArgumentParser(description="HRSID(COCO) -> YOLO 转换")
    ap.add_argument("--coco-root", default=str(RAW_ROOT / "HRSID"), help="HRSID 解压根目录")
    ap.add_argument("--out", default=str(PROCESSED_ROOT / "hrsid"))
    ap.add_argument("--val-ratio", type=float, default=0.05,
                    help="无官方 val 时从 train 切出的比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--panorama-dir", default=None,
                    help="全景大图目录(可选), 登记为 split=panorama")
    args = ap.parse_args()

    root = Path(args.coco_root)
    out = Path(args.out)
    if not root.is_dir():
        die(f"HRSID 根目录不存在: {root}\n"
            "  请将 HRSID 解压至 data/raw/HRSID (含 COCO json 与图片目录)")

    jsons = load_coco_jsons(root)
    if not jsons:
        die(f"未发现 COCO 格式 json(需含 images/annotations 键): {root}")
    print(f"[HRSID] 发现 COCO 标注文件: {[p.name for p in jsons]}")

    # ---- 逐 json 收集样本 ----
    # sample: {id, src_img, w, h, boxes, split}
    samples: List[dict] = []
    cat_names, missing, dup = set(), 0, 0
    seen_files: set = set()  # 跨 json 去重(防 train_test 合并文件重复导入)
    for jp in jsons:
        split = ("test" if "test" in jp.name.lower()
                 else "val" if "val" in jp.name.lower() else "train")
        data = json.loads(jp.read_text(encoding="utf-8"))
        cat_names |= {c.get("name", "?") for c in data.get("categories", [])}
        img_by_id = {im["id"]: im for im in data.get("images", [])}
        boxes_by_img: Dict[int, list] = {}
        for ann in data.get("annotations", []):
            if ann.get("bbox") is None:
                continue
            x, y, w, h = map(float, ann["bbox"])
            boxes_by_img.setdefault(ann["image_id"], []).append((x, y, x + w, y + h))
        for _id, im in img_by_id.items():
            fn_base = PurePosixPath(im.get("file_name", str(_id))
                                    .replace("\\", "/")).parts[-1]
            if fn_base in seen_files:
                dup += 1
                continue
            src = resolve_image(root, jp, im.get("file_name", ""))
            if src is None:
                missing += 1
                continue
            seen_files.add(fn_base)
            w, h = int(im.get("width", 0)), int(im.get("height", 0))
            if w <= 0 or h <= 0:
                from PIL import Image
                with Image.open(src) as f:
                    w, h = f.size
            samples.append({"id": Path(im.get("file_name", str(_id))).stem,
                            "src_img": str(src), "w": w, "h": h,
                            "boxes": boxes_by_img.get(_id, []), "split": split})
    if missing:
        print(f"[警告] {missing} 张图片未找到, 已跳过")
    if dup:
        print(f"[提示] {dup} 条与其余 json 重复的图片条目已跳过(如 train_test 合并文件)")
    if not samples:
        die("没有可用的 (COCO 标注, 图片) 样本对")
    if not {n.lower() for n in cat_names} <= {"ship"}:
        print(f"[提示] COCO 类别: {sorted(cat_names)} -> 全部映射为单类 0(ship)")

    # ---- 无 val 时从 train 切出 ----
    if not any(s["split"] == "val" for s in samples) and args.val_ratio > 0:
        rng = random.Random(args.seed)
        tr = [s for s in samples if s["split"] == "train"]
        rng.shuffle(tr)
        n_va = round(len(tr) * args.val_ratio)
        for s in tr[:n_va]:
            s["split"] = "val"
        if n_va:
            print(f"[HRSID] 从 train 切出 {n_va} 张作为 val (ratio={args.val_ratio})")

    # ---- 写出 ----
    meta_images, degenerated = [], 0
    seen_ids: Dict[str, int] = {}
    for s in samples:
        seen_ids[s["id"]] = seen_ids.get(s["id"], 0) + 1
        uid = s["id"] if seen_ids[s["id"]] == 1 else f"{s['id']}_{seen_ids[s['id']]}"
        img_dst = ensure_dir(out / "images" / s["split"]) / (uid + Path(s["src_img"]).suffix)
        lbl_dst = ensure_dir(out / "labels" / s["split"]) / f"{uid}.txt"
        shutil.copy2(s["src_img"], img_dst)

        lines = []
        for (x0, y0, x1, y1) in s["boxes"]:
            cb = clip_box(x0, y0, x1, y1, s["w"], s["h"])
            if cb is None:
                degenerated += 1
                continue
            lines.append("0 " + " ".join(f"{v:.6f}" for v in
                         xyxy_to_yolo(*cb, s["w"], s["h"])))
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        meta_images.append({"id": uid, "file": f"images/{s['split']}/{img_dst.name}",
                            "split": s["split"], "scene": None,
                            "width": s["w"], "height": s["h"], "n_boxes": len(lines)})

    # ---- 全景图登记(无标签, split=panorama) ----
    if args.panorama_dir:
        pdir = Path(args.panorama_dir)
        if not pdir.is_dir():
            print(f"[警告] 全景图目录不存在, 跳过: {pdir}")
        else:
            p_dst = ensure_dir(out / "images" / "panorama")
            n_pan = 0
            for p in sorted(pdir.rglob("*")):
                if p.suffix.lower() in IMG_EXTS:
                    shutil.copy2(p, p_dst / p.name)
                    meta_images.append({"id": p.stem, "file": f"images/panorama/{p.name}",
                                        "split": "panorama", "scene": None,
                                        "width": None, "height": None, "n_boxes": None})
                    n_pan += 1
            print(f"[HRSID] 登记全景图 {n_pan} 张 -> split=panorama")

    meta = {"dataset": "hrsid", "source": str(root), "seed": args.seed,
            "n_images": len(meta_images), "images": meta_images}
    write_json(out / "hrsid_meta.json", meta)

    yaml_text = (f"# 由 coco2yolo.py 自动生成\n"
                 f"path: {out.resolve().as_posix()}\n"
                 f"train: images/train\nval: images/val\ntest: images/test\n"
                 f"names:\n  0: ship\n")
    (out / "hrsid.yaml").write_text(yaml_text, encoding="utf-8")

    # ---- 报告 ----
    print(f"[HRSID] 共 {len(meta_images)} 张, 退化框 {degenerated}")
    for split in ("train", "val", "test", "panorama"):
        items = [m for m in meta_images if m["split"] == split]
        if items:
            print(f"  {split:9s}: {len(items):4d} 张 | 框 "
                  f"{sum(m['n_boxes'] or 0 for m in items)}")
    print(f"  输出: {out}")
    print("下一步: python mine_gate_patches.py --datasets ssdd hrsid")


if __name__ == "__main__":
    main()