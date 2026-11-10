"""PC 侧参考检测: 用训练好的 SSDD 检测器对板端 patch 流做 CPU 推理.

用途: 与板端 seaskip_eval full 模式结果对比, 验证 ATC+C++ 解码管线的一致性 (M2 板端对齐).
输出: board/pc_ref_ssdd_test.json  {file: [[x1,y1,x2,y2,conf], ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

DATA = Path(__file__).resolve().parent
STREAM = json.loads(
    (DATA / "board" / "patch_stream_ssdd_test.json").read_text(encoding="utf-8"))
PATCH_DIR = DATA / "board" / "patches" / "ssdd_test"
OUT = DATA / "board" / "pc_ref_ssdd_test.json"

model = YOLO(str(DATA / "runs" / "det" / "ssdd_v1" / "weights" / "best.pt"))
files = [PATCH_DIR / p["file"].split("/")[-1] for p in STREAM["patches"]]

ref = {}
BATCH = 32
for i in range(0, len(files), BATCH):
    results = model.predict(source=[str(f) for f in files[i:i + BATCH]],
                            imgsz=640, conf=0.25, iou=0.45, device="cpu",
                            verbose=False)
    for f, r in zip(files[i:i + BATCH], results):
        boxes = r.boxes.xyxy.cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        ref[f.name] = [[round(b[0], 1), round(b[1], 1), round(b[2], 1),
                        round(b[3], 1), round(c, 4)] for b, c in zip(boxes, confs)]
    print(f"[{i + len(results)}/{len(files)}]")

OUT.write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
n_tot = sum(len(v) for v in ref.values())
print(f"[done] {len(ref)} files, {n_tot} boxes -> {OUT}")