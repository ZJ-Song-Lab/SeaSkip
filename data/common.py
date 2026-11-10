"""SeaSkip 数据管线 - 公共工具模块.

适用平台: PC (Windows/Linux) 与 Atlas 200I DK (AArch64 Linux, Python 3.8+).
依赖: numpy(必须), Pillow(切图/可视化时需要).

路径约定(相对本文件所在目录):
    raw/        原始数据集(SSDD/HRSID 解压后放这里)
    processed/  格式转换产物(YOLO txt + 元数据)
    gate/       gate 分支训练数据(patch + 三态标签)
    board/      Atlas 200I DK 板端实验输入(patch 流清单与预切块)

三态标签(与论文公式 1、2 一致): 0=empty, 1=simple, 2=complex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image  # 可选依赖: 切图/可视化需要
except ImportError:  # pragma: no cover
    Image = None

# ---------------- 路径 ----------------
DATA_ROOT = Path(__file__).resolve().parent
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
GATE_ROOT = DATA_ROOT / "gate"
BOARD_ROOT = DATA_ROOT / "board"

GATE_CLASSES: Tuple[str, ...] = ("empty", "simple", "complex")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def utf8_console() -> None:
    """Windows 控制台中文输出保护."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def require_pillow() -> None:
    if Image is None:
        raise SystemExit("缺少 Pillow, 请先安装: pip install pillow")


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_lines(path: Path, lines: Sequence[str]) -> None:
    ensure_dir(Path(path).parent)
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------- 几何 ----------------
def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """归一化 YOLO 框 -> 绝对坐标 xyxy."""
    return ((cx - w / 2) * img_w, (cy - h / 2) * img_h,
            (cx + w / 2) * img_w, (cy + h / 2) * img_h)


def xyxy_to_yolo(x0: float, y0: float, x1: float, y1: float,
                 img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """绝对坐标 xyxy -> 归一化 YOLO (cx, cy, w, h)."""
    return ((x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h,
            (x1 - x0) / img_w, (y1 - y0) / img_h)


def iou_xyxy(a, b) -> np.ndarray:
    """IoU 矩阵. a:(N,4), b:(M,4) -> (N,M); 空输入安全."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix0 = np.maximum(ax0[:, None], bx0[None, :])
    iy0 = np.maximum(ay0[:, None], by0[None, :])
    ix1 = np.minimum(ax1[:, None], bx1[None, :])
    iy1 = np.minimum(ay1[:, None], by1[None, :])
    iw = np.clip(ix1 - ix0, 0, None)
    ih = np.clip(iy1 - iy0, 0, None)
    inter = iw * ih
    area_a = np.clip(ax1 - ax0, 0, None) * np.clip(ay1 - ay0, 0, None)
    area_b = np.clip(bx1 - bx0, 0, None) * np.clip(by1 - by0, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """读取 YOLO txt -> (N,5) [cls,x0,y0,x1,y1] 绝对坐标; 无标签返回 (0,5)."""
    p = Path(label_path)
    if not p.exists():
        return np.zeros((0, 5))
    rows: List[List[float]] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        rows.append([cls, *yolo_to_xyxy(cx, cy, w, h, img_w, img_h)])
    if not rows:
        return np.zeros((0, 5))
    return np.asarray(rows, dtype=np.float64)


def boxes_in_window(window, boxes_xyxy) -> np.ndarray:
    """与窗口'相关'的 GT 集合(论文公式 1 的补集): IoU>0 或中心点落入窗口."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    w = np.asarray(window, dtype=np.float64).reshape(1, 4)
    iou = iou_xyxy(w, boxes)[0]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    inside = ((cx >= w[0, 0]) & (cx <= w[0, 2]) &
              (cy >= w[0, 1]) & (cy <= w[0, 3]))
    return (iou > 0) | inside


def label_patch(window, boxes_xyxy, scene: Optional[str] = None,
                k_simple: int = 5, small_px: float = 32.0) -> Tuple[int, dict]:
    """论文公式(1)(2): patch 三态标签.

    empty   : 无相关 GT
    complex : n > k_simple 或 最小边 < small_px 或 近岸(inshore)
    simple  : 其余
    返回 (class_idx, info).
    """
    sel = boxes_in_window(window, boxes_xyxy)
    n = int(sel.sum())
    if n == 0:
        return 0, {"n": 0, "min_side": None}
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)[sel]
    sides = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    min_side = float(sides.min())
    is_complex = (n > k_simple) or (min_side < small_px) or (scene == "inshore")
    idx = 2 if is_complex else 1
    return idx, {"n": n, "min_side": round(min_side, 2)}


def sliding_windows(w: int, h: int, patch: int, stride: int) -> List[Tuple[int, int, int, int]]:
    """滑窗列表 [(x0,y0,x1,y1)].

    - 图像边长 <= patch: 返回单一整图窗(切图时 letterbox 补边);
    - 否则按 stride 滑动, 并保证右/下边界被覆盖(追加贴边窗).
    """

    def starts(size: int) -> List[int]:
        if size <= patch:
            return [0]
        xs = list(range(0, size - patch + 1, stride))
        if xs[-1] != size - patch:
            xs.append(size - patch)
        return xs

    return [(x, y, min(x + patch, w), min(y + patch, h))
            for y in starts(h) for x in starts(w)]


# ---------------- 图像 ----------------
def open_image(path: Path):
    """打开图像并统一为 RGB."""
    require_pillow()
    img = Image.open(path)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def crop_patch(img, window, patch: int, save_path: Path, quality: int = 90) -> None:
    """裁剪窗口并 letterbox 到 patch x patch(左上对齐, 右/下补黑)后保存."""
    require_pillow()
    x0, y0, x1, y1 = (int(v) for v in window)
    x1, y1 = min(x1, img.width), min(y1, img.height)
    crop = img.crop((x0, y0, x1, y1))
    if crop.size != (patch, patch):
        canvas = Image.new("RGB", (patch, patch), 0)
        canvas.paste(crop, (0, 0))
        crop = canvas
    ensure_dir(Path(save_path).parent)
    crop.save(save_path, quality=quality)


# ---------------- 其它 ----------------
def guess_scene(path: Path) -> Optional[str]:
    """从父目录名/文件名推断 inshore/offshore."""
    text = f"{path.parent.name}/{path.name}".lower()
    if "inshore" in text:
        return "inshore"
    if "offshore" in text:
        return "offshore"
    return None


def find_image(stem_dir: Path, stem: str) -> Optional[Path]:
    """在目录中按主干名尝试各扩展名."""
    for ext in IMG_EXTS:
        p = stem_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def clip_box(x0, y0, x1, y1, w: int, h: int):
    """框裁剪到图像边界; 退化(空/负)返回 None."""
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)
    if x1 - x0 <= 1 or y1 - y0 <= 1:
        return None
    return x0, y0, x1, y1


def die(msg: str) -> None:
    raise SystemExit(f"[错误] {msg}")