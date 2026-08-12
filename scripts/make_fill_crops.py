"""为 FLUX.1-Fill LoRA 训练切局部训练块(在 pod 上运行)。

输入(prepare_training_data.py + make_iconback_holes.py 的产出,已上传 pod):
  <data>/icon_back/{stem}_no_text_icon.png   目标完整图(洞外与 no_text 一致)
  <data>/icon_back/{stem}_hole_mask.png      洞 mask(白=icon 挖洞区)

输出:
  <out>/images/{stem}_{k}.png   512x512 局部图(来自 no_text_icon)
  <out>/masks/{stem}_{k}.png    同窗对齐的二值 mask

切块策略:洞的连通域逐个定位,以其中心开 512 方窗(贴边时夹回图内);
与已接受窗口中心距 <192px 的洞跳过(已被覆盖)。保证每块都含洞,
训练分布贴近推理时"局部补洞"的真实任务。

用法:
  /workspace/venvs/omnipsd-cu128/bin/python make_fill_crops.py \
      [--data /workspace/inputs/train_data_20260804/icon_back] \
      [--out  /workspace/inputs/train_data_20260804/fill_crops] [--size 512]
"""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def label_components(mask: np.ndarray):
    """返回每个连通域的中心 (cy, cx)。优先 scipy/cv2,兜底 BFS。"""
    try:
        from scipy import ndimage
        labels, n = ndimage.label(mask)
        return [tuple(map(float, c)) for c in
                ndimage.center_of_mass(mask, labels, range(1, n + 1))]
    except ImportError:
        pass
    try:
        import cv2
        n, labels = cv2.connectedComponents(mask.astype(np.uint8))
        return [tuple(map(float, np.argwhere(labels == i).mean(axis=0)))
                for i in range(1, n)]
    except ImportError:
        pass
    # 纯 numpy BFS 兜底
    visited = np.zeros_like(mask, dtype=bool)
    centers = []
    h, w = mask.shape
    for sy, sx in zip(*np.nonzero(mask & ~visited)):
        if visited[sy, sx]:
            continue
        ys, xs, q = [], [], deque([(sy, sx)])
        visited[sy, sx] = True
        while q:
            y, x = q.popleft()
            ys.append(y)
            xs.append(x)
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        centers.append((float(np.mean(ys)), float(np.mean(xs))))
    return centers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("/workspace/inputs/train_data_20260804/icon_back"))
    parser.add_argument("--out", type=Path,
                        default=Path("/workspace/inputs/train_data_20260804/fill_crops"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--dedupe-dist", type=int, default=192,
                        help="洞中心与已接受窗口中心的最小距离,小于则跳过")
    parser.add_argument("--target-suffix", type=str, default="_no_text_icon.png",
                        help="目标完整图后缀(mid_fill 数据传 _mid.png)")
    parser.add_argument("--min-hole", type=float, default=0.0,
                        help="窗口内洞占比下限,低于跳过(0=不过滤)")
    parser.add_argument("--max-hole", type=float, default=1.0,
                        help="窗口内洞占比上限,高于跳过(防整窗全洞无上下文)")
    parser.add_argument("--grid", action="store_true",
                        help="大洞(连通域包围盒大于窗口)沿包围盒按半窗步长多窗采样")
    parser.add_argument("--coverage-suffix", type=str, default="",
                        help="底图覆盖 mask 后缀(如 _base_cov.png);提供则按覆盖率过滤窗口")
    parser.add_argument("--min-coverage", type=float, default=0.6,
                        help="窗口内底图覆盖率下限(需配合 --coverage-suffix)")
    args = parser.parse_args()

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    half = args.size // 2

    total, skipped_imgs = 0, 0
    for mask_path in sorted(args.data.glob("*_hole_mask.png")):
        stem = mask_path.name[: -len("_hole_mask.png")]
        img_path = args.data / f"{stem}{args.target_suffix}"
        if not img_path.is_file():
            skipped_imgs += 1
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        if not mask.any():
            skipped_imgs += 1  # 无 icon 样本
            continue
        cov = None
        if args.coverage_suffix:
            cov_path = args.data / f"{stem}{args.coverage_suffix}"
            if cov_path.is_file():
                cov = np.asarray(Image.open(cov_path).convert("L")) > 127
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w < args.size or h < args.size:
            skipped_imgs += 1
            continue

        # 候选窗中心:连通域中心;--grid 时大洞再沿包围盒按半窗步长铺点
        centers = list(label_components(mask))
        if args.grid:
            ys, xs = np.nonzero(mask)
            # 逐连通域太慢时按整体 mask 包围盒近似即可(洞大且多时网格覆盖为主)
            if len(ys):
                y0, y1b = ys.min(), ys.max()
                x0, x1b = xs.min(), xs.max()
                step = half
                for gy in range(int(y0), int(y1b) + 1, step):
                    for gx in range(int(x0), int(x1b) + 1, step):
                        centers.append((float(gy), float(gx)))

        accepted = []
        k = 0
        for cy, cx in centers:
            if any(abs(cy - ay) < args.dedupe_dist and abs(cx - ax) < args.dedupe_dist
                   for ay, ax in accepted):
                continue
            x1 = int(min(max(cx - half, 0), w - args.size))
            y1 = int(min(max(cy - half, 0), h - args.size))
            win = mask[y1:y1 + args.size, x1:x1 + args.size]
            ratio = float(win.mean())
            if ratio < args.min_hole or ratio > args.max_hole:
                continue
            if cov is not None:
                cov_win = cov[y1:y1 + args.size, x1:x1 + args.size]
                if float(cov_win.mean()) < args.min_coverage:
                    continue  # 大面积无图层覆盖(压平后的假黑),不进训练集
            crop_img = img.crop((x1, y1, x1 + args.size, y1 + args.size))
            crop_mask = Image.fromarray((win * 255).astype(np.uint8))
            name = f"{stem}_{k:02d}.png"
            crop_img.save(args.out / "images" / name)
            crop_mask.save(args.out / "masks" / name, optimize=True)
            accepted.append((cy, cx))
            k += 1
            total += 1
        print(f"{stem}: {k} 块", flush=True)

    print(f"完成:共 {total} 块,跳过 {skipped_imgs} 张(无洞/过小)", flush=True)


if __name__ == "__main__":
    main()
