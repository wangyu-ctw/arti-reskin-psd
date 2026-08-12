"""为 panel 专项 YOLO(amodal 全貌检测)准备数据。

关键设计:标签用图层 bbox 真值——包含被上层元素遮挡的部分,
训练模型预测 panel 的"完整边界"而非可见边界。

每个 PSD 生成三种合成图,共享同一套标签(单类 panel):
  {stem}_full.png   全量合成(text 除外,近似去字图分布)
  {stem}_mid.png    bg + 全部 panel 组(mid_fill 分布,panel_extract 的输入)
  {stem}_stack.png  bg + panel + assets/bar/button(中间态分布)

标签:所有 panel 图层组(含上下两组)的每个直接子图层/子组,
bbox = layer.bbox(clamp 到画布),class 0。

输出 <output>/panel_yolo/:
  images/{train,val}/  labels/{train,val}/  dataset.yaml
验证集 = ansatsu 整游戏 holdout(与 SAM2 训练同约定)。

用法:
  python scripts/make_panel_yolo_data.py [--val-game ansatsu]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_layers import make_opaque  # noqa: E402


def norm(name: str) -> str:
    return (name or "").strip().lower()


def composite_by(psd, keep) -> "Image":
    """按谓词挑选顶层图层组合成(自底向上,保持原位)。"""
    from PIL import Image
    result = Image.new("RGBA", (psd.width, psd.height), (0, 0, 0, 0))
    for layer in psd:
        if not layer.is_visible() or not keep(layer):
            continue
        img = layer.composite()
        if img is None:
            continue
        result.alpha_composite(img, (layer.bbox[0], layer.bbox[1]))
    return result


def panel_labels(psd) -> list:
    """所有 panel 组的直接子图层 → 单类 YOLO 标签(amodal bbox)。"""
    W, H = psd.width, psd.height
    labels = []
    for g in psd:
        if not (g.is_group() and norm(g.name) == "panel"):
            continue
        for child in g:
            if not child.is_visible():
                continue
            left, top, right, bottom = child.bbox
            left, top = max(0, left), max(0, top)
            right, bottom = min(W, right), min(H, bottom)
            if right - left < 4 or bottom - top < 4:
                continue
            labels.append((
                (left + right) / 2 / W, (top + bottom) / 2 / H,
                (right - left) / W, (bottom - top) / H,
            ))
    return labels


VARIANTS = {
    "full": lambda l: norm(l.name) != "text",
    "mid": lambda l: norm(l.name) in ("panel", "bg"),
    "stack": lambda l: norm(l.name) in ("panel", "bg", "assets", "bar",
                                        "button"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path.home() / "Desktop" / "标注")
    parser.add_argument("--output", type=Path,
                        default=Path.home() / "Desktop" / "训练数据")
    parser.add_argument("--val-game", type=str, default="ansatsu")
    args = parser.parse_args()

    out = args.output / "panel_yolo"
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    games = sorted(p.name for p in args.input_root.iterdir() if p.is_dir())
    for game in games:
        split = "val" if game == args.val_game else "train"
        psds = sorted((args.input_root / game).rglob("*.psd"))
        print(f"[{game}] {len(psds)} 个 PSD → {split}", flush=True)
        for psd_path in psds:
            try:
                psd = PSDImage.open(psd_path)
                labels = panel_labels(psd)
                if not labels:
                    stats["skip_no_panel"] += 1
                    continue
                label_text = "\n".join(
                    f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    for cx, cy, w, h in labels) + "\n"
                stem = f"{game}_{psd_path.stem}"
                for var, keep in VARIANTS.items():
                    img = make_opaque(composite_by(psd, keep)).convert("RGB")
                    name = f"{stem}_{var}"
                    img.save(out / "images" / split / f"{name}.png")
                    (out / "labels" / split / f"{name}.txt").write_text(
                        label_text, encoding="utf-8")
                    stats[f"img_{split}"] += 1
                stats["psd_ok"] += 1
                stats["labels"] += len(labels)
            except Exception as e:
                stats["error"] += 1
                print(f"  [错误] {psd_path.name}: {e}")

    (out / "dataset.yaml").write_text(
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: panel\n", encoding="utf-8")

    print("\n==== 汇总 ====")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"输出目录: {out}")


if __name__ == "__main__":
    main()
