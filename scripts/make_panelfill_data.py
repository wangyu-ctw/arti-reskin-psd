"""为 panel_fill(panel 层互叠修补)训练准备数据。

输入: ~/Desktop/标注/<游戏名>/**/*.psd。
取"视觉最底"的 panel 图层组(两个 panel 组时取迭代序第一个 = 被压在最下面的),
其它图层组全部无视(等效隐藏)。

组内每个直接子图层/子组视为一个 panel 单元,自顶向下逐个处理:
  - 与它下方的任一 panel 无 bbox 重叠 → 跳过;
  - 有重叠 → 以它的 alpha 为 mask,对"它下方所有 panel 的合成图"挖洞:
      {game}_{stem}_{k}_mid.png        修补示例图(下方合成,不透明黑底)
      {game}_{stem}_{k}_mid_hole.png   破洞图(透明洞)
      {game}_{stem}_{k}_hole_mask.png  洞 mask(白=洞)
      {game}_{stem}_{k}_base_cov.png   下方合成的覆盖 mask(切块过滤"假黑"用)
  - 处理完视为隐藏,继续向下(实现上 = 每个 panel 只合成其下方兄弟,天然等效)。

与 mid_fill 数据同构,可直接复用 make_fill_crops.py 切块:
  python scripts/make_fill_crops.py --data ~/Desktop/训练数据/panel_fill \
      --out ~/Desktop/训练数据/panel_fill_crops --size 512 \
      --target-suffix _mid.png --coverage-suffix _base_cov.png \
      --min-coverage 0.6 --min-hole 0.05 --max-hole 0.9 --grid

用法:
  python scripts/make_panelfill_data.py [--games dragon,binan] [--grow 0]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_layers import make_opaque  # noqa: E402


def norm(name: str) -> str:
    return (name or "").strip().lower()


def unit_alpha(layer, size) -> Image.Image:
    """panel 单元在全画布上的二值 alpha;无实体时返回 None。"""
    img = layer.composite()
    if img is None:
        return None
    alpha = (img.getchannel("A") if "A" in img.getbands()
             else Image.new("L", img.size, 255))
    canvas = Image.new("L", size, 0)
    canvas.paste(alpha.point(lambda v: 255 if v > 0 else 0),
                 (layer.bbox[0], layer.bbox[1]))
    return canvas


def bbox_overlap(a, b, tol: int = 2) -> bool:
    return (a[0] + tol < b[2] and a[2] - tol > b[0]
            and a[1] + tol < b[3] and a[3] - tol > b[1])


def process_psd(psd_path: Path, game: str, out_dir: Path,
                grow: int, stats: Counter) -> None:
    psd = PSDImage.open(psd_path)
    size = (psd.width, psd.height)
    stem = f"{game}_{psd_path.stem}"

    panel_groups = [l for l in psd
                    if l.is_group() and norm(l.name) == "panel"]
    if not panel_groups:
        stats["skip_no_panel_group"] += 1
        return
    # 迭代序自底向上 → 第一个 = 视觉最底的 panel 组
    group = panel_groups[0]

    # 组内直接子图层/子组 = panel 单元(自底向上)
    units = []
    for child in group:
        if not child.is_visible():
            continue
        alpha = unit_alpha(child, size)
        if alpha is None or not alpha.getbbox():
            continue
        img = child.composite()
        units.append({
            "layer": child,
            "alpha": alpha,
            "bbox": child.bbox,  # (l, t, r, b)
            "img": img,
            "pos": (child.bbox[0], child.bbox[1]),
        })
    if len(units) < 2:
        stats["skip_lt2_units"] += 1
        return

    k = 0
    # 自顶向下:迭代序末尾 = 视觉最上
    for i in range(len(units) - 1, 0, -1):
        top = units[i]
        below = units[:i]
        if not any(bbox_overlap(top["bbox"], b["bbox"]) for b in below):
            continue  # 无叠放,跳过(其下合成时它已天然不在)

        # 它下方所有 panel 的合成(自底向上叠)
        base_rgba = Image.new("RGBA", size, (0, 0, 0, 0))
        for b in below:
            base_rgba.alpha_composite(b["img"], b["pos"])
        coverage = base_rgba.getchannel("A").point(
            lambda v: 255 if v > 0 else 0)

        hole_mask = top["alpha"]
        if grow > 0:
            hole_mask = hole_mask.filter(ImageFilter.MaxFilter(2 * grow + 1))
        # 洞必须真的打在下方内容上,否则是无效样本
        effective = Image.composite(
            coverage, Image.new("L", size, 0), hole_mask)
        if not effective.getbbox():
            stats["skip_hole_off_base"] += 1
            continue

        target = make_opaque(base_rgba).convert("RGB")
        holed = target.convert("RGBA")
        holed.putalpha(hole_mask.point(lambda v: 0 if v > 0 else 255))

        name = f"{stem}_{k}"
        target.save(out_dir / f"{name}_mid.png")
        holed.save(out_dir / f"{name}_mid_hole.png")
        hole_mask.save(out_dir / f"{name}_hole_mask.png")
        coverage.save(out_dir / f"{name}_base_cov.png")
        k += 1
        stats["samples"] += 1
    if k:
        stats["psd_with_samples"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path.home() / "Desktop" / "标注")
    parser.add_argument("--output", type=Path,
                        default=Path.home() / "Desktop" / "训练数据")
    parser.add_argument("--games", type=str, default="")
    parser.add_argument("--grow", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.output / "panel_fill"
    out_dir.mkdir(parents=True, exist_ok=True)

    games = ([g.strip() for g in args.games.split(",") if g.strip()]
             or sorted(p.name for p in args.input_root.iterdir()
                       if p.is_dir()))

    stats = Counter()
    for game in games:
        psds = sorted((args.input_root / game).rglob("*.psd"))
        print(f"[{game}] {len(psds)} 个 PSD", flush=True)
        for psd_path in psds:
            try:
                process_psd(psd_path, game, out_dir, args.grow, stats)
            except Exception as e:
                stats["error"] += 1
                print(f"  [错误] {psd_path.name}: {e}")

    print("\n==== 汇总 ====")
    for key, v in sorted(stats.items()):
        print(f"  {key}: {v}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
