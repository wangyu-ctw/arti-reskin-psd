"""为 mid_fill(中景修补)训练准备数据。

输入目录约定: <input_root>/<游戏名>/**/*.psd,根节点图层组按
text/icon/assets/bar/button/panel/bg 命名(panel 可能有两个)。

每个 PSD:
  1. text/icon 图层组不参与(等效隐藏);
  2. 底图 = bg + panel 合成(若有两个 panel 组,视觉上层的那个不进底图);
  3. 洞 mask = assets/bar/button(+ 上层 panel 组,若存在)的 alpha 并集;
  4. 输出三件套(<output>/mid_fill/ 下):
     {game}_{stem}_mid.png        修补示例图(目标:底图合成,不透明)
     {game}_{stem}_mid_hole.png   破洞图(底图上洞区 alpha=0,RGBA)
     {game}_{stem}_hole_mask.png  洞 mask(白=待补区域)

"视觉上层的 panel 组":psd-tools 自底向上迭代,同名 panel 组取迭代序
最靠后者为上层(与画面叠放一致)。

用法:
  python scripts/make_midfill_data.py                       # 全部游戏
  python scripts/make_midfill_data.py --games dragon,binan
  python scripts/make_midfill_data.py --grow 4              # 洞统一外扩 4px
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_layers import make_opaque  # noqa: E402

HOLE_GROUPS = {"assets", "bar", "button"}


def norm(name: str) -> str:
    return (name or "").strip().lower()


def composite_groups(psd, groups) -> Image.Image:
    """按迭代序(自底向上)合成指定图层组,保持原位。"""
    result = Image.new("RGBA", (psd.width, psd.height), (0, 0, 0, 0))
    wanted = {id(g) for g in groups}
    for layer in psd:
        if id(layer) not in wanted or not layer.is_visible():
            continue
        img = layer.composite()
        if img is None:
            continue
        result.alpha_composite(img, (layer.bbox[0], layer.bbox[1]))
    return result


def groups_alpha_union(psd, groups, grow: int) -> Image.Image:
    """指定图层组的 alpha 并集(255=洞),可选统一外扩。"""
    mask = Image.new("L", (psd.width, psd.height), 0)
    for g in groups:
        if not g.is_visible():
            continue
        img = g.composite()
        if img is None:
            continue
        alpha = (img.getchannel("A") if "A" in img.getbands()
                 else Image.new("L", img.size, 255))
        binary = alpha.point(lambda v: 255 if v > 0 else 0)
        canvas = Image.new("L", mask.size, 0)
        canvas.paste(binary, (g.bbox[0], g.bbox[1]))
        mask = Image.composite(
            Image.new("L", mask.size, 255), mask, canvas)
    if grow > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * grow + 1))
    return mask


def process_psd(psd_path: Path, game: str, out_dir: Path,
                grow: int, stats: Counter) -> None:
    psd = PSDImage.open(psd_path)
    stem = f"{game}_{psd_path.stem}"

    groups = [layer for layer in psd if layer.is_group()]
    by_name = {}
    for g in groups:
        by_name.setdefault(norm(g.name), []).append(g)  # 迭代序=自底向上

    panels = by_name.get("panel", [])
    bgs = by_name.get("bg", [])
    if not panels and not bgs:
        stats["skip_no_base"] += 1
        print(f"  [跳过] {psd_path.name}: 没有 panel/bg 图层组")
        return

    # 两个 panel 组时,迭代序最后的 = 视觉上层 → 归入洞,不进底图
    upper_panel = panels[-1] if len(panels) >= 2 else None
    base_panels = panels[:-1] if len(panels) >= 2 else panels

    hole_groups = [g for name in HOLE_GROUPS for g in by_name.get(name, [])]
    if upper_panel is not None:
        hole_groups.append(upper_panel)
        stats["double_panel"] += 1
    if not hole_groups:
        stats["skip_no_hole"] += 1
        print(f"  [跳过] {psd_path.name}: 没有可挖洞的图层组")
        return

    # 底图(修补目标):bg + 下层 panel,合成到不透明黑底;
    # 覆盖 mask 单独存(黑底压平后无法区分"真黑 UI"和"无图层区域",
    # 切块时靠它过滤掉大面积无覆盖的窗口,防止模型学会"填黑")
    base_rgba = composite_groups(psd, bgs + base_panels)
    coverage = base_rgba.getchannel("A").point(lambda v: 255 if v > 0 else 0)
    base = make_opaque(base_rgba).convert("RGB")

    # 洞 mask 与破洞图
    hole_mask = groups_alpha_union(psd, hole_groups, grow)
    if not hole_mask.getbbox():
        stats["skip_empty_mask"] += 1
        print(f"  [跳过] {psd_path.name}: 洞 mask 为空")
        return
    holed = base.convert("RGBA")
    holed.putalpha(hole_mask.point(lambda v: 0 if v > 0 else 255))

    base.save(out_dir / f"{stem}_mid.png")
    coverage.save(out_dir / f"{stem}_base_cov.png")
    holed.save(out_dir / f"{stem}_mid_hole.png")
    hole_mask.save(out_dir / f"{stem}_hole_mask.png")
    stats["ok"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path.home() / "Desktop" / "标注")
    parser.add_argument("--output", type=Path,
                        default=Path.home() / "Desktop" / "训练数据")
    parser.add_argument("--games", type=str, default="",
                        help="逗号分隔的游戏名,默认全部")
    parser.add_argument("--grow", type=int, default=0,
                        help="洞统一外扩像素,默认 0(按图层 alpha 原样)")
    args = parser.parse_args()

    out_dir = args.output / "mid_fill"
    out_dir.mkdir(parents=True, exist_ok=True)

    games = ([g.strip() for g in args.games.split(",") if g.strip()]
             or sorted(p.name for p in args.input_root.iterdir() if p.is_dir()))

    stats = Counter()
    for game in games:
        game_dir = args.input_root / game
        psds = sorted(game_dir.rglob("*.psd"))
        print(f"[{game}] {len(psds)} 个 PSD")
        for psd_path in psds:
            try:
                process_psd(psd_path, game, out_dir, args.grow, stats)
            except Exception as e:  # 单文件失败不拖垮整批
                stats["error"] += 1
                print(f"  [错误] {psd_path.name}: {e}")

    print("\n==== 汇总 ====")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
