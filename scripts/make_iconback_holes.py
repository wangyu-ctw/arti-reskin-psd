"""为 icon_back(Fill 补洞范式)补生成洞图与洞 mask。

依赖 prepare_training_data.py 的产出:
  icon_back/{stem}_no_text.png          洞图的底
  sam2/masks/{stem}/NNN_icon.png        每个 icon 的图层级精确 mask

对每个 stem 生成:
  icon_back/{stem}_hole_mask.png        洞 mask(白=待补区域;icon mask 并集 -> 外扩 grow px -> 封闭内部孔)
  icon_back/{stem}_no_text_hole.png     no_text 上洞内涂标记色的图(给只吃图不吃 mask 的训练管线)

洞内标记色默认纯品红(255,0,255)——游戏 UI 里几乎不出现,模型可无歧义识别洞区;
黑/绿等常见色有与真实 UI 混淆的风险。--fill-color 可改(magenta/green/black 或 R,G,B)。
注意:若走"只吃洞图"的训练路线,推理时也必须用同一标记色挖洞,训练推理必须同色。

洞外扩默认按第 7 步提icon的三档参数逐 icon 计算(读 ui/src/config/stepDefaults.json 的
iconExtract:按 icon 像素长边分小/中/大档,grow = max(长边*paddingRatio, minPadding)),
与推理端提取 mask 的松紧同分布;--grow N 可强制统一外扩 N 像素(实验用)。

训练目标仍是已有的 {stem}_no_text_icon.png。

用法:
  python scripts/make_iconback_holes.py [--data ~/Desktop/训练数据] [--grow 5]
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

CONFIG_PATH = Path(__file__).resolve().parent.parent / "ui/src/config/stepDefaults.json"


def load_tiers():
    """读第 7 步提icon的三档外扩参数;读不到时用内置兜底值。"""
    try:
        ie = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["iconExtract"]
        return {
            "small_max": ie["smallMaxSide"],
            "large_min": ie["largeMinSide"],
            "small": (ie["small"]["paddingRatio"], ie["small"]["minPadding"]),
            "medium": (ie["medium"]["paddingRatio"], ie["medium"]["minPadding"]),
            "large": (ie["large"]["paddingRatio"], ie["large"]["minPadding"]),
        }
    except Exception:
        return {"small_max": 40, "large_min": 90,
                "small": (0.01, 1), "medium": (0.02, 2), "large": (0.03, 4)}


def grow_px_for(side: float, tiers: dict) -> int:
    if side <= tiers["small_max"]:
        ratio, min_pad = tiers["small"]
    elif side >= tiers["large_min"]:
        ratio, min_pad = tiers["large"]
    else:
        ratio, min_pad = tiers["medium"]
    return max(int(round(side * ratio)), int(min_pad), 1)


def dilate_mask(mask: Image.Image, grow: int) -> Image.Image:
    """对单个 mask 外扩 grow 像素(只在 bbox 邻域内做 MaxFilter,避免全图卷积)。"""
    box = mask.getbbox()
    if box is None or grow <= 0:
        return mask
    left, top, right, bottom = box
    left, top = max(0, left - grow - 1), max(0, top - grow - 1)
    right, bottom = min(mask.width, right + grow + 1), min(mask.height, bottom + grow + 1)
    region = mask.crop((left, top, right, bottom)).filter(
        ImageFilter.MaxFilter(size=2 * grow + 1))
    out = mask.copy()
    out.paste(region, (left, top))
    return out


def fill_enclosed_holes(mask: Image.Image) -> Image.Image:
    """封闭 mask 内部的孔(与推理端 fill_holes 同逻辑)。"""
    w, h = mask.size
    padded = ImageOps.expand(mask.point(lambda v: 255 if v > 0 else 0), border=1, fill=0)
    ImageDraw.floodfill(padded, (0, 0), 128)
    holes = padded.point(lambda v: 255 if v == 0 else 0).crop((1, 1, w + 1, h + 1))
    return ImageChops.lighter(mask, holes)


def main() -> None:
    parser = argparse.ArgumentParser(description="icon_back 洞图/洞 mask 生成")
    parser.add_argument("--data", type=Path, default=Path.home() / "Desktop" / "训练数据")
    parser.add_argument("--grow", type=int, default=0,
                        help="强制统一外扩 N 像素;默认 0 = 按第 7 步三档参数逐 icon 计算")
    parser.add_argument("--fill-color", default="magenta",
                        help="洞内标记色: magenta/green/black 或 'R,G,B',默认 magenta")
    args = parser.parse_args()

    named = {"magenta": (255, 0, 255), "green": (0, 255, 0), "black": (0, 0, 0)}
    if args.fill_color in named:
        fill_color = named[args.fill_color]
    else:
        fill_color = tuple(int(v) for v in args.fill_color.split(","))
        assert len(fill_color) == 3, "--fill-color 需为 R,G,B 三元组"

    icon_back = args.data / "icon_back"
    masks_root = args.data / "sam2" / "masks"
    tiers = load_tiers()
    print(f"分档外扩参数: {tiers}" if args.grow <= 0 else f"统一外扩 {args.grow}px", flush=True)

    done, no_icon, failed = 0, 0, []
    for no_text_path in sorted(icon_back.glob("*_no_text.png")):
        stem = no_text_path.name[: -len("_no_text.png")]
        try:
            icon_masks = sorted((masks_root / stem).glob("*_icon.png"))
            base = Image.open(no_text_path).convert("RGB")

            union = Image.new("L", base.size, 0)
            for mp in icon_masks:
                with Image.open(mp) as m:
                    mask = m.convert("L")
                if args.grow > 0:
                    grow = args.grow
                else:
                    box = mask.getbbox()
                    side = max(box[2] - box[0], box[3] - box[1]) if box else 0
                    grow = grow_px_for(side, tiers)
                union = ImageChops.lighter(union, dilate_mask(mask, grow))
            union = fill_enclosed_holes(union)

            union.save(icon_back / f"{stem}_hole_mask.png", optimize=True)
            hole_img = base.copy()
            hole_img.paste(fill_color, mask=union)
            hole_img.save(icon_back / f"{stem}_no_text_hole.png")

            if not icon_masks:
                no_icon += 1  # 没有 icon 的图:mask 全黑、洞图=原图,保留以维持数据对齐
            done += 1
            if done % 25 == 0:
                print(f"  已处理 {done} ...", flush=True)
        except Exception as e:
            failed.append((stem, str(e)))
            print(f"  失败: {stem} - {e}", flush=True)

    print(f"完成: {done} 个(其中无 icon 的 {no_icon} 个),失败 {len(failed)} 个", flush=True)
    for s, e in failed:
        print(f"  失败: {s} - {e}", flush=True)


if __name__ == "__main__":
    main()
