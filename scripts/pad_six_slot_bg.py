"""六槽数据紧急垫底(alpha 可观测性修复):

无 bg 组的样本,元素浮在黑 void 上——板子的 alpha 在输入里不可观测,
这批样本的 alpha 监督等于噪声,是"预测层被洗淡/乱加透明度"的根源之一。

修法(纯后处理,不动 PSD):bg 槽为空的样本,从"bg 覆盖率高的样本池"
随机抽一张真实场景图垫底(优先同游戏),少数样本垫纯色/渐变(教会
"纯色也是 bg 层"——纯黑 UI 推理时 bg 该留黑而不是留空);
composite.png 用垫底 + 现有六层重新合成,meta 同步更新。

用法:python scripts/pad_six_slot_bg.py [--root ~/Desktop/训练数据/six_slot_v3]
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image

SLOTS = ["bg", "panel", "controls", "assets", "panel_f", "icon", "text"]
SOLID_RATIO = 0.2  # 两成垫纯色/渐变
SOLIDS = [(0, 0, 0), (255, 255, 255), (24, 24, 32), (235, 235, 228)]


def make_gradient(w: int, h: int, rng: random.Random) -> Image.Image:
    a = tuple(rng.randint(0, 90) for _ in range(3))
    b = tuple(rng.randint(120, 235) for _ in range(3))
    img = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(1, h - 1)
        img.putpixel((0, y), tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)))
    return img.resize((w, h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=Path.home() / "Desktop" / "训练数据" / "six_slot_v3")
    ap.add_argument("--split", type=str, default="train")
    args = ap.parse_args()
    root = args.root / args.split

    samples = []
    donors = []  # (game, bg文件, size)
    for d in sorted(root.iterdir()):
        meta_p = d / "meta.json"
        if not d.is_dir() or not meta_p.is_file():
            continue
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        samples.append((d, m))
        if m["coverage"].get("bg", 0) > 0.5:
            donors.append((d.name.split("_")[0], d / "layer_00_bg.png"))
    assert donors, "没有可用的 bg 捐赠样本"

    n_pad = 0
    for d, m in samples:
        if m["present"].get("bg"):
            continue
        rng = random.Random(d.name)  # 按样本名定种,可复现
        W, H = m["size"]
        game = d.name.split("_")[0]
        if rng.random() < SOLID_RATIO:
            if rng.random() < 0.5:
                bg_rgb = Image.new("RGB", (W, H), rng.choice(SOLIDS))
                mode = "solid"
            else:
                bg_rgb = make_gradient(W, H, rng)
                mode = "gradient"
        else:
            pool = [p for g, p in donors if g == game] or [p for _, p in donors]
            donor = rng.choice(pool)
            with Image.open(donor) as im:
                bg_rgb = im.convert("RGB").resize((W, H), Image.LANCZOS)
            mode = f"donor:{donor.parent.name}"
        bg = bg_rgb.convert("RGBA")
        bg.save(d / "layer_00_bg.png")
        # composite 重建:垫底 + 现有六层依帧序叠合
        comp = bg.copy()
        for i, slot in enumerate(SLOTS[1:], start=1):
            with Image.open(d / f"layer_{i:02d}_{slot}.png") as im:
                comp.alpha_composite(im.convert("RGBA"))
        comp.convert("RGB").save(d / "composite.png")
        m["present"]["bg"] = True
        m["coverage"]["bg"] = 1.0
        m["bg_padded"] = mode
        (d / "meta.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        n_pad += 1
        print(f"  {d.name} <- {mode}", flush=True)
    print(f"垫底完成: {n_pad} 个样本(共 {len(samples)})")


if __name__ == "__main__":
    main()
