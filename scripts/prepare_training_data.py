"""从标注 PSD 单遍生成四套训练数据。

输入目录约定: <input_root>/<游戏名>/**/*.psd,图层组按 text/icon/assets/button/bar/panel 命名。

输出(<output>/ 下):
  text_back/   {game}_{stem}_origin.png + {game}_{stem}_no_text.png     OmniPSD 去字训练对
  icon_back/   {game}_{stem}_no_text.png + {game}_{stem}_no_text_icon.png  去icon LoRA 训练对
  yolo/        images/ labels/ dataset.yaml                              六类检测(与 psd_to_yolo 同规范)
  sam2/        images/ masks/{name}/{idx:03d}_{class}.png annotations.jsonl 逐实例二值 mask

复用 output_layers.render_layers/make_opaque 做渲染;bbox 规范与 psd_to_yolo 一致。
每个 PSD 只做 3 次全图合成(origin / no_text / no_text_icon),四套数据共享。

用法:
  python scripts/prepare_training_data.py                       # 默认全部游戏
  python scripts/prepare_training_data.py --games dragon,binan
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_layers import make_opaque, render_layers  # noqa: E402

CLASS_NAMES = ["text", "icon", "assets", "button", "bar", "panel"]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}


def norm(name: str) -> str:
    return (name or "").strip().lower()


def class_group_names(psd, wanted: set) -> list:
    """顶层图层组里,规范名命中 wanted 的真实名字(render_layers 按真实名匹配)。"""
    return [layer.name for layer in psd if layer.is_group() and norm(layer.name) in wanted]


def collect_instances(node, active_class, out):
    """递归收集六类组内的叶子图层:(class_id, layer)。语义与 psd_to_yolo 一致。"""
    for layer in node:
        if not layer.is_visible():
            continue
        if layer.is_group():
            child = CLASS_IDS.get(norm(layer.name), active_class)
            collect_instances(layer, child, out)
            continue
        if active_class is None:
            continue
        out.append((active_class, layer))


def layer_bbox_yolo(layer, W, H):
    left, top, right, bottom = layer.bbox
    left, top = max(0, left), max(0, top)
    right, bottom = min(W, right), min(H, bottom)
    if right - left <= 0 or bottom - top <= 0:
        return None
    return ((left + right) / 2 / W, (top + bottom) / 2 / H,
            (right - left) / W, (bottom - top) / H)


def layer_mask(layer, W, H):
    """全画布尺寸的二值 mask(255=该图层实体像素)。"""
    from PIL import Image
    img = layer.composite()
    if img is None:
        return None
    alpha = img.getchannel("A") if "A" in img.getbands() else None
    if alpha is None:
        alpha = Image.new("L", img.size, 255)
    canvas = Image.new("L", (W, H), 0)
    left, top = layer.bbox[0], layer.bbox[1]
    canvas.paste(alpha.point(lambda v: 255 if v > 0 else 0), (left, top))
    return canvas


def process_psd(psd_path: Path, stem: str, dirs: dict, stats: Counter, ann_file) -> None:
    psd = PSDImage.open(psd_path)
    W, H = psd.width, psd.height

    # ---- 三次全图渲染,四套数据共享 ----
    origin = psd.composite().convert("RGB")
    text_names = class_group_names(psd, {"text"})
    text_icon_names = class_group_names(psd, {"text", "icon"})
    no_text = make_opaque(render_layers(psd, exclude_names=text_names or ["text"])).convert("RGB")
    no_text_icon = make_opaque(
        render_layers(psd, exclude_names=text_icon_names or ["text", "icon"])).convert("RGB")

    # 1. text_back 训练对
    origin_path = dirs["text_back"] / f"{stem}_origin.png"
    no_text_path = dirs["text_back"] / f"{stem}_no_text.png"
    origin.save(origin_path)
    no_text.save(no_text_path)

    # 4. icon_back 训练对(no_text 复用字节拷贝,不重编码)
    shutil.copyfile(no_text_path, dirs["icon_back"] / f"{stem}_no_text.png")
    no_text_icon.save(dirs["icon_back"] / f"{stem}_no_text_icon.png")

    # 2/3. 实例收集(bbox + mask)
    instances = []
    collect_instances(psd, None, instances)

    yolo_img = dirs["yolo"] / "images" / f"{stem}.png"
    shutil.copyfile(origin_path, yolo_img)
    shutil.copyfile(origin_path, dirs["sam2"] / "images" / f"{stem}.png")

    mask_dir = dirs["sam2"] / "masks" / stem
    mask_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx, (cls, layer) in enumerate(instances):
        bbox = layer_bbox_yolo(layer, W, H)
        if bbox is None:
            continue
        cx, cy, w, h = bbox
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        stats[CLASS_NAMES[cls]] += 1

        mask = layer_mask(layer, W, H)
        if mask is None:
            continue
        mask_path = mask_dir / f"{idx:03d}_{CLASS_NAMES[cls]}.png"
        mask.save(mask_path, optimize=True)
        ann_file.write(json.dumps({
            "image": f"images/{stem}.png",
            "mask": f"masks/{stem}/{mask_path.name}",
            "class": CLASS_NAMES[cls],
            "bbox": [round(v, 6) for v in (cx, cy, w, h)],
        }, ensure_ascii=False) + "\n")

    (dirs["yolo"] / "labels" / f"{stem}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="标注 PSD -> 四套训练数据(单遍)")
    parser.add_argument("--input-root", type=Path, default=Path.home() / "Desktop" / "标注")
    parser.add_argument("--games", default="", help="逗号分隔;留空自动发现含 PSD 的游戏目录")
    parser.add_argument("--output", type=Path, default=Path.home() / "Desktop" / "训练数据")
    args = parser.parse_args()

    if args.games:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
    else:
        games = sorted(d.name for d in args.input_root.iterdir()
                       if d.is_dir() and d.name not in {"bbox"} and any(d.rglob("*.psd")))

    dirs = {
        "text_back": args.output / "text_back",
        "icon_back": args.output / "icon_back",
        "yolo": args.output / "yolo",
        "sam2": args.output / "sam2",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    (dirs["yolo"] / "images").mkdir(exist_ok=True)
    (dirs["yolo"] / "labels").mkdir(exist_ok=True)
    (dirs["sam2"] / "images").mkdir(exist_ok=True)
    (dirs["sam2"] / "masks").mkdir(exist_ok=True)

    stats: Counter = Counter()
    used_stems: set = set()
    done, failed = 0, []

    with open(dirs["sam2"] / "annotations.jsonl", "w", encoding="utf-8") as ann_file:
        for game in games:
            psd_files = sorted((args.input_root / game).rglob("*.psd"))
            print(f"===== {game}: {len(psd_files)} 个 PSD =====", flush=True)
            for psd_path in psd_files:
                stem, n = f"{game}_{psd_path.stem}", 2
                base = stem
                while stem in used_stems:
                    stem = f"{base}_{n}"
                    n += 1
                used_stems.add(stem)
                try:
                    process_psd(psd_path, stem, dirs, stats, ann_file)
                    done += 1
                    print(f"  [{done}] {psd_path.name} OK", flush=True)
                except Exception as e:
                    failed.append((str(psd_path), str(e)))
                    print(f"  处理失败: {psd_path} - {e}", flush=True)

    yaml_text = (
        f"path: {dirs['yolo']}\n"
        "train: images\n"
        "val: images  # 训练前用 train_yolo.sh 按游戏切分\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    (dirs["yolo"] / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    print("\n===== 完成 =====", flush=True)
    print(f"成功 {done} 个 PSD,失败 {len(failed)} 个", flush=True)
    for p, e in failed:
        print(f"  失败: {p} - {e}", flush=True)
    print("逐类实例统计:", flush=True)
    for name in CLASS_NAMES:
        print(f"  {name:8s} {stats.get(name, 0)}", flush=True)


if __name__ == "__main__":
    main()
