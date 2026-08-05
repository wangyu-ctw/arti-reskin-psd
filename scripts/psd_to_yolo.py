"""从标注 PSD 生成 YOLO 训练数据。

目录约定:
    <input_root>/<游戏名>/<文件夹名2>/xxx.psd

规则:
    - 只处理名字为 text/icon/assets/button/bar/panel(不区分大小写)的图层组;
    - bbox 来自组内的每一个"叶子图层"(像素图层),逐层一条标注;
      图层组自身的 bbox 不使用——组的外接矩形会把分散元素框成一大块;
    - 嵌套组会继续下钻;嵌套组的名字若匹配另一个类别,则其内部图层按新类别计;
    - 不可见图层跳过(合成图里不存在的东西不能进标注);
    - 空 bbox(纯调整层等)跳过,越界部分裁剪到画布内。

输出(YOLO 标准结构):
    <output>/images/<游戏名>_<PSD名>.png    PSD 合成图(训练输入)
    <output>/labels/<游戏名>_<PSD名>.txt    每行: class cx cy w h(归一化)
    <output>/dataset.yaml                    类别表
    结束时打印逐类实例统计。

用法:
    python scripts/psd_to_yolo.py                          # 用默认游戏列表
    python scripts/psd_to_yolo.py --games dragon,binan     # 只跑指定游戏
"""
import argparse
from collections import Counter
from pathlib import Path

from psd_tools import PSDImage

# 类别顺序与检测流水线保持一致
CLASS_NAMES = ["text", "icon", "assets", "button", "bar", "panel"]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}

DEFAULT_GAMES = ["ansatsu", "binan", "highschool", "dragon", "arifure", "kingofprism"]


def norm_name(name: str) -> str:
    return (name or "").strip().lower()


def collect_boxes(node, active_class, canvas_w, canvas_h, out):
    """递归遍历图层树,active_class 是当前所处类别组;叶子图层产出 bbox。"""
    for layer in node:
        if not layer.is_visible():
            continue
        if layer.is_group():
            child_class = CLASS_IDS.get(norm_name(layer.name), active_class)
            collect_boxes(layer, child_class, canvas_w, canvas_h, out)
            continue
        if active_class is None:
            continue  # 不在六类组内的散层,不产出标注
        left, top, right, bottom = layer.bbox
        # 裁剪到画布内
        left, top = max(0, left), max(0, top)
        right, bottom = min(canvas_w, right), min(canvas_h, bottom)
        if right - left <= 0 or bottom - top <= 0:
            continue
        cx = (left + right) / 2 / canvas_w
        cy = (top + bottom) / 2 / canvas_h
        w = (right - left) / canvas_w
        h = (bottom - top) / canvas_h
        out.append((active_class, cx, cy, w, h))


def process_psd(psd_path: Path, out_images: Path, out_labels: Path,
                stem: str, stats: Counter) -> int:
    psd = PSDImage.open(psd_path)
    boxes = []
    collect_boxes(psd, None, psd.width, psd.height, boxes)
    if not boxes:
        print(f"跳过(无六类图层组标注): {psd_path}")
        return 0

    image = psd.composite()
    if image is None:
        print(f"跳过(合成失败): {psd_path}")
        return 0
    image.convert("RGB").save(out_images / f"{stem}.png")

    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
             for cls, cx, cy, w, h in boxes]
    (out_labels / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for cls, *_ in boxes:
        stats[CLASS_NAMES[cls]] += 1
    return len(boxes)


def main() -> None:
    parser = argparse.ArgumentParser(description="标注 PSD -> YOLO 训练数据")
    parser.add_argument("--input-root", type=Path,
                        default=Path.home() / "Desktop" / "标注")
    parser.add_argument("--games", default=",".join(DEFAULT_GAMES),
                        help="逗号分隔的游戏目录名")
    parser.add_argument("--output", type=Path,
                        default=Path.home() / "Desktop" / "标注" / "bbox")
    args = parser.parse_args()

    out_images = args.output / "images"
    out_labels = args.output / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    used_stems: set = set()
    total_images = 0

    for game in [g.strip() for g in args.games.split(",") if g.strip()]:
        game_dir = args.input_root / game
        if not game_dir.is_dir():
            print(f"警告: 游戏目录不存在,跳过 {game_dir}")
            continue
        psd_files = sorted(game_dir.rglob("*.psd")) + sorted(game_dir.rglob("*.PSD"))
        print(f"===== {game}: {len(psd_files)} 个 PSD =====")
        for psd_path in psd_files:
            stem = f"{game}_{psd_path.stem}"
            # 同名兜底(同游戏不同子目录重名时追加序号)
            base, n = stem, 2
            while stem in used_stems:
                stem = f"{base}_{n}"
                n += 1
            used_stems.add(stem)
            try:
                count = process_psd(psd_path, out_images, out_labels, stem, stats)
                if count:
                    total_images += 1
                    print(f"  {psd_path.name}: {count} 条标注 -> {stem}")
            except Exception as e:
                print(f"  处理失败: {psd_path} - {e}")

    # dataset.yaml(train/val 建议按游戏划分,这里先给全量)
    yaml_text = (
        f"path: {args.output}\n"
        "train: images\n"
        "val: images  # TODO: 建议按游戏切分验证集,避免同游戏图片同时出现在训练和验证\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (args.output / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    print("\n===== 完成 =====")
    print(f"图片数: {total_images}, 输出: {args.output}")
    print("逐类实例统计:")
    for name in CLASS_NAMES:
        print(f"  {name:8s} {stats.get(name, 0)}")


if __name__ == "__main__":
    main()
