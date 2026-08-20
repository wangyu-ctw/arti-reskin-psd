"""为"六槽整图分解"的 Qwen-Image-Layered 微调准备数据。

每个 PSD 一个样本,固定 7 帧(空槽输出全透明层,帧序即训练帧序):
  composite.png            条件输入 = 全部可见顶层节点按 PSD 原堆叠序合成(≈原图)
  layer_00_bg.png          bg 组 + 一切未识别杂名节点(沉底)
  layer_01_panel.png       底板面板(v3 口径,见下)
  layer_02_controls.png    bar + button 组(合槽,槽内保持原堆叠序)
  layer_03_assets.png      assets 组
  layer_04_panel_f.png     浮层面板(v3 口径,见下)+ 其余 panel 组

panel / panel_f 分界(v3,按堆叠事实而非组归属):最底 panel 组的叶子做
自底向上紧凑分层——z0 → panel;z1+ 且被 controls/assets 内容压住(alpha
像素交叠)→ 降级回 panel(必须留在中景之下);其余 z1+ → panel_f;
panel 槽向下闭包(成员脚下重叠的叶子强制同槽,保证帧序叠回正确)。
上层 panel 组(在 assets 之上)照旧整组进 panel_f。
  layer_05_icon.png        icon 组
  layer_06_text.png        text 组
  meta.json                {slots, coverage, recompose_error, ...}

cover/遮罩/黑遮罩/黑框(全画布盖板,非换皮素材)从输入与输出一并剔除。
质检:七槽按帧序叠回 与 原堆叠序合成 逐像素比,均值误差超阈值剔除
(能抓住"实际堆叠序与帧序约定冲突"的样本),剔除清单写 rejected.txt。

用法:
  python scripts/make_six_slot_data.py                     # 全量(全部进 train)
  python scripts/make_six_slot_data.py --only binan_U132_clean_annotated ...
  python scripts/make_six_slot_data.py --only-file redo.txt
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from psd_tools import PSDImage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_layers import make_opaque  # noqa: E402
from PIL import Image  # noqa: E402

# 帧序(底→顶):与标注 PSD 的主流堆叠约定一致
SLOTS = ["bg", "panel", "controls", "assets", "panel_f", "icon", "text"]
CAPTIONS = {
    "bg": "背景层(bg 及一切非六类的底部内容)",
    "panel": "panel 层(堆叠最底的 panel 组,面板底板)",
    "controls": "controls 层(bar + button)",
    "assets": "assets 层",
    "panel_f": "panel_f 层(夹层面板,上层 panel 组;可为空)",
    "icon": "icon 层",
    "text": "text 层",
}
# 根节点下第一层(顶层节点)里的盖板类图层,输入输出一并剔除
COVER = {"cover", "遮罩", "黑遮罩", "黑框", "mask", "余白部分"}
ALIAS = {"asets": "assets", "panels": "panel"}  # 标注笔误


def norm(name: str) -> str:
    return (name or "").strip().lower()


def classify_nodes(psd):
    """顶层可见节点 → (slot, node) 列表(保持 PSD 底→顶堆叠序)+ 剔除名单。

    panel 组规则(用户口径):psd-tools 自底向上遍历遇到的第一个 panel 组
    (= PS 软件列表里最后一个)进 panel 槽,其余(上面的)全进 panel_f。
    未识别的杂名节点一律沉入 bg 槽,靠叠回质检拦截放错位置的。
    """
    entries, dropped = [], []
    seen_panel = False
    for layer in psd:
        if not layer.is_visible():
            continue
        n = ALIAS.get(norm(layer.name), norm(layer.name))
        if n in COVER:
            dropped.append(layer.name)
            continue
        if n == "panel":
            slot = "panel" if not seen_panel else "panel_f"
            seen_panel = True
        elif n in ("bar", "button"):
            slot = "controls"
        elif n in ("assets", "icon", "text", "bg"):
            slot = n if n != "bg" else "bg"
        else:
            slot = "bg"
        entries.append((slot, layer))
    return entries, dropped


def leaf_iter(node):
    for c in node:
        if not c.is_visible():
            continue
        if c.is_group():
            yield from leaf_iter(c)
        else:
            yield c


def paste_alpha(canvas, img, x, y):
    """img 的 alpha>8 布尔并入全画布 canvas(越界裁剪)。"""
    a = np.asarray(img.getchannel("A")) > 8
    h, w = a.shape
    H, W = canvas.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] |= a[y0 - y:y1 - y, x0 - x:x1 - x]


def split_bottom_panel(leaves, mid_alpha, tol=3):
    """最底 panel 组的叶子按 v3 口径分 panel/panel_f(见模块 docstring)。

    leaves: [(bbox, img)] 堆叠序底→顶;mid_alpha: 全画布 controls+assets 布尔。
    返回与 leaves 对齐的槽名列表。"""
    def overlap(a, b):
        return (a[0] < b[2] - tol and b[0] < a[2] - tol and
                a[1] < b[3] - tol and b[1] < a[3] - tol)

    levels = []
    for i, (bb, _) in enumerate(leaves):
        l = 0
        for j in range(i):
            if overlap(leaves[j][0], bb):
                l = max(l, levels[j] + 1)
        levels.append(l)

    H, W = mid_alpha.shape
    slots = []
    for (bb, img), lv in zip(leaves, levels):
        if lv == 0:
            slots.append("panel")
            continue
        a = np.asarray(img.getchannel("A")) > 8
        h, w = a.shape
        x, y = bb[0], bb[1]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        inter = 0
        if x1 > x0 and y1 > y0:
            inter = int((a[y0 - y:y1 - y, x0 - x:x1 - x] & mid_alpha[y0:y1, x0:x1]).sum())
        pressed = inter > max(32, 0.005 * a.sum())
        slots.append("panel" if pressed else "panel_f")
    # 向下闭包:panel 槽成员脚下重叠的叶子强制留在 panel 槽
    for i in range(len(leaves) - 1, -1, -1):
        if slots[i] == "panel":
            for j in range(i):
                if overlap(leaves[j][0], leaves[i][0]):
                    slots[j] = "panel"
    return slots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path.home() / "Desktop" / "标注")
    parser.add_argument("--output", type=Path,
                        default=Path.home() / "Desktop" / "训练数据")
    parser.add_argument("--name", type=str, default="six_slot",
                        help="输出目录名(<output>/<name>,多版本并存不覆盖)")
    parser.add_argument("--split", type=str, default="train",
                        help="输出到哪个 split 子目录(train/val)")
    parser.add_argument("--max-recompose-err", type=float, default=2.0,
                        help="叠回误差均值阈值(0~255),超过剔除")
    parser.add_argument("--only", nargs="*", default=None,
                        help="只重跑这些样本(<game>_<psd名> stem,可多个)")
    parser.add_argument("--only-file", type=Path, default=None,
                        help="只重跑清单文件里的样本(txt,每行一个 stem)")
    args = parser.parse_args()

    only = set(args.only or [])
    if args.only_file:
        only |= {ln.strip() for ln in
                 args.only_file.read_text(encoding="utf-8").splitlines()
                 if ln.strip()}
    only = only or None

    out_root = args.output / args.name
    stats = Counter()
    slot_present = Counter()
    rejected, no_panel = [], []

    # 常规:<input-root>/<游戏>/**.psd;平铺:PSD 直接在 input-root 下
    # (如 val 标注目录),游戏名取目录名
    units = [(p.name, sorted(p.rglob("*.psd")))
             for p in sorted(args.input_root.iterdir()) if p.is_dir()]
    root_psds = sorted(args.input_root.glob("*.psd"))
    if root_psds:
        units.append((args.input_root.name, root_psds))
    for game, psd_paths in units:
        print(f"== {game}", flush=True)
        for psd_path in psd_paths:
            stem = f"{game}_{psd_path.stem}"
            if only is not None and stem not in only:
                continue
            try:
                psd = PSDImage.open(psd_path)
                W, H = psd.width, psd.height
                entries, dropped = classify_nodes(psd)
                if not entries:
                    stats["skip_empty"] += 1
                    continue

                # 每个节点只 composite 一次;最底 panel 组延后按叶子拆(v3)
                rendered = []  # (slot, img, (x, y));占位 None = 最底 panel 组
                bottom_leaves, bottom_idx = None, None
                for slot, layer in entries:
                    if slot == "panel" and bottom_idx is None:
                        lvs = []
                        nodes = leaf_iter(layer) if layer.is_group() else [layer]
                        for leaf in nodes:
                            img = leaf.composite()
                            l, t, r, b = leaf.bbox
                            if img is None or r - l < 4 or b - t < 4:
                                continue
                            lvs.append((leaf.bbox, img))
                        bottom_leaves, bottom_idx = lvs, len(rendered)
                        rendered.append(None)
                        continue
                    img = layer.composite()
                    if img is None:
                        continue
                    rendered.append((slot, img, (layer.bbox[0], layer.bbox[1])))
                # 中景内容 alpha(压住判定用)
                mid_alpha = np.zeros((H, W), dtype=bool)
                for ent in rendered:
                    if ent is not None and ent[0] in ("controls", "assets"):
                        paste_alpha(mid_alpha, ent[1], ent[2][0], ent[2][1])
                n_v3 = {"panel": 0, "panel_f": 0}
                if bottom_idx is not None:
                    leaf_entries = []
                    if bottom_leaves:
                        v3_slots = split_bottom_panel(bottom_leaves, mid_alpha)
                        for (bb, img), s in zip(bottom_leaves, v3_slots):
                            leaf_entries.append((s, img, (bb[0], bb[1])))
                            n_v3[s] += 1
                    rendered = (rendered[:bottom_idx] + leaf_entries
                                + rendered[bottom_idx + 1:])
                rendered = [e for e in rendered if e is not None]

                # 参考图 = 原堆叠序合成(条件输入)
                ref = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                for _, img, pos in rendered:
                    ref.alpha_composite(img, pos)

                # 槽层(槽内保持原堆叠序)
                layer_imgs = {}
                for slot in SLOTS:
                    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    for s, img, pos in rendered:
                        if s == slot:
                            canvas.alpha_composite(img, pos)
                    layer_imgs[slot] = canvas

                # 质检:按帧序叠回 vs 原堆叠序
                recomp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                for slot in SLOTS:
                    recomp.alpha_composite(layer_imgs[slot])
                a = np.asarray(make_opaque(recomp).convert("RGB"), dtype=np.int16)
                b = np.asarray(make_opaque(ref).convert("RGB"), dtype=np.int16)
                err = float(np.abs(a - b).mean())
                if err > args.max_recompose_err:
                    stats["skip_recompose_err"] += 1
                    rejected.append(f"{stem}  err={err:.2f}")
                    continue

                alpha = {s: np.asarray(layer_imgs[s].getchannel("A")) for s in SLOTS}
                coverage = {s: round(float((alpha[s] > 8).mean()), 4) for s in SLOTS}
                present = {s: coverage[s] > 0 for s in SLOTS}
                for s in SLOTS:
                    if present[s]:
                        slot_present[s] += 1
                if not present["panel"]:
                    no_panel.append(stem)

                sample_dir = out_root / args.split / stem
                sample_dir.mkdir(parents=True, exist_ok=True)
                make_opaque(ref).convert("RGB").save(sample_dir / "composite.png")
                for i, slot in enumerate(SLOTS):
                    layer_imgs[slot].save(sample_dir / f"layer_{i:02d}_{slot}.png")
                (sample_dir / "meta.json").write_text(json.dumps({
                    "slots": SLOTS,
                    "captions": [CAPTIONS[s] for s in SLOTS],
                    "present": present,
                    "coverage": coverage,
                    "node_count": Counter(s for s, _ in entries),
                    "v3_bottom_split": n_v3,
                    "dropped_cover": dropped,
                    "recompose_error": round(err, 3),
                    "size": [W, H],
                }, ensure_ascii=False, indent=1), encoding="utf-8")
                stats["ok"] += 1
            except Exception as e:  # noqa: BLE001
                stats["error"] += 1
                print(f"  [错误] {psd_path.name}: {e}", flush=True)

    print("\n==== 汇总 ====")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print("  各槽非空样本数:", dict(slot_present))
    if no_panel:
        print(f"  panel 槽为空({len(no_panel)}): {no_panel}")
    if rejected:
        (out_root / "rejected.txt").write_text(
            "\n".join(rejected) + "\n", encoding="utf-8")
        print(f"  剔除清单已写 {out_root / 'rejected.txt'}({len(rejected)} 条)")
    print(f"输出目录: {out_root}")


if __name__ == "__main__":
    main()
