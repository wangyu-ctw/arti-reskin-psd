"""为"只分层 panel"的 Qwen-Image-Layered 微调准备数据。

每个 PSD 一个样本:
  composite.png            条件输入 = bg + panel 组的合成(≈管线的 mid_fill 分布)
  layer_00_bg.png          bg 层(全画布 RGBA)
  layer_01..0K_panel.png   panel 按 z 分层(互相重叠的进不同层,层内互不重叠;
                           amodal 完整——直接取 PSD 图层,被遮挡部分天然存在)
  meta.json                {levels, captions, recompose_error}

z 分层规则(与前端 16/17 步口径一致):按 PSD 堆叠序自底向上,
level_i = 与已分配 panel 有重叠者的最大 level + 1,无重叠为 0。

质检:bg + 各层 alpha 依序叠回 与 直接合成图 逐像素比,误差超阈值剔除。

用法:
  python scripts/make_layered_panel_data.py [--val-game ansatsu]
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


def norm(name: str) -> str:
    return (name or "").strip().lower()


def composite_groups(psd, names: set) -> Image.Image:
    """按顶层组名合成(自底向上,保持原位)。"""
    result = Image.new("RGBA", (psd.width, psd.height), (0, 0, 0, 0))
    for layer in psd:
        if not layer.is_visible() or norm(layer.name) not in names:
            continue
        img = layer.composite()
        if img is None:
            continue
        result.alpha_composite(img, (layer.bbox[0], layer.bbox[1]))
    return result


def pick_panel_group(psd):
    """可能有多个 panel 图层组:取 PS 图层列表里**最后一个**(= 堆叠序最底,
    psd-tools 自底向上遍历遇到的第一个)。上面的组是前景夹层(panel_f 类),
    不属于本任务的分层对象。"""
    for g in psd:
        if g.is_group() and g.is_visible() and norm(g.name) == "panel":
            return g
    return None


def collect_panels(group) -> list:
    """指定 panel 组的**叶子图层**,按全局堆叠序(底→顶)。

    直接子节点可能还是嵌套组——整组当一个单元会把组内互相堆叠的
    panel 合在同一层;必须递归到叶子逐层计算。
    """
    def leaves(node):
        for child in node:
            if not child.is_visible():
                continue
            if child.is_group():
                yield from leaves(child)
            else:
                yield child

    out = []
    for leaf in leaves(group):
        img = leaf.composite()
        if img is None:
            continue
        left, top, right, bottom = leaf.bbox
        if right - left < 4 or bottom - top < 4:
            continue
        out.append((leaf.bbox, img))
    return out


def assign_levels(panels, tol=3) -> list:
    """自顶向下分层(用户口径):从 PS 图层列表第一个叶子(堆叠最顶)开始
    往下找,与上方已处理图层不搭边的浮进尽量高的层;被压住的往下沉一层。
    返回每个 panel 的 level(0=最底,越大越上层),层内互不重叠。"""
    def overlap(a, b):
        return (a[0] < b[2] - tol and b[0] < a[2] - tol and
                a[1] < b[3] - tol and b[1] < a[3] - tol)

    n = len(panels)
    # topdown:depth=距顶层的深度(0=顶)
    depth = [0] * n
    order = list(range(n - 1, -1, -1))  # 堆叠序反转 = PS 列表自上而下
    for pos, i in enumerate(order):
        d = 0
        for j in order[:pos]:  # j 在 i 上方
            if overlap(panels[i][0], panels[j][0]):
                d = max(d, depth[j] + 1)
        depth[i] = d
    top = max(depth)
    return [top - d for d in depth]  # 转成 0=最底 的 level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path.home() / "Desktop" / "标注")
    parser.add_argument("--output", type=Path,
                        default=Path.home() / "Desktop" / "训练数据")
    parser.add_argument("--val-game", type=str, default="ansatsu")
    parser.add_argument("--max-recompose-err", type=float, default=2.0,
                        help="叠回误差均值阈值(0~255),超过剔除")
    parser.add_argument("--only", type=Path, default=None,
                        help="只重跑清单里的样本(txt,每行一个 <game>_<psd名> stem)")
    parser.add_argument("--name", type=str, default="layered_panel",
                        help="输出目录名(写到 <output>/<name>,便于多版本并存不覆盖)")
    args = parser.parse_args()
    only = None
    if args.only:
        only = {ln.strip() for ln in args.only.read_text(encoding="utf-8").splitlines() if ln.strip()}

    out_root = args.output / args.name
    stats = Counter()
    level_hist = Counter()

    games = sorted(p.name for p in args.input_root.iterdir() if p.is_dir())
    for game in games:
        split = "val" if game == args.val_game else "train"
        for psd_path in sorted((args.input_root / game).rglob("*.psd")):
            if only is not None and f"{game}_{psd_path.stem}" not in only:
                continue
            try:
                psd = PSDImage.open(psd_path)
                group = pick_panel_group(psd)
                panels = collect_panels(group) if group is not None else []
                if not panels:
                    stats["skip_no_panel"] += 1
                    continue
                levels = assign_levels(panels)
                n_levels = max(levels) + 1
                level_hist[n_levels] += 1

                W, H = psd.width, psd.height
                bg = composite_groups(psd, {"bg"})
                # 条件输入 = bg + 选中的 panel 组(上层 panel 组是 panel_f
                # 前景夹层,管线在 mid_fill 之前已移除,不进条件图)
                comp = bg.copy()
                gimg = group.composite()
                if gimg is not None:
                    comp.alpha_composite(gimg, (group.bbox[0], group.bbox[1]))

                # 逐层合成(amodal:直接取 PSD 子图层)
                layer_imgs = []
                for lv in range(n_levels):
                    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    for (bbox, img), l in zip(panels, levels):
                        if l == lv:
                            canvas.alpha_composite(img, (bbox[0], bbox[1]))
                    layer_imgs.append(canvas)

                # 质检:bg + 各层叠回 vs 直接合成
                recomp = bg.copy()
                for li in layer_imgs:
                    recomp.alpha_composite(li)
                a = np.asarray(make_opaque(recomp).convert("RGB"), dtype=np.int16)
                b = np.asarray(make_opaque(comp).convert("RGB"), dtype=np.int16)
                err = float(np.abs(a - b).mean())
                if err > args.max_recompose_err:
                    stats["skip_recompose_err"] += 1
                    continue

                stem = f"{game}_{psd_path.stem}"
                sample_dir = out_root / split / stem
                sample_dir.mkdir(parents=True, exist_ok=True)
                make_opaque(comp).convert("RGB").save(sample_dir / "composite.png")
                bg.save(sample_dir / "layer_00_bg.png")
                captions = ["背景层(bg,面板之下的一切)"]
                for lv, li in enumerate(layer_imgs):
                    li.save(sample_dir / f"layer_{lv + 1:02d}_panel.png")
                    captions.append(
                        f"面板层{lv + 1}(z={lv},自底向上第 {lv + 1} 层,层内互不重叠)")
                (sample_dir / "meta.json").write_text(json.dumps({
                    "levels": n_levels,
                    "panels": len(panels),
                    "captions": captions,
                    "recompose_error": round(err, 3),
                    "size": [W, H],
                }, ensure_ascii=False, indent=1), encoding="utf-8")
                stats[f"ok_{split}"] += 1
            except Exception as e:  # noqa: BLE001
                stats["error"] += 1
                print(f"  [错误] {psd_path.name}: {e}", flush=True)

    print("\n==== 汇总 ====")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print("  panel 层数分布(层数: 样本数):",
          dict(sorted(level_hist.items())))
    print(f"输出目录: {out_root}")


if __name__ == "__main__":
    main()
