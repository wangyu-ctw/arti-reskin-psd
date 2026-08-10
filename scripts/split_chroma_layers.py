#!/usr/bin/env python3
"""切分五等分色键图、移除底色，并按 UI 图层顺序重新合成。

默认输入布局（从左到右）：
  1. reward_icons
  2. current_progress_indicator
  3. border
  4. progress_fill
  5. base_plate

默认合成顺序（从底到顶）：
  base_plate -> progress_fill -> border -> reward_icons
  -> current_progress_indicator

示例：
  python scripts/split_chroma_layers.py input.png --background '#00FF00'
  python scripts/split_chroma_layers.py input.png --background 0,255,0 \
      --output-dir output/bar_layers
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


DEFAULT_LAYER_NAMES = (
    "reward_icons",
    "current_progress_indicator",
    "border",
    "progress_fill",
    "base_plate",
)
DEFAULT_COMPOSITE_ORDER = (
    "base_plate",
    "progress_fill",
    "border",
    "reward_icons",
    "current_progress_indicator",
)


def parse_color(value: str) -> tuple[int, int, int]:
    """解析 #RRGGBB、RRGGBB 或 R,G,B。"""
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("颜色必须是 #RRGGBB 或 R,G,B")
        try:
            rgb = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("R,G,B 必须是整数") from exc
    else:
        if len(text) != 6:
            raise argparse.ArgumentTypeError("十六进制颜色必须是 #RRGGBB")
        try:
            rgb = tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("无效的十六进制颜色") from exc
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("颜色通道必须位于 0..255")
    return rgb  # type: ignore[return-value]


def black_column_scores(rgb: np.ndarray, threshold: int) -> np.ndarray:
    """返回每一列接近纯黑的像素占比。"""
    near_black = np.max(rgb, axis=2) <= threshold
    return near_black.mean(axis=0)


def estimate_actual_background(
    rgb: np.ndarray,
    requested: tuple[int, int, int],
    border_ratio: float = 0.035,
    candidate_distance: float = 120.0,
) -> tuple[int, int, int]:
    """从画面四边估计生图模型实际生成的底色。"""
    height, width = rgb.shape[:2]
    band_x = max(2, int(round(width * border_ratio)))
    band_y = max(2, int(round(height * border_ratio)))
    border = np.concatenate(
        (
            rgb[:band_y].reshape(-1, 3),
            rgb[-band_y:].reshape(-1, 3),
            rgb[:, :band_x].reshape(-1, 3),
            rgb[:, -band_x:].reshape(-1, 3),
        ),
        axis=0,
    ).astype(np.float32)
    target = np.asarray(requested, dtype=np.float32)
    distances = np.sqrt(np.sum((border - target) ** 2, axis=1))
    candidates = border[distances <= candidate_distance]
    minimum = max(64, int(border.shape[0] * 0.05))
    if candidates.shape[0] < minimum:
        return requested
    median = np.rint(np.median(candidates, axis=0)).astype(np.uint8)
    return tuple(int(channel) for channel in median)


def detect_separator_groups(
    rgb: np.ndarray,
    layer_count: int,
    black_threshold: int,
    min_coverage: float,
    search_ratio: float,
) -> list[tuple[int, int]]:
    """在理论等分边界附近检测贯穿全高的黑色分隔线。

    返回半开区间 ``[(start, end), ...]``。限制搜索范围是为了避免把
    图层内部的黑色竖向图案误判为分隔线。
    """
    width = rgb.shape[1]
    scores = black_column_scores(rgb, black_threshold)
    groups: list[tuple[int, int]] = []
    search_radius = max(4, int(round((width / layer_count) * search_ratio)))
    expand_threshold = max(0.35, min_coverage * 0.65)

    for boundary_index in range(1, layer_count):
        expected = int(round(width * boundary_index / layer_count))
        search_start = max(0, expected - search_radius)
        search_end = min(width, expected + search_radius + 1)
        local = scores[search_start:search_end]
        peak = search_start + int(np.argmax(local))
        peak_score = float(scores[peak])
        if peak_score < min_coverage:
            raise ValueError(
                f"未在第 {boundary_index} 个五等分边界附近检测到黑色分隔线："
                f"最高覆盖率 {peak_score:.3f} < {min_coverage:.3f}。"
                "可降低 --separator-min-coverage 或扩大 --separator-search-ratio。"
            )

        start = peak
        while start > search_start and scores[start - 1] >= expand_threshold:
            start -= 1
        end = peak + 1
        while end < search_end and scores[end] >= expand_threshold:
            end += 1
        groups.append((start, end))

    for previous, current in zip(groups, groups[1:]):
        if previous[1] >= current[0]:
            raise ValueError(f"检测到重叠的分隔线：{previous} 与 {current}")
    return groups


def split_without_separators(
    image: Image.Image,
    separator_groups: Sequence[tuple[int, int]],
    layer_count: int,
    background: tuple[int, int, int],
) -> list[Image.Image]:
    """排除分隔线后切图，并把各区居中放到统一宽度的画布。"""
    width, height = image.size
    starts = [0, *(end for _, end in separator_groups)]
    ends = [*(start for start, _ in separator_groups), width]
    if len(starts) != layer_count or len(ends) != layer_count:
        raise ValueError("分隔线数量与图层数量不匹配")

    # 原图声明为严格等分，因此使用总宽度/layer_count 作为规范单层宽度。
    target_width = int(round(width / layer_count))
    normalized: list[Image.Image] = []
    for index, (left, right) in enumerate(zip(starts, ends), start=1):
        if right <= left:
            raise ValueError(f"第 {index} 个图层区间为空：{left}..{right}")
        crop = image.crop((left, 0, right, height)).convert("RGB")
        canvas = Image.new("RGB", (target_width, height), background)
        if crop.width <= target_width:
            x = (target_width - crop.width) // 2
            canvas.paste(crop, (x, 0))
        else:
            x = (crop.width - target_width) // 2
            canvas.paste(crop.crop((x, 0, x + target_width, height)), (0, 0))
        normalized.append(canvas)
    return normalized


def remove_chroma_key(
    image: Image.Image,
    background: tuple[int, int, int],
    transparent_distance: float,
    opaque_distance: float,
    dominant_channel_alpha: bool = True,
) -> Image.Image:
    """以色差生成软 Alpha，并对半透明边缘做底色反混合。"""
    if opaque_distance <= transparent_distance:
        raise ValueError("--opaque-distance 必须大于 --transparent-distance")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    key = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    distance = np.sqrt(np.sum((rgb - key) ** 2, axis=2))
    alpha = np.clip(
        (distance - transparent_distance)
        / (opaque_distance - transparent_distance),
        0.0,
        1.0,
    )

    # 对纯红/绿/蓝色键额外按主通道占比估算 Alpha。相比单纯 RGB
    # 距离，这能正确处理“黑色边缘 + 绿色背景”形成的深绿色抗锯齿。
    key_flat = key.reshape(3)
    dominant = int(np.argmax(key_flat))
    other_channels = [index for index in range(3) if index != dominant]
    key_excess = float(
        key_flat[dominant] - np.max(key_flat[other_channels])
    )
    if dominant_channel_alpha and key_excess >= 40.0:
        pixel_excess = (
            rgb[..., dominant] - np.max(rgb[..., other_channels], axis=2)
        )
        background_fraction = np.clip(pixel_excess / key_excess, 0.0, 1.0)
        alpha = np.minimum(alpha, 1.0 - background_fraction)

    # 反解 observed = alpha * foreground + (1-alpha) * key，降低绿边。
    safe_alpha = np.maximum(alpha[..., None], 1.0 / 255.0)
    foreground = (rgb - (1.0 - alpha[..., None]) * key) / safe_alpha
    foreground = np.clip(foreground, 0.0, 255.0)
    foreground[alpha <= 0.0] = 0.0

    rgba = np.dstack((foreground, alpha[..., None] * 255.0))
    return Image.fromarray(np.rint(rgba).astype(np.uint8))


def clear_full_height_black_lines_at_edges(
    image: Image.Image,
    black_threshold: int,
    min_coverage: float = 0.80,
    edge_ratio: float = 0.035,
) -> Image.Image:
    """清掉切片最外缘残留的贯穿全高黑线，不触碰内部图案。"""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    width = rgba.shape[1]
    edge_width = max(2, int(round(width * edge_ratio)))
    opaque = rgba[..., 3] > 0
    near_black = np.max(rgba[..., :3], axis=2) <= black_threshold
    coverage = (near_black & opaque).mean(axis=0)
    edge_columns = list(range(edge_width)) + list(range(width - edge_width, width))
    for column in edge_columns:
        if coverage[column] >= min_coverage:
            rgba[:, column] = 0
    return Image.fromarray(rgba)


def alpha_bbox(image: Image.Image, threshold: int = 8) -> tuple[int, int, int, int] | None:
    alpha = image.convert("RGBA").getchannel("A")
    binary = alpha.point(lambda value: 255 if value > threshold else 0)
    return binary.getbbox()


def remove_cross_axis_edge_components(
    image: Image.Image,
    orientation: str,
    alpha_threshold: int = 8,
    edge_depth: int | None = None,
) -> Image.Image:
    """删除与横向边缘相连的残留色键/输出边框。

    纵向 bar 只清理接触左右边缘的连通块；横向 bar 只清理接触上下
    边缘的连通块，因此不会删除沿进度方向接触画布端点的正常内容。
    """
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    mask = rgba[..., 3] > alpha_threshold
    height, width = mask.shape
    if edge_depth is None:
        cross_size = width if orientation == "vertical" else height
        edge_depth = max(2, int(round(cross_size * 0.04)))
    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    if orientation == "vertical":
        columns = list(range(min(edge_depth, width))) + list(
            range(max(0, width - edge_depth), width)
        )
        for x in columns:
            for y in np.flatnonzero(mask[:, x]):
                queue.append((int(y), x))
    elif orientation == "horizontal":
        rows = list(range(min(edge_depth, height))) + list(
            range(max(0, height - edge_depth), height)
        )
        for y in rows:
            for x in np.flatnonzero(mask[y]):
                queue.append((y, int(x)))
    else:
        return image

    while queue:
        y, x = queue.popleft()
        if y < 0 or y >= height or x < 0 or x >= width:
            continue
        if visited[y, x] or not mask[y, x]:
            continue
        visited[y, x] = True
        queue.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))

    rgba[visited] = 0
    return Image.fromarray(rgba)


def translate_rgba(image: Image.Image, dx: int, dy: int) -> Image.Image:
    """在固定画布中平移 RGBA 图层，超出画布的部分裁掉。"""
    source = image.convert("RGBA")
    width, height = source.size
    src_left = max(0, -dx)
    src_top = max(0, -dy)
    dst_left = max(0, dx)
    dst_top = max(0, dy)
    copy_width = min(width - src_left, width - dst_left)
    copy_height = min(height - src_top, height - dst_top)
    canvas = Image.new("RGBA", source.size, (0, 0, 0, 0))
    if copy_width <= 0 or copy_height <= 0:
        return canvas
    region = source.crop(
        (src_left, src_top, src_left + copy_width, src_top + copy_height)
    )
    canvas.alpha_composite(region, (dst_left, dst_top))
    return canvas


def align_linear_bar_layers(
    layers: dict[str, Image.Image],
) -> tuple[dict[str, Image.Image], str, dict[str, tuple[int, int]]]:
    """以底板中心轴配准线性 bar，并把当前节点对齐到进度终点。"""
    base_box = alpha_bbox(layers["base_plate"])
    if base_box is None:
        return layers, "unknown", {}
    base_width = base_box[2] - base_box[0]
    base_height = base_box[3] - base_box[1]
    if base_height >= base_width * 1.5:
        orientation = "vertical"
        base_cross_center = (base_box[0] + base_box[2]) / 2.0
    elif base_width >= base_height * 1.5:
        orientation = "horizontal"
        base_cross_center = (base_box[1] + base_box[3]) / 2.0
    else:
        return layers, "non_linear", {}

    cleaned = {
        name: remove_cross_axis_edge_components(layer, orientation)
        for name, layer in layers.items()
    }
    shifts: dict[str, tuple[int, int]] = {}
    aligned: dict[str, Image.Image] = {}
    for name, layer in cleaned.items():
        box = alpha_bbox(layer)
        if box is None:
            aligned[name] = layer
            shifts[name] = (0, 0)
            continue
        if orientation == "vertical":
            center = (box[0] + box[2]) / 2.0
            dx, dy = int(round(base_cross_center - center)), 0
        else:
            center = (box[1] + box[3]) / 2.0
            dx, dy = 0, int(round(base_cross_center - center))
        aligned[name] = translate_rgba(layer, dx, dy)
        shifts[name] = (dx, dy)

    # 当前节点还应沿进度方向对齐填充层的开放端点。
    fill_box = alpha_bbox(aligned["progress_fill"])
    current_box = alpha_bbox(aligned["current_progress_indicator"])
    if fill_box is not None and current_box is not None:
        canvas_width, canvas_height = aligned["progress_fill"].size
        if orientation == "vertical":
            current_center = (current_box[1] + current_box[3]) / 2.0
            if fill_box[1] <= canvas_height * 0.05:
                endpoint = fill_box[3]
            elif fill_box[3] >= canvas_height * 0.95:
                endpoint = fill_box[1]
            else:
                endpoint = current_center
            extra = int(round(endpoint - current_center))
            if extra and abs(extra) <= canvas_height * 0.20:
                dx, dy = shifts["current_progress_indicator"]
                aligned["current_progress_indicator"] = translate_rgba(
                    aligned["current_progress_indicator"], 0, extra
                )
                shifts["current_progress_indicator"] = (dx, dy + extra)
        else:
            current_center = (current_box[0] + current_box[2]) / 2.0
            if fill_box[0] <= canvas_width * 0.05:
                endpoint = fill_box[2]
            elif fill_box[2] >= canvas_width * 0.95:
                endpoint = fill_box[0]
            else:
                endpoint = current_center
            extra = int(round(endpoint - current_center))
            if extra and abs(extra) <= canvas_width * 0.20:
                dx, dy = shifts["current_progress_indicator"]
                aligned["current_progress_indicator"] = translate_rgba(
                    aligned["current_progress_indicator"], extra, 0
                )
                shifts["current_progress_indicator"] = (dx + extra, dy)

    return aligned, orientation, shifts


def alpha_composite_layers(
    layers: dict[str, Image.Image],
    order: Sequence[str],
) -> Image.Image:
    first = layers[order[0]]
    result = Image.new("RGBA", first.size, (0, 0, 0, 0))
    for name in order:
        result.alpha_composite(layers[name])
    return result


def save_preview_on_color(
    composite: Image.Image,
    output_path: Path,
    color: tuple[int, int, int],
) -> None:
    preview = Image.new("RGBA", composite.size, (*color, 255))
    preview.alpha_composite(composite)
    preview.convert("RGB").save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="移除五等分色键图底色、排除黑色分隔线、导出并重组图层"
    )
    parser.add_argument("image", type=Path, help="五等分色键图片")
    parser.add_argument(
        "--background",
        "--bg",
        required=True,
        type=parse_color,
        help="待移除的底色，例如 '#00FF00' 或 '0,255,0'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；默认是输入图片旁的 <文件名>_layers",
    )
    parser.add_argument("--black-threshold", type=int, default=32,
                        help="分隔线黑色阈值，默认 32")
    parser.add_argument("--separator-min-coverage", type=float, default=0.80,
                        help="黑线纵向覆盖率下限，默认 0.80")
    parser.add_argument("--separator-search-ratio", type=float, default=0.12,
                        help="在每个理论边界附近搜索的单格宽度比例，默认 0.12")
    parser.add_argument("--transparent-distance", type=float, default=35.0,
                        help="与实际底色色差小于该值时完全透明，默认 35")
    parser.add_argument("--opaque-distance", type=float, default=110.0,
                        help="与实际底色色差大于该值时完全不透明，默认 110")
    parser.add_argument(
        "--no-auto-background",
        action="store_true",
        help="禁用边缘采样，严格使用 --background 的字面 RGB 值",
    )
    parser.add_argument(
        "--no-dominant-channel-alpha",
        action="store_true",
        help="禁用针对纯红/绿/蓝底的主通道 Alpha 与去色边处理",
    )
    parser.add_argument(
        "--no-auto-align",
        action="store_true",
        help="禁用线性 bar 的中心轴及当前节点自动配准",
    )
    parser.add_argument(
        "--preview-background",
        type=parse_color,
        default=(255, 255, 255),
        help="重组预览底色，默认 255,255,255",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.image.is_file():
        raise SystemExit(f"输入图片不存在：{args.image}")
    if not 0 <= args.black_threshold <= 255:
        raise SystemExit("--black-threshold 必须位于 0..255")
    if not 0.0 <= args.separator_min_coverage <= 1.0:
        raise SystemExit("--separator-min-coverage 必须位于 0..1")
    if args.separator_search_ratio <= 0:
        raise SystemExit("--separator-search-ratio 必须大于 0")

    output_dir = args.output_dir or args.image.with_name(
        f"{args.image.stem}_layers"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(args.image) as opened:
        source = opened.convert("RGB")
    rgb = np.asarray(source, dtype=np.uint8)
    actual_background = (
        args.background
        if args.no_auto_background
        else estimate_actual_background(rgb, args.background)
    )
    separators = detect_separator_groups(
        rgb,
        layer_count=len(DEFAULT_LAYER_NAMES),
        black_threshold=args.black_threshold,
        min_coverage=args.separator_min_coverage,
        search_ratio=args.separator_search_ratio,
    )
    crops = split_without_separators(
        source,
        separators,
        layer_count=len(DEFAULT_LAYER_NAMES),
        background=actual_background,
    )

    layers: dict[str, Image.Image] = {}
    for name, crop in zip(DEFAULT_LAYER_NAMES, crops):
        layer = remove_chroma_key(
            crop,
            background=actual_background,
            transparent_distance=args.transparent_distance,
            opaque_distance=args.opaque_distance,
            dominant_channel_alpha=not args.no_dominant_channel_alpha,
        )
        layer = clear_full_height_black_lines_at_edges(
            layer,
            black_threshold=args.black_threshold,
            min_coverage=args.separator_min_coverage,
        )
        layers[name] = layer

    if args.no_auto_align:
        orientation, shifts = "disabled", {}
    else:
        layers, orientation, shifts = align_linear_bar_layers(layers)

    for index, name in enumerate(DEFAULT_LAYER_NAMES, start=1):
        path = output_dir / f"{index:02d}_{name}.png"
        layers[name].save(path, optimize=True)

    composite = alpha_composite_layers(layers, DEFAULT_COMPOSITE_ORDER)
    composite_path = output_dir / "composite.png"
    composite.save(composite_path, optimize=True)
    preview_path = output_dir / "composite_preview.png"
    save_preview_on_color(composite, preview_path, args.preview_background)

    print(f"输入尺寸: {source.width}x{source.height}")
    print(f"指定底色: {args.background}; 实际底色估计: {actual_background}")
    print(f"检测到分隔线: {separators}")
    print(f"bar 方向: {orientation}; 自动配准位移: {shifts}")
    print(f"单层尺寸: {composite.width}x{composite.height}")
    print(f"导出目录: {output_dir.resolve()}")
    print(f"透明重组图: {composite_path.resolve()}")
    print(f"预览图: {preview_path.resolve()}")


if __name__ == "__main__":
    main()
