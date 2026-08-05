#!/usr/bin/env python3
"""Create transparent PNG cutouts from YOLO boxes with SAM 2.

YOLO label format (one detection per line):

    <class_id> <x_center> <y_center> <width> <height> <confidence>

The four geometry values are normalized to [0, 1]. Each retained box is used
as an independent SAM 2 box prompt. The resulting masks are merged into one
alpha channel and saved as a PNG.

Examples:

    python scripts/sam2_yolo_cutout_range.py \
        --input input.png \
        --yolo-txt detections.txt \
        --output-dir outputs

    python scripts/sam2_yolo_cutout_range.py \
        --input images \
        --yolo-txt labels \
        --output-dir outputs \
        --recursive \
        --confidence-threshold 0.6 \
        --class-ids 0,2
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LOGGER = logging.getLogger("sam2-yolo-cutout")
DEFAULT_EXTENSIONS = ".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAM2_ROOT = SCRIPT_DIR.parent
SWEEP_PARAMETERS = (
    ("mask_threshold", float),
    ("max_hole_area", int),
    ("max_sprinkle_area", int),
    ("confidence_threshold", float),
    ("padding_ratio", float),
    ("min_padding", float),
    ("nms_iou", float),
    ("max_detections", int),
    ("feather_radius", float),
)


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]
    source_line: int


def parse_class_ids(value: str) -> frozenset[int]:
    try:
        values = frozenset(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("class IDs must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one class ID is required")
    return values


def parse_extensions(value: str) -> frozenset[str]:
    extensions = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    if not extensions:
        raise argparse.ArgumentTypeError("at least one image extension is required")
    return frozenset(extensions)


def add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    sweep = parser.add_argument_group(
        "single-parameter sweep",
        "Provide start, end, and interval together. At most one parameter may "
        "produce multiple values in a run.",
    )
    for parameter, value_type in SWEEP_PARAMETERS:
        option = parameter.replace("_", "-")
        for part in ("start", "end", "interval"):
            sweep.add_argument(
                f"--{option}-{part}",
                type=value_type,
                default=None,
                help=f"{part} value for a {option} sweep",
            )


def expand_numeric_range(
    parameter: str,
    start: int | float,
    end: int | float,
    interval: int | float,
    value_type: type,
) -> list[int | float]:
    if not all(math.isfinite(float(value)) for value in (start, end, interval)):
        raise ValueError(f"--{parameter.replace('_', '-')} sweep values must be finite")
    if interval == 0:
        raise ValueError(f"--{parameter.replace('_', '-')}-interval must not be zero")
    if end > start and interval < 0:
        raise ValueError(f"--{parameter.replace('_', '-')}-interval must be positive")
    if end < start and interval > 0:
        raise ValueError(f"--{parameter.replace('_', '-')}-interval must be negative")

    if value_type is int:
        stop = end + (1 if interval > 0 else -1)
        value_range = range(int(start), int(stop), int(interval))
        if len(value_range) > 10_000:
            raise ValueError(
                f"--{parameter.replace('_', '-')} sweep exceeds 10000 values"
            )
        values = list(value_range)
    else:
        distance = float(end) - float(start)
        step_count = abs(distance / float(interval))
        if not math.isfinite(step_count) or step_count >= 10_000:
            raise ValueError(
                f"--{parameter.replace('_', '-')} sweep exceeds 10000 values"
            )
        count = int(math.floor(step_count + 1e-12)) + 1
        values = [float(start) + index * float(interval) for index in range(count)]
        tolerance = max(1.0, abs(float(end))) * 1e-12
        values = [
            value
            for value in values
            if (
                value <= float(end) + tolerance
                if interval > 0
                else value >= float(end) - tolerance
            )
        ]
        values = [
            float(end) if abs(value - float(end)) <= tolerance else value
            for value in values
        ]
        values = [float(f"{value:.12g}") for value in values]

    if not values:
        values = [value_type(start)]
    return values


def build_sweep_configurations(
    args: argparse.Namespace,
) -> tuple[str | None, list[argparse.Namespace]]:
    specified: list[tuple[str, list[int | float]]] = []
    for parameter, value_type in SWEEP_PARAMETERS:
        values = [
            getattr(args, f"{parameter}_{part}") for part in ("start", "end", "interval")
        ]
        provided = [value is not None for value in values]
        if any(provided) and not all(provided):
            option = parameter.replace("_", "-")
            raise ValueError(
                f"--{option}-start, --{option}-end, and --{option}-interval "
                "must be provided together"
            )
        if all(provided):
            specified.append(
                (
                    parameter,
                    expand_numeric_range(
                        parameter, values[0], values[1], values[2], value_type
                    ),
                )
            )

    varying = [(name, values) for name, values in specified if len(values) > 1]
    if len(varying) > 1:
        names = ", ".join(name.replace("_", "-") for name, _ in varying)
        raise ValueError(f"only one parameter may be swept per run; got: {names}")

    sweep_name = (
        varying[0][0]
        if varying
        else (specified[0][0] if len(specified) == 1 else None)
    )
    sweep_values = (
        varying[0][1]
        if varying
        else (specified[0][1] if sweep_name is not None else [None])
    )
    configurations: list[argparse.Namespace] = []
    for sweep_value in sweep_values:
        configuration = copy.copy(args)
        for parameter, values in specified:
            setattr(
                configuration,
                parameter,
                sweep_value if parameter == sweep_name else values[0],
            )
        validate_numeric_args(configuration)
        configurations.append(configuration)
    return sweep_name, configurations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use normalized YOLO detections as SAM 2 box prompts and save one "
            "transparent cutout PNG per input image."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required = parser.add_argument_group("required paths")
    required.add_argument(
        "--input",
        required=True,
        type=Path,
        help="input image file or directory",
    )
    required.add_argument(
        "--yolo-txt",
        required=True,
        type=Path,
        help=(
            "YOLO txt file, or a label directory containing one same-stem txt "
            "file per input image"
        ),
    )
    required.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for output PNG files",
    )

    model = parser.add_argument_group("SAM 2 model")
    model.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/sam2.1_hiera_large.pt"),
        help=(
            "SAM 2 checkpoint path; relative paths are checked from the current "
            "directory and then from the script's parent directory"
        ),
    )
    model.add_argument(
        "--model-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM 2 Hydra model config name",
    )
    model.add_argument(
        "--device",
        default="auto",
        help="torch device, such as auto, cuda, cuda:0, mps, or cpu",
    )
    model.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use automatic mixed precision on CUDA",
    )
    model.add_argument(
        "--amp-dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help="CUDA automatic mixed-precision data type",
    )
    model.add_argument(
        "--multimask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request multiple candidate masks and retain the best SAM score",
    )
    model.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        help="SAM logit threshold used to create binary masks",
    )
    model.add_argument(
        "--max-hole-area",
        type=int,
        default=0,
        help="fill mask holes up to this area; requires SAM 2 CUDA extension",
    )
    model.add_argument(
        "--max-sprinkle-area",
        type=int,
        default=0,
        help="remove mask islands up to this area; requires SAM 2 CUDA extension",
    )

    detections = parser.add_argument_group("YOLO detections")
    detections.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="discard YOLO detections below this confidence",
    )
    detections.add_argument(
        "--class-ids",
        type=parse_class_ids,
        default=None,
        help="optional comma-separated class IDs to retain, for example 0,2,5",
    )
    detections.add_argument(
        "--padding-ratio",
        type=float,
        default=0.15,
        help="expand each side of a YOLO box by this fraction of box size",
    )
    detections.add_argument(
        "--min-padding",
        type=float,
        default=4.0,
        help="minimum box expansion in pixels on each side",
    )
    detections.add_argument(
        "--nms-iou",
        type=float,
        default=0.0,
        help="optional class-aware NMS IoU threshold; 0 disables NMS",
    )
    detections.add_argument(
        "--max-detections",
        type=int,
        default=0,
        help="maximum boxes per image after filtering; 0 means unlimited",
    )
    detections.add_argument(
        "--strict-yolo",
        action="store_true",
        help="fail instead of warning when a YOLO line is malformed",
    )

    output = parser.add_argument_group("input and output")
    output.add_argument(
        "--recursive",
        action="store_true",
        help="search input directories recursively and preserve subdirectories",
    )
    output.add_argument(
        "--extensions",
        type=parse_extensions,
        default=parse_extensions(DEFAULT_EXTENSIONS),
        help="comma-separated image extensions used in directory mode",
    )
    output.add_argument(
        "--output-suffix",
        default="_cutout",
        help="suffix added to each source image stem",
    )
    output.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overwrite existing output PNG files",
    )
    output.add_argument(
        "--feather-radius",
        type=float,
        default=0.0,
        help="Gaussian blur radius for the output alpha edge; 0 keeps a hard mask",
    )
    output.add_argument(
        "--crop",
        action="store_true",
        help="crop the PNG to the non-transparent result bounds",
    )
    output.add_argument(
        "--crop-padding",
        type=int,
        default=2,
        help="extra pixels around a cropped cutout",
    )
    output.add_argument(
        "--save-mask",
        action="store_true",
        help="also save the merged grayscale mask as a PNG",
    )
    output.add_argument(
        "--mask-suffix",
        default="_mask",
        help="suffix for optional mask PNG files",
    )
    output.add_argument(
        "--png-compress-level",
        type=int,
        choices=range(0, 10),
        default=6,
        metavar="0-9",
        help="PNG compression level",
    )
    output.add_argument(
        "--empty-policy",
        choices=("transparent", "copy", "skip"),
        default="transparent",
        help="behavior when no YOLO detections remain after filtering",
    )
    output.add_argument(
        "--missing-label-policy",
        choices=("error", "transparent", "skip"),
        default="error",
        help="behavior when label-directory mode cannot find an image label file",
    )
    output.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue processing other images after an image fails",
    )
    output.add_argument(
        "--verbose",
        action="store_true",
        help="enable verbose logging",
    )

    add_sweep_arguments(parser)
    return parser


def validate_numeric_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")
    if args.padding_ratio < 0:
        raise ValueError("--padding-ratio must be non-negative")
    if args.min_padding < 0:
        raise ValueError("--min-padding must be non-negative")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be between 0 and 1")
    if args.max_detections < 0:
        raise ValueError("--max-detections must be non-negative")
    if args.max_hole_area < 0 or args.max_sprinkle_area < 0:
        raise ValueError("mask post-processing areas must be non-negative")
    if args.feather_radius < 0:
        raise ValueError("--feather-radius must be non-negative")
    if args.crop_padding < 0:
        raise ValueError("--crop-padding must be non-negative")


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise ValueError(f"input does not exist: {args.input}")
    if not args.yolo_txt.exists():
        raise ValueError(f"YOLO path does not exist: {args.yolo_txt}")
    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    validate_numeric_args(args)
    if args.input.is_dir() and args.yolo_txt.is_file():
        LOGGER.warning(
            "A single YOLO txt is being reused for every image in the input directory"
        )


def resolve_checkpoint_path(checkpoint: Path) -> Path:
    """Resolve a checkpoint without depending on a machine-specific path."""
    checkpoint = checkpoint.expanduser()
    candidates = [checkpoint]
    if not checkpoint.is_absolute():
        candidates.append(DEFAULT_SAM2_ROOT / checkpoint)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise ValueError(
        "SAM 2 checkpoint does not exist. Checked: "
        f"{checked}. Pass its Pod path with --checkpoint."
    )


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_images(
    input_path: Path,
    output_dir: Path,
    extensions: frozenset[str],
    recursive: bool,
) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    resolved_output = output_dir.resolve()
    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.iterdir()
    images = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if is_relative_to(path.resolve(), resolved_output):
            continue
        images.append(path)
    return sorted(images)


def label_path_for_image(
    image_path: Path,
    input_path: Path,
    yolo_path: Path,
) -> Path | None:
    if yolo_path.is_file():
        return yolo_path

    if input_path.is_dir():
        relative = image_path.relative_to(input_path).with_suffix(".txt")
        nested_candidate = yolo_path / relative
        if nested_candidate.is_file():
            return nested_candidate

    flat_candidate = yolo_path / f"{image_path.stem}.txt"
    return flat_candidate if flat_candidate.is_file() else None


def malformed_line(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    LOGGER.warning(message)


def parse_yolo_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    confidence_threshold: float,
    class_ids: frozenset[int] | None,
    padding_ratio: float,
    min_padding: float,
    strict: bool,
) -> list[Detection]:
    detections: list[Detection] = []

    with label_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) != 6:
                malformed_line(
                    f"{label_path}:{line_number}: expected 6 fields, got {len(fields)}",
                    strict,
                )
                continue

            try:
                class_value, x_center, y_center, width, height, confidence = map(
                    float, fields
                )
                class_id = int(class_value)
            except (ValueError, OverflowError):
                malformed_line(
                    f"{label_path}:{line_number}: fields must be numeric",
                    strict,
                )
                continue

            numeric_values = (
                class_value,
                x_center,
                y_center,
                width,
                height,
                confidence,
            )
            if not all(math.isfinite(value) for value in numeric_values):
                malformed_line(
                    f"{label_path}:{line_number}: fields must be finite numbers",
                    strict,
                )
                continue

            if class_value != class_id:
                malformed_line(
                    f"{label_path}:{line_number}: class ID must be an integer",
                    strict,
                )
                continue
            if width <= 0 or height <= 0:
                malformed_line(
                    f"{label_path}:{line_number}: width and height must be positive",
                    strict,
                )
                continue
            if confidence < confidence_threshold:
                continue
            if class_ids is not None and class_id not in class_ids:
                continue

            x1 = (x_center - width / 2.0) * image_width
            y1 = (y_center - height / 2.0) * image_height
            x2 = (x_center + width / 2.0) * image_width
            y2 = (y_center + height / 2.0) * image_height

            pad_x = max((x2 - x1) * padding_ratio, min_padding)
            pad_y = max((y2 - y1) * padding_ratio, min_padding)
            x1 = max(0.0, x1 - pad_x)
            y1 = max(0.0, y1 - pad_y)
            x2 = min(float(image_width - 1), x2 + pad_x)
            y2 = min(float(image_height - 1), y2 + pad_y)

            if x2 <= x1 or y2 <= y1:
                malformed_line(
                    f"{label_path}:{line_number}: box is outside the image or empty",
                    strict,
                )
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    confidence=confidence,
                    box=(x1, y1, x2, y2),
                    source_line=line_number,
                )
            )

    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def apply_nms(detections: Sequence[Detection], iou_threshold: float) -> list[Detection]:
    if iou_threshold <= 0:
        return list(detections)

    retained: list[Detection] = []
    for candidate in detections:
        duplicate = any(
            candidate.class_id == existing.class_id
            and box_iou(candidate.box, existing.box) > iou_threshold
            for existing in retained
        )
        if not duplicate:
            retained.append(candidate)
    return retained


def resolve_device(torch_module, requested: str):
    if requested != "auto":
        device = torch_module.device(requested)
        if device.type == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return device

    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def resolve_amp_dtype(torch_module, device, requested: str):
    if requested == "float16":
        return torch_module.float16
    if requested == "bfloat16":
        return torch_module.bfloat16
    if device.type == "cuda" and torch_module.cuda.get_device_capability(device)[0] >= 8:
        return torch_module.bfloat16
    return torch_module.float16


def output_path_for_image(
    image_path: Path,
    input_path: Path,
    output_dir: Path,
    suffix: str,
    timestamp: int,
    sweep_name: str | None = None,
    sweep_value: int | float | None = None,
) -> Path:
    relative_parent = (
        image_path.relative_to(input_path).parent if input_path.is_dir() else Path()
    )
    scan_suffix = ""
    if sweep_name is not None and sweep_value is not None:
        parameter = sweep_name.replace("_", "-")
        value = format_parameter_value(sweep_value)
        scan_suffix = f"_{parameter}-{value}"
    filename = f"{image_path.stem}{suffix}_{timestamp}{scan_suffix}.png"
    return output_dir / relative_parent / filename


def format_parameter_value(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.12g}"


def save_empty_result(
    image_rgba,
    output_path: Path,
    policy: str,
    compress_level: int,
) -> str:
    if policy == "skip":
        return "skipped"
    if policy == "transparent":
        image_rgba.putalpha(0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_rgba.save(output_path, format="PNG", compress_level=compress_level)
    return "written"


def process_image(
    *,
    args: argparse.Namespace,
    image_path: Path,
    label_path: Path | None,
    output_path: Path,
    predictor,
    torch_module,
    np_module,
    image_module,
    image_filter_module,
    image_ops_module,
    device,
    amp_dtype,
) -> tuple[str, int]:
    if output_path.exists() and not args.overwrite:
        LOGGER.info("Skipping existing output: %s", output_path)
        return "skipped", 0

    with image_module.open(image_path) as opened_image:
        image_rgba = image_ops_module.exif_transpose(opened_image).convert("RGBA")
    rgba_array = np_module.asarray(image_rgba)
    image_rgb = np_module.ascontiguousarray(rgba_array[:, :, :3])
    original_alpha = rgba_array[:, :, 3]
    image_height, image_width = image_rgb.shape[:2]

    if label_path is None:
        if args.missing_label_policy == "error":
            raise FileNotFoundError(f"no YOLO txt found for {image_path}")
        if args.missing_label_policy == "skip":
            LOGGER.warning("Skipping image without a YOLO txt: %s", image_path)
            return "skipped", 0
        result = save_empty_result(
            image_rgba, output_path, "transparent", args.png_compress_level
        )
        return result, 0

    detections = parse_yolo_file(
        label_path=label_path,
        image_width=image_width,
        image_height=image_height,
        confidence_threshold=args.confidence_threshold,
        class_ids=args.class_ids,
        padding_ratio=args.padding_ratio,
        min_padding=args.min_padding,
        strict=args.strict_yolo,
    )
    detections = apply_nms(detections, args.nms_iou)
    if args.max_detections:
        detections = detections[: args.max_detections]

    if not detections:
        LOGGER.warning("No retained YOLO detections for %s", image_path)
        result = save_empty_result(
            image_rgba, output_path, args.empty_policy, args.png_compress_level
        )
        return result, 0

    merged_mask = np_module.zeros((image_height, image_width), dtype=np_module.bool_)
    use_amp = args.amp and device.type == "cuda"
    amp_context = (
        torch_module.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else nullcontext()
    )

    with torch_module.inference_mode(), amp_context:
        predictor.set_image(image_rgb)
        for detection in detections:
            box = np_module.asarray(detection.box, dtype=np_module.float32)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=args.multimask,
                return_logits=False,
            )
            best_index = int(np_module.argmax(scores))
            merged_mask |= masks[best_index].astype(np_module.bool_)

    predictor.reset_predictor()

    alpha = merged_mask.astype(np_module.uint8) * 255
    if args.feather_radius > 0:
        alpha_image = image_module.fromarray(alpha).filter(
            image_filter_module.GaussianBlur(radius=args.feather_radius)
        )
        alpha = np_module.asarray(alpha_image)

    alpha = np_module.minimum(alpha, original_alpha)
    result_image = image_module.fromarray(
        np_module.dstack((image_rgb, alpha)).astype(np_module.uint8)
    )
    mask_image = image_module.fromarray(alpha.astype(np_module.uint8))

    if args.crop:
        bounds = mask_image.getbbox()
        if bounds is not None:
            left = max(0, bounds[0] - args.crop_padding)
            top = max(0, bounds[1] - args.crop_padding)
            right = min(image_width, bounds[2] + args.crop_padding)
            bottom = min(image_height, bounds[3] + args.crop_padding)
            crop_box = (left, top, right, bottom)
            result_image = result_image.crop(crop_box)
            mask_image = mask_image.crop(crop_box)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(
        output_path,
        format="PNG",
        compress_level=args.png_compress_level,
    )

    if args.save_mask:
        mask_path = output_path.with_name(
            f"{output_path.stem}{args.mask_suffix}.png"
        )
        mask_image.save(
            mask_path,
            format="PNG",
            compress_level=args.png_compress_level,
        )

    return "written", len(detections)


def run(args: argparse.Namespace) -> int:
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageFilter, ImageOps
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Activate the SAM 2 environment and install "
            f"numpy, Pillow, torch, and the sam2 package. Import error: {exc}"
        ) from exc

    validate_args(args)
    sweep_name, configurations = build_sweep_configurations(args)
    images = discover_images(
        args.input,
        args.output_dir,
        args.extensions,
        args.recursive,
    )
    if not images:
        raise ValueError(f"no input images found under: {args.input}")

    device = resolve_device(torch, args.device)
    amp_dtype = resolve_amp_dtype(torch, device, args.amp_dtype)
    LOGGER.info("Loading SAM 2 on %s from %s", device, args.checkpoint)

    model = build_sam2(
        args.model_config,
        str(args.checkpoint),
        device=device,
    )
    model.eval()

    written = 0
    skipped = 0
    failed = 0
    total_detections = 0
    run_timestamp = time.time_ns() // 1_000_000
    total_jobs = len(configurations) * len(images)
    job_index = 0

    for configuration in configurations:
        sweep_value = (
            getattr(configuration, sweep_name) if sweep_name is not None else None
        )
        if sweep_name is not None:
            LOGGER.info(
                "Sweep %s=%s",
                sweep_name.replace("_", "-"),
                format_parameter_value(sweep_value),
            )
        predictor = SAM2ImagePredictor(
            model,
            mask_threshold=configuration.mask_threshold,
            max_hole_area=configuration.max_hole_area,
            max_sprinkle_area=configuration.max_sprinkle_area,
        )

        for image_path in images:
            job_index += 1
            label_path = label_path_for_image(
                image_path, configuration.input, configuration.yolo_txt
            )
            output_path = output_path_for_image(
                image_path,
                configuration.input,
                configuration.output_dir,
                configuration.output_suffix,
                run_timestamp,
                sweep_name,
                sweep_value,
            )
            LOGGER.info("[%d/%d] Processing %s", job_index, total_jobs, image_path)
            try:
                status, detection_count = process_image(
                    args=configuration,
                    image_path=image_path,
                    label_path=label_path,
                    output_path=output_path,
                    predictor=predictor,
                    torch_module=torch,
                    np_module=np,
                    image_module=Image,
                    image_filter_module=ImageFilter,
                    image_ops_module=ImageOps,
                    device=device,
                    amp_dtype=amp_dtype,
                )
                total_detections += detection_count
                if status == "written":
                    written += 1
                    LOGGER.info("Wrote %s using %d boxes", output_path, detection_count)
                else:
                    skipped += 1
            except Exception:
                failed += 1
                LOGGER.exception("Failed to process %s", image_path)
                if not configuration.continue_on_error:
                    raise

    LOGGER.info(
        "Finished timestamp %d: %d written, %d skipped, %d failed, "
        "%d retained boxes",
        run_timestamp,
        written,
        skipped,
        failed,
        total_detections,
    )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        return run(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
