#!/usr/bin/env python3
"""
Generate PNG previews for PSD files grouped by folders.

Default behavior:
  ~/Desktop/标注/<aaa>/**/<file>.psd
    -> ~/Desktop/预览图/<aaa>/<file>.png

PNG files are placed directly under each output <aaa> folder. If multiple PSD
files in the same <aaa> group have the same filename, later files get a
numeric suffix such as "__2" to avoid overwriting earlier previews.
"""

from __future__ import annotations

import argparse
import binascii
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    desktop = Path.home() / "Desktop"
    parser = argparse.ArgumentParser(
        description="Convert PSD files under ~/Desktop/标注/<aaa> to PNG previews under ~/Desktop/预览图/<aaa>."
    )
    parser.add_argument(
        "source_root",
        nargs="?",
        default=desktop / "标注",
        type=Path,
        help="source root, default: ~/Desktop/标注",
    )
    parser.add_argument(
        "output_root",
        nargs="?",
        default=desktop / "预览图",
        type=Path,
        help="output root, default: ~/Desktop/预览图",
    )
    parser.add_argument(
        "--preserve-folders",
        action="store_true",
        help="preserve the source subfolder structure under each output <aaa> folder",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip PNG files that already exist",
    )
    return parser.parse_args()


def expand_path(path: Path) -> Path:
    return path.expanduser().resolve()


def find_converter() -> str:
    if shutil.which("magick"):
        return "magick"
    if shutil.which("sips"):
        return "sips"
    raise RuntimeError(
        "No PSD converter found. Install ImageMagick, or run this on macOS with /usr/bin/sips available."
    )


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        len(data).to_bytes(4, "big")
        + kind
        + data
        + binascii.crc32(kind + data).to_bytes(4, "big")
    )


def rewrite_png_without_compression(src: Path, dst: Path) -> None:
    """Rewrite PNG IDAT data with zlib compression level 0.

    PNG remains lossless either way; this only avoids deflate compression so the
    resulting file is larger and quicker to encode.
    """

    data = src.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise RuntimeError(f"Converted file is not a PNG: {src}")

    output = bytearray(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE)
    idat_parts: list[bytes] = []
    wrote_idat = False

    def flush_idat() -> None:
        nonlocal wrote_idat
        if not idat_parts:
            return
        raw = zlib.decompress(b"".join(idat_parts))
        output.extend(png_chunk(b"IDAT", zlib.compress(raw, level=0)))
        idat_parts.clear()
        wrote_idat = True

    while offset < len(data):
        if offset + 8 > len(data):
            raise RuntimeError(f"Invalid PNG chunk header: {src}")

        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        chunk_start = offset
        chunk_end = offset + 12 + length

        if chunk_end > len(data):
            raise RuntimeError(f"Invalid PNG chunk length: {src}")

        chunk_data = data[offset + 8 : offset + 8 + length]
        offset = chunk_end

        if kind == b"IDAT":
            idat_parts.append(chunk_data)
            continue

        if idat_parts:
            flush_idat()

        output.extend(data[chunk_start:chunk_end])

        if kind == b"IEND":
            break

    if idat_parts:
        flush_idat()

    if not wrote_idat:
        raise RuntimeError(f"PNG has no IDAT data: {src}")

    dst.write_bytes(output)


def convert_with_magick(psd_path: Path, png_path: Path) -> None:
    # [0] selects the flattened composite image from PSD instead of exporting every layer.
    subprocess.run(
        [
            "magick",
            f"{psd_path}[0]",
            "-define",
            "png:compression-level=0",
            str(png_path),
        ],
        check=True,
    )


def convert_with_sips(psd_path: Path, png_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="psd-preview-") as temp_dir:
        temp_png = Path(temp_dir) / "preview.png"
        subprocess.run(
            ["sips", "-s", "format", "png", str(psd_path), "--out", str(temp_png)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        rewrite_png_without_compression(temp_png, png_path)


def output_path_for(
    psd_path: Path,
    group_dir: Path,
    output_group_dir: Path,
    preserve_folders: bool,
    used_flat_names: set[str],
) -> Path:
    relative = psd_path.relative_to(group_dir).with_suffix(".png")
    if preserve_folders:
        return output_group_dir / relative

    stem = relative.stem
    suffix = relative.suffix
    index = 1

    while True:
        filename = f"{stem}{suffix}" if index == 1 else f"{stem}__{index}{suffix}"
        key = filename.casefold()
        if key not in used_flat_names:
            used_flat_names.add(key)
            return output_group_dir / filename
        index += 1


def iter_group_dirs(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def iter_psd_files(group_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in group_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".psd"
    )


def main() -> int:
    args = parse_args()
    source_root = expand_path(args.source_root)
    output_root = expand_path(args.output_root)

    if not source_root.is_dir():
        print(f"Source directory does not exist: {source_root}", file=sys.stderr)
        return 2

    converter = find_converter()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Converter: {converter}")

    converted = 0
    skipped = 0
    failed = 0

    for group_dir in iter_group_dirs(source_root):
        output_group_dir = output_root / group_dir.name
        psd_files = iter_psd_files(group_dir)
        used_flat_names: set[str] = set()

        if not psd_files:
            continue

        print(f"\n[{group_dir.name}] {len(psd_files)} PSD file(s)")

        for psd_path in psd_files:
            png_path = output_path_for(
                psd_path,
                group_dir,
                output_group_dir,
                args.preserve_folders,
                used_flat_names,
            )
            display_src = psd_path.relative_to(source_root)
            display_dst = png_path.relative_to(output_root)

            if args.skip_existing and png_path.exists():
                skipped += 1
                print(f"  skip  {display_src} -> {display_dst}")
                continue

            png_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                if converter == "magick":
                    convert_with_magick(psd_path, png_path)
                else:
                    convert_with_sips(psd_path, png_path)
                converted += 1
                print(f"  ok    {display_src} -> {display_dst}")
            except subprocess.CalledProcessError as exc:
                failed += 1
                detail = exc.stderr.strip() if isinstance(exc.stderr, str) else str(exc)
                print(f"  fail  {display_src}: {detail}", file=sys.stderr)
            except Exception as exc:
                failed += 1
                print(f"  fail  {display_src}: {exc}", file=sys.stderr)

    print(f"\nDone. converted={converted}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
