import csv
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}

PROMPT = (
    "Remove all text, including numbers, from the game UI, then restore the UI "
    "image. If the text has a background or border, preserve the background or "
    "border. Do not remove icons such as arrows or stars. Ignore the green '仮' "
    "character."
)


def collect_images(input_path: str, output_path: str) -> None:
    """复制全部图片到输出目录，并为成组的分层图片生成 CSV。"""
    input_dir = Path(input_path).expanduser().resolve()
    output_dir = Path(output_path).expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("input_path 和 output_path 不能是同一个目录")

    image_files = []
    metadata_rows = []

    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)

        # output_path 位于 input_path 内时，避免再次扫描复制后的文件。
        dirs[:] = [
            directory
            for directory in dirs
            if (root_path / directory).resolve() != output_dir
        ]

        images_in_folder = {
            filename: root_path / filename
            for filename in files
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS
        }
        image_files.extend(images_in_folder.values())

        lower_name_map = {name.lower(): name for name in images_in_folder}
        for name in images_in_folder:
            suffix = "_back_layers.png"
            if not name.lower().endswith(suffix):
                continue

            base_name = name[: -len(suffix)]
            front_name = lower_name_map.get(f"{base_name}_front_layers.png".lower())
            origin_name = lower_name_map.get(f"{base_name}_origin.png".lower())
            if front_name and origin_name:
                metadata_rows.append(
                    {
                        "image": name,
                        "control": origin_name,
                        "prompt": PROMPT,
                    }
                )

    # 所有图片会放在同一个目录中，因此同名文件不能安全覆盖。
    names = {}
    for image_path in image_files:
        key = image_path.name.lower()
        if key in names:
            raise FileExistsError(
                f"发现同名图片，无法平铺复制:\n"
                f"  {names[key]}\n"
                f"  {image_path}"
            )
        names[key] = image_path

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_files:
        shutil.copy2(image_path, output_dir / image_path.name)

    csv_path = output_dir / "back_metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image", "control", "prompt"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"已复制 {len(image_files)} 张图片到: {output_dir}")
    print(f"已写入 {len(metadata_rows)} 条记录到: {csv_path}")


if __name__ == "__main__":
    # ====== 修改这里的路径 ======
    input_path = "/Users/ctw/Desktop/分层"
    output_path = "/Users/ctw/Desktop/收集"
    # ============================

    collect_images(input_path, output_path)
