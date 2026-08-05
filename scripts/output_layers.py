import os
from psd_tools import PSDImage
from PIL import Image


def render_layers(psd, include_names=None, exclude_names=None):
    """
    渲染 PSD 第一层中指定名称的图层/图层组，
    并保持它们在原 PSD 中的位置。
    """
    result = Image.new("RGBA", (psd.width, psd.height), (0, 0, 0, 0))

    # 只遍历 PSD 第一层
    for layer in psd:
        if not layer.is_visible():
            continue

        name = layer.name

        # 只包含指定名字
        if include_names is not None and name not in include_names:
            continue

        # 排除指定名字
        if exclude_names is not None and name in exclude_names:
            continue

        # 获取图层/图层组的合成图像
        img = layer.composite()
        if img is None:
            continue

        # bbox 是 (left, top, right, bottom)
        bbox = layer.bbox
        x, y = bbox[0], bbox[1]

        # 按原位置粘贴到结果画布
        result.alpha_composite(img, (x, y))

    return result


def make_opaque(image, background=(0, 0, 0, 255)):
    """将图像合成到不透明背景，确保 alpha 范围为 (255, 255)。"""
    image = image.convert("RGBA")
    canvas = Image.new("RGBA", image.size, background)
    canvas.alpha_composite(image)
    return canvas


def process_psd(psd_path, input_path, output_path, front_layers):
    psd = PSDImage.open(psd_path)

    # 前景图：只包含 front_layers
    front_img = render_layers(psd, include_names=front_layers)

    # 背景图：排除 front_layers，并合成到黑色背景以消除透明区域。
    back_img = make_opaque(render_layers(psd, exclude_names=front_layers))

    # 原图作为训练条件图，也使用相同背景，避免输入与目标透明区域不一致。
    origin_img = make_opaque(psd.composite())

    # 保持原目录结构
    rel_path = os.path.relpath(psd_path, input_path)
    rel_dir = os.path.dirname(rel_path)
    base_name = os.path.splitext(os.path.basename(psd_path))[0]

    out_dir = os.path.join(output_path, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    front_path = os.path.join(out_dir, f"{base_name}_front_layers.png")
    back_path = os.path.join(out_dir, f"{base_name}_back_layers.png")
    origin_path = os.path.join(out_dir, f"{base_name}_origin.png")

    front_img.save(front_path)
    back_img.save(back_path)
    origin_img.save(origin_path)

    print(f"已导出: {front_path}")
    print(f"已导出: {back_path}")
    print(f"已导出: {origin_path}")


def batch_process(input_path, front_layers, output_path):
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(".psd"):
                psd_path = os.path.join(root, file)
                try:
                    process_psd(psd_path, input_path, output_path, front_layers)
                except Exception as e:
                    print(f"处理失败: {psd_path}，错误: {e}")


if __name__ == "__main__":
    # ====== 修改这里的参数 ======
    input_path = "/Users/ctw/Desktop/标注/binan"
    front_layers = ["text"]
    output_path = "/Users/ctw/Desktop/分层/binan"
    # ============================

    batch_process(input_path, front_layers, output_path)