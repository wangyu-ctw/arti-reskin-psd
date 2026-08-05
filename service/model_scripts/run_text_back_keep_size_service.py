"""去字模型推理脚本(service 版,参照 OmniPSD/inference/run_text_back_keep_size.py)。

输入输出同目录:读 <run_dir>/origin.png,输出写 <run_dir>/text_back.png。

由 service 的 text_back 任务 handler 通过 omnipsd venv 的 python 调起:
    /workspace/venvs/omnipsd-cu128/bin/python run_text_back_keep_size_service.py \
        /workspace/servData/<run_id> --seed 5 --steps 20 --prompt "..."
需要 PYTHONPATH 包含 /workspace/OmniPSD(handler 里已设置)。
"""
import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageOps

from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig

LORA_DIR = "/workspace/output/text_back"
DEFAULT_PROMPT = (
    "Remove all letters, words, and numbers from the game UI and reconstruct "
    "the clean UI underneath. Keep every icon, symbol, button, border, and "
    "non-text graphic unchanged."
)


def latest_lora() -> str:
    candidates = list(Path(LORA_DIR).glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No LoRA found in {LORA_DIR}, pass --lora explicitly")
    return str(max(candidates, key=lambda x: x.stat().st_mtime))


def keep_size_16(w: int, h: int, max_pixels: int):
    scale = min(1.0, math.sqrt(max_pixels / (w * h)))
    tw = max(16, int(round(w * scale / 16)) * 16)
    th = max(16, int(round(h * scale / 16)) * 16)

    while tw * th > max_pixels:
        if tw / w >= th / h and tw > 16:
            tw -= 16
        elif th > 16:
            th -= 16
        else:
            break
    return tw, th


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="text_back inference on one run dir")
    parser.add_argument("run_dir", help="目录,内含 origin.png,输出 text_back.png 写回同目录")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--vram-limit", type=float, default=40)
    parser.add_argument("--lora", default="", help="不传则取 LORA_DIR 里最新的 safetensors")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    origin_path = run_dir / "origin.png"
    if not origin_path.is_file():
        raise FileNotFoundError(f"origin image not found: {origin_path}")

    lora_path = args.lora or latest_lora()

    print("run_dir:", run_dir)
    print("lora:", lora_path)
    print("prompt:", args.prompt)
    print("seed:", args.seed, "steps:", args.steps, "max_pixels:", args.max_pixels)

    model_configs = [
        ModelConfig(path="/workspace/models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors", offload_device="cpu"),
        ModelConfig(path="/workspace/models/FLUX.1-dev/text_encoder/model.safetensors", offload_device="cpu"),
        ModelConfig(path="/workspace/models/FLUX.1-dev/text_encoder_2/", offload_device="cpu"),
        ModelConfig(path="/workspace/models/FLUX.1-dev/ae.safetensors", offload_device="cpu"),
    ]

    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
    )
    pipe.load_lora(pipe.dit, lora_path, alpha=1)
    pipe.enable_vram_management(vram_limit=args.vram_limit, vram_buffer=1.0)

    img = load_image(origin_path)
    w, h = img.size
    tw, th = keep_size_16(w, h, args.max_pixels)

    kontext = img
    if (tw, th) != img.size:
        kontext = img.resize((tw, th), Image.Resampling.LANCZOS)

    print(f"origin: {w}x{h} -> {tw}x{th}")

    with torch.inference_mode():
        out = pipe(
            prompt=args.prompt,
            width=tw,
            height=th,
            kontext_images=kontext,
            seed=args.seed,
            cfg_scale=1.0,
            embedded_guidance=1.0,
            num_inference_steps=args.steps,
            tiled=False,
        )

    save_path = run_dir / "text_back.png"
    out.save(save_path)
    print("saved:", save_path)


if __name__ == "__main__":
    main()
