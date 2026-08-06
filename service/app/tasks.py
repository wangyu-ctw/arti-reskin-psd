"""任务 handler 注册处。

每个 handler 在 GPU worker 线程里串行执行,签名: handler(payload: dict) -> Any(可 JSON 序列化)。
payload 里约定带 run_id,handler 用 storage.get_run_dir(run_id) 拿到目录,把输出写回去。

后续在这里实现 omnipsd / yolo / sam2 的真实逻辑,模型权重建议模块级加载一次、常驻显存。
"""
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from . import comfy, sam2c, storage, yoloc
from .config import (
    FLUX_FILL_UNET_NAME,
    ICON_BACK_LORA_NAME,
    OMNIPSD_PYTHON,
    OMNIPSD_ROOT,
    TEXT_BACK_LORA_NAME,
    TEXT_BACK_SCRIPT,
    TEXT_BACK_TIMEOUT,
)
from .worker import worker

DEFAULT_TEXT_BACK_PROMPT = (
    "Remove all letters, words, and numbers from the game UI and reconstruct "
    "the clean UI underneath. Keep every icon, symbol, button, border, and "
    "non-text graphic unchanged."
)


def keep_size_16(w: int, h: int, max_pixels: int):
    """和 OmniPSD/run_text_back_keep_size.py 相同的尺寸对齐逻辑(16 对齐、限像素)。"""
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


def _resolve_run_dir(payload: dict) -> Path:
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    if not (run_dir / "origin.png").is_file():
        raise FileNotFoundError(f"origin.png not found in {run_dir}")
    return run_dir


def build_text_back_workflow(image_name: str, prompt: str, seed: int, steps: int,
                             width: int, height: int, lora_name: str,
                             guidance: float = 1.0, cfg: float = 1.0) -> dict:
    """FLUX Kontext + OmniPSD LoRA 的去字 workflow(ComfyUI API 格式)。"""
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "flux1-kontext-dev.safetensors",
                            "weight_dtype": "default"}},
        "clip": {"class_type": "DualCLIPLoader",
                 "inputs": {"clip_name1": "clip_l.safetensors",
                            "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": lora_name,
                            "strength_model": 1.0}},
        "load": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "scale": {"class_type": "ImageScale",
                  "inputs": {"image": ["load", 0], "upscale_method": "lanczos",
                             "width": width, "height": height, "crop": "disabled"}},
        "encode": {"class_type": "VAEEncode",
                   "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip", 0], "text": prompt}},
        "ref": {"class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["pos", 0], "latent": ["encode", 0]}},
        "guide": {"class_type": "FluxGuidance",
                  "inputs": {"conditioning": ["ref", 0], "guidance": guidance}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "sample": {"class_type": "KSampler",
                   "inputs": {"model": ["lora", 0], "positive": ["guide", 0],
                              "negative": ["neg", 0], "latent_image": ["encode", 0],
                              "seed": seed, "steps": steps, "cfg": cfg,
                              "sampler_name": "euler", "scheduler": "simple",
                              "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": "text_back"}},
    }


def handle_hello(payload: dict) -> dict:
    """演示任务:模拟一次占用 GPU 的耗时计算。"""
    sleep = float(payload.get("sleep", 3))
    time.sleep(sleep)
    result = {"message": f"hello, {payload.get('name', 'world')}!", "slept": sleep}

    # 如果带了 run_id,演示写回 run 目录
    run_id = payload.get("run_id")
    if run_id:
        run_dir = storage.get_run_dir(run_id)
        out = run_dir / "hello.txt"
        out.write_text(result["message"], encoding="utf-8")
        result["output_path"] = str(out)
    return result


def _build_text_protect_mask(run_dir: Path, size: tuple, grow: int,
                             feather: float, conf_min: float):
    """保护合成 mask(白=文字区,取 Kontext 重生成像素;黑=非文字区,保留原图像素)。

    用 YOLO text 框圈出文字区并外扩;icon 等非文字元素落在黑区,物理上不会被改动。
    检测结果顺手写 yolo.txt 留档(与 handle_yolo 同格式)。
    """
    det = yoloc.detect({
        "dir": str(run_dir), "image": "origin.png",
        "imgsz": 1600, "conf": 0.05, "iou": 0.7,
        "augment": False, "slice": False, "slice_size": 640,
    })
    lines = det.get("lines", [])
    (run_dir / "yolo.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tw, th = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    used = 0
    for line in lines:
        parts = line.split()
        if len(parts) < 5 or parts[0] != "0":
            continue
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        if conf < conf_min:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        draw.rectangle(
            [(cx - w / 2) * tw - grow, (cy - h / 2) * th - grow,
             (cx + w / 2) * tw + grow, (cy + h / 2) * th + grow],
            fill=255)
        used += 1
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask, used


def handle_text_back(payload: dict) -> dict:
    """去字模型(常驻 ComfyUI 版):读 <run_dir>/origin.png,输出 <run_dir>/text_back.png。

    payload: run_id(必填,或用 dir 直接指定目录)、seed、steps、prompt、
             max_pixels、guidance、lora(均可选)。
             protect(默认 True):保护合成——只有 YOLO text 框内取重生成像素,
             其余保留原图,icon 物理上不可能被误删;
             protect_grow(默认 8)文字框外扩 px、protect_feather(默认 4)边缘羽化、
             protect_conf(默认 0.2)text 框置信度门槛(太低的框不给重生成权,
             防止误检的"假文字"框让 icon 失去保护)。
    """
    run_dir = _resolve_run_dir(payload)
    origin = run_dir / "origin.png"

    seed = int(payload.get("seed", 5))
    steps = int(payload.get("steps", 20))
    prompt = payload.get("prompt") or DEFAULT_TEXT_BACK_PROMPT
    max_pixels = int(payload.get("max_pixels", 1048576))
    guidance = float(payload.get("guidance", 1.0))
    lora_name = payload.get("lora") or TEXT_BACK_LORA_NAME

    with Image.open(origin) as img:
        w, h = img.size
    tw, th = keep_size_16(w, h, max_pixels)

    started = time.time()
    image_name = comfy.place_input_image(origin, prefix="text_back_")
    workflow = build_text_back_workflow(
        image_name=image_name, prompt=prompt, seed=seed, steps=steps,
        width=tw, height=th, lora_name=lora_name, guidance=guidance,
    )
    entry = comfy.run_workflow(workflow)
    images = comfy.output_image_paths(entry)
    if not images:
        raise RuntimeError(f"ComfyUI finished but produced no output image "
                           f"(prompt_id={entry.get('prompt_id')})")

    output_path = run_dir / "text_back.png"
    protect = bool(payload.get("protect", True))
    protected_boxes = None
    if protect:
        mask, protected_boxes = _build_text_protect_mask(
            run_dir, (tw, th),
            grow=int(payload.get("protect_grow", 8)),
            feather=float(payload.get("protect_feather", 4)),
            conf_min=float(payload.get("protect_conf", 0.2)))
        with Image.open(images[0]) as g:
            gen = g.convert("RGB")
            if gen.size != (tw, th):
                gen = gen.resize((tw, th), Image.Resampling.LANCZOS)
        with Image.open(origin) as o:
            base = o.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
        # mask 白区取重生成像素,黑区保留原图像素
        Image.composite(gen, base, mask).save(output_path)
    else:
        shutil.copyfile(images[0], output_path)
    return {
        "output_path": str(output_path),
        "size": [tw, th],
        "protect": protect,
        "protected_text_boxes": protected_boxes,
        "prompt_id": entry.get("prompt_id"),
        "elapsed_sec": round(time.time() - started, 1),
    }


def handle_comfy_workflow(payload: dict) -> dict:
    """通用 ComfyUI 任务:payload.workflow 是完整的 API 格式 workflow JSON。

    可选 run_id/dir:提供时把所有输出图片拷到 <run_dir>/comfy/ 下。
    """
    workflow = payload.get("workflow")
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("payload.workflow must be a non-empty workflow dict (API format)")

    started = time.time()
    entry = comfy.run_workflow(workflow)
    images = comfy.output_image_paths(entry)

    copied = []
    if payload.get("run_id") or payload.get("dir"):
        run_dir = Path(payload["dir"]) if payload.get("dir") \
            else storage.get_run_dir(payload["run_id"])
        out_dir = run_dir / "comfy"
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in images:
            dst = out_dir / src.name
            shutil.copyfile(src, dst)
            copied.append(str(dst))
    return {
        "images": copied or [str(p) for p in images],
        "prompt_id": entry.get("prompt_id"),
        "elapsed_sec": round(time.time() - started, 1),
    }


def handle_text_back_cold(payload: dict) -> dict:
    """去字模型(备用冷启动版):子进程走 diffsynth,每次任务重新加载模型。

    payload 同 text_back,另支持 vram_limit;日志写 <run_dir>/text_back.log。
    """
    run_dir = _resolve_run_dir(payload)

    cmd = [OMNIPSD_PYTHON, str(TEXT_BACK_SCRIPT), str(run_dir)]
    for key, flag in [("prompt", "--prompt"), ("seed", "--seed"), ("steps", "--steps"),
                      ("max_pixels", "--max-pixels"), ("vram_limit", "--vram-limit"),
                      ("lora", "--lora")]:
        if payload.get(key) is not None:
            cmd += [flag, str(payload[key])]

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{OMNIPSD_ROOT}:{env.get('PYTHONPATH', '')}"

    log_path = run_dir / "text_back.log"
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd, cwd=OMNIPSD_ROOT, env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
            timeout=TEXT_BACK_TIMEOUT,
        )

    output_path = run_dir / "text_back.png"
    if proc.returncode != 0 or not output_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(
            f"text_back failed (exit {proc.returncode}), log tail:\n{tail}"
        )
    return {
        "output_path": str(output_path),
        "log_path": str(log_path),
        "elapsed_sec": round(time.time() - started, 1),
    }


def build_flux_fill_workflow(image_name: str, mask_name: str, prompt: str,
                             seed: int, steps: int, width: int, height: int,
                             guidance: float = 30.0,
                             lora_name: Optional[str] = None,
                             lora_strength: float = 1.0) -> dict:
    """FLUX.1-Fill-dev 按 mask 修补的 workflow(ComfyUI API 格式)。

    mask 图的红通道即重绘区域(白=修补,黑=保留)。
    lora_name 非空时在 UNET 后挂模型侧 LoRA(LoraLoaderModelOnly)。
    """
    model_ref = ["lora", 0] if lora_name else ["unet", 0]
    workflow = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": FLUX_FILL_UNET_NAME,
                            "weight_dtype": "default"}},
        "clip": {"class_type": "DualCLIPLoader",
                 "inputs": {"clip_name1": "clip_l.safetensors",
                            "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "load": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "scale": {"class_type": "ImageScale",
                  "inputs": {"image": ["load", 0], "upscale_method": "lanczos",
                             "width": width, "height": height, "crop": "disabled"}},
        "loadmask": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "maskscale": {"class_type": "ImageScale",
                      "inputs": {"image": ["loadmask", 0], "upscale_method": "bilinear",
                                 "width": width, "height": height, "crop": "disabled"}},
        "tomask": {"class_type": "ImageToMask",
                   "inputs": {"image": ["maskscale", 0], "channel": "red"}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip", 0], "text": prompt}},
        "guide": {"class_type": "FluxGuidance",
                  "inputs": {"conditioning": ["pos", 0], "guidance": guidance}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip", 0], "text": ""}},
        "inpaint": {"class_type": "InpaintModelConditioning",
                    "inputs": {"positive": ["guide", 0], "negative": ["neg", 0],
                               "vae": ["vae", 0], "pixels": ["scale", 0],
                               "mask": ["tomask", 0], "noise_mask": True}},
        "sample": {"class_type": "KSampler",
                   "inputs": {"model": model_ref, "positive": ["inpaint", 0],
                              "negative": ["inpaint", 1], "latent_image": ["inpaint", 2],
                              "seed": seed, "steps": steps, "cfg": 1.0,
                              "sampler_name": "euler", "scheduler": "simple",
                              "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": "flux_fill"}},
    }
    if lora_name:
        workflow["lora"] = {"class_type": "LoraLoaderModelOnly",
                            "inputs": {"model": ["unet", 0], "lora_name": lora_name,
                                       "strength_model": lora_strength}}
    return workflow


def _fill_enclosed_holes(mask: Image.Image) -> Image.Image:
    """封闭 mask 内部的孔洞:被 mask 区域完全包围的透明小孔一律并入 mask。

    SAM2 的分割 mask 在 icon 内部常留针孔,导致修补时残留 icon 碎片像素;
    从图像外部做 floodfill,凡是外部不可达的非 mask 像素都是内部孔,补成 mask。
    """
    w, h = mask.size
    # 四周垫 1px 黑边,保证 (0,0) 一定属于外部背景
    padded = ImageOps.expand(mask.point(lambda v: 255 if v > 0 else 0), border=1, fill=0)
    ImageDraw.floodfill(padded, (0, 0), 128)
    # 仍为 0 的像素 = 外部不可达的内部孔
    holes = padded.point(lambda v: 255 if v == 0 else 0).crop((1, 1, w + 1, h + 1))
    return ImageChops.lighter(mask, holes)


def _build_fill_mask(run_dir: Path, payload: dict, size: tuple) -> Image.Image:
    """生成修补 mask(白=重绘区域)。

    默认从 mask_from(如 icons.png)的 alpha 通道取:alpha>0 的像素即要修补的区域;
    mask_from_holes 则相反:取图片的透明区(alpha≈0,如 mid_hole.png 的破洞)为修补区;
    也可用 payload.mask 直接指定一张灰度 mask 图(白=修补)。
    grow_mask 外扩、mask_blur 羽化边缘,单位都是像素(按目标尺寸)。
    """
    if payload.get("mask"):
        src = run_dir / payload["mask"]
        if not src.is_file():
            raise FileNotFoundError(f"mask file not found: {src}")
        with Image.open(src) as m:
            mask = m.convert("L").resize(size, Image.Resampling.BILINEAR)
    elif payload.get("mask_from_holes"):
        src = run_dir / payload["mask_from_holes"]
        if not src.is_file():
            raise FileNotFoundError(f"mask_from_holes file not found: {src}")
        with Image.open(src) as m:
            if "A" not in m.getbands():
                raise ValueError(f"{payload['mask_from_holes']} has no alpha channel")
            alpha = m.getchannel("A").resize(size, Image.Resampling.BILINEAR)
        mask = alpha.point(lambda a: 255 if a < 128 else 0)
    else:
        mask_from = payload.get("mask_from") or "icons.png"
        src = run_dir / mask_from
        if not src.is_file():
            raise FileNotFoundError(f"mask_from file not found: {src}")
        with Image.open(src) as m:
            if "A" not in m.getbands():
                raise ValueError(f"{mask_from} has no alpha channel, "
                                 "pass an explicit grayscale mask via payload.mask")
            alpha = m.getchannel("A").resize(size, Image.Resampling.BILINEAR)
        mask = alpha.point(lambda a: 255 if a > 0 else 0)

    # 先封孔再外扩:内部针孔并入重绘区,避免 icon 残片留给 Fill 模型
    if payload.get("fill_holes", True):
        mask = _fill_enclosed_holes(mask)
    grow = int(payload.get("grow_mask", 8))
    if grow > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=2 * grow + 1))
    blur = float(payload.get("mask_blur", 4))
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    return mask.convert("RGB")  # ImageToMask 读红通道


def handle_flux_fill(payload: dict) -> dict:
    """精准修补(FLUX.1-Fill-dev):只重绘 mask 区域,其余像素保持原样。

    payload:
        run_id / dir     图片所在目录(二选一)
        image            输入图片名,默认 text_back.png
        mask_from        默认 icons.png,取其 alpha>0 的区域作为修补区
        mask_from_holes  取指定图片的透明区(alpha≈0)为修补区,如 mid_hole.png 的破洞
        mask             或直接指定灰度 mask 图片名(白=修补),优先级最高
        output           输出图片名,默认 inpainted.png
        prompt           默认 "clean UI background, seamless"
        seed / steps     默认 5 / 20
        guidance         默认 30(Fill 官方推荐)
        lora             模型侧 LoRA 名,默认 icon_back_fill.safetensors;传 "" 禁用
        lora_strength    LoRA 强度,默认 1.0
        grow_mask        mask 外扩像素,默认 8
        mask_blur        mask 边缘羽化,默认 4
        max_pixels       默认 1048576
        hole_output      可选,传文件名(如 icon_hole.png)则顺手输出"破洞图":
                         输入图 + 修补区变透明(用原始 alpha 硬抠,不含 grow/blur)
    """
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    image = payload.get("image") or "text_back.png"
    origin = run_dir / image
    if not origin.is_file():
        raise FileNotFoundError(f"input image not found: {origin}")

    seed = int(payload.get("seed", 5))
    steps = int(payload.get("steps", 20))
    prompt = payload.get("prompt") or "clean UI background, seamless"
    guidance = float(payload.get("guidance", 30.0))
    max_pixels = int(payload.get("max_pixels", 1048576))

    with Image.open(origin) as img:
        w, h = img.size
    tw, th = keep_size_16(w, h, max_pixels)

    started = time.time()
    hole_path = None
    if payload.get("hole_output"):
        hole_path = _write_hole_image(run_dir, payload, origin)
    mask_img = _build_fill_mask(run_dir, payload, (tw, th))
    # 最终修补 mask 落盘,方便排查"孔洞残留/花纹脑补"类问题
    mask_debug_path = run_dir / "fill_mask.png"
    mask_img.save(mask_debug_path)
    image_name = comfy.place_input_image(origin, prefix="fill_img_")
    mask_name = comfy.place_input_pil(mask_img, prefix="fill_mask_")

    lora_name = payload.get("lora", ICON_BACK_LORA_NAME) or None
    workflow = build_flux_fill_workflow(
        image_name=image_name, mask_name=mask_name, prompt=prompt,
        seed=seed, steps=steps, width=tw, height=th, guidance=guidance,
        lora_name=lora_name,
        lora_strength=float(payload.get("lora_strength", 1.0)),
    )
    entry = comfy.run_workflow(workflow)
    images = comfy.output_image_paths(entry)
    if not images:
        raise RuntimeError(f"ComfyUI finished but produced no output image "
                           f"(prompt_id={entry.get('prompt_id')})")

    output_path = run_dir / (payload.get("output") or "inpainted.png")
    shutil.copyfile(images[0], output_path)
    result = {
        "output_path": str(output_path),
        "size": [tw, th],
        "mask_path": str(mask_debug_path),
        "prompt_id": entry.get("prompt_id"),
        "elapsed_sec": round(time.time() - started, 1),
    }
    if hole_path:
        result["hole_path"] = str(hole_path)
    return result


def _write_hole_image(run_dir: Path, payload: dict, origin: Path) -> Path:
    """生成破洞图:输入图上把修补区抠成透明(原始 alpha 硬抠,不含 grow/blur)。"""
    if payload.get("mask"):
        with Image.open(run_dir / payload["mask"]) as m:
            region = m.convert("L").point(lambda v: 255 if v > 127 else 0)
    else:
        mask_from = payload.get("mask_from") or "icons.png"
        with Image.open(run_dir / mask_from) as m:
            if "A" not in m.getbands():
                raise ValueError(f"{mask_from} has no alpha channel")
            region = m.getchannel("A").point(lambda a: 255 if a > 0 else 0)

    with Image.open(origin) as img:
        base = img.convert("RGBA")
    if region.size != base.size:
        region = region.resize(base.size, Image.Resampling.NEAREST)
    # 洞内 alpha=0,洞外保留原 alpha
    keep = region.point(lambda v: 0 if v > 0 else 255)
    base.putalpha(ImageChops.darker(base.getchannel("A"), keep))
    hole_path = run_dir / payload["hole_output"]
    base.save(hole_path, format="PNG")
    return hole_path


def handle_mid_hole(payload: dict) -> dict:
    """中景层破洞图:image 上把各 source 图 alpha>0 的区域抠成透明。

    payload:
        run_id / dir   图片所在目录(二选一)
        image          底图,默认 icon_back.png(第 8 步结果)
        sources        要挖掉的图层,默认 ["assets.png", "bar.png", "button.png"],
                       不存在的自动跳过(至少要有一个存在)
        output         输出文件名,默认 mid_hole.png
        grow           洞外扩(腐蚀保留区)像素,默认 0:洞按各层 alpha 原样挖;
                       >0 时洞向外多挖 N 像素,吃掉元素边缘残留
        fill_rgb       可选 [r,g,b]:洞内 RGB 也替换为该色(alpha 仍为 0)。
                       串行提取的中间图必须传(如 [0,0,0]),否则 SAM2 通过 RGB
                       仍能"看见"已移除的元素,起不到排除干扰的作用
    """
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    image = payload.get("image") or "icon_back.png"
    base_path = run_dir / image
    if not base_path.is_file():
        raise FileNotFoundError(f"base image not found: {base_path}")
    sources = payload.get("sources") or ["assets.png", "bar.png", "button.png"]

    started = time.time()
    with Image.open(base_path) as img:
        base = img.convert("RGBA")

    keep = Image.new("L", base.size, 255)  # 255=保留,0=挖掉
    used = []
    for name in sources:
        src = run_dir / name
        if not src.is_file():
            continue
        with Image.open(src) as m:
            if "A" not in m.getbands():
                continue
            alpha = m.getchannel("A")
        if alpha.size != base.size:
            alpha = alpha.resize(base.size, Image.Resampling.NEAREST)
        keep = ImageChops.darker(keep, alpha.point(lambda v: 0 if v > 0 else 255))
        used.append(name)
    if not used:
        raise FileNotFoundError(
            f"没有可用的中景层图({'/'.join(sources)} 均不存在),请先完成提取")

    grow = int(payload.get("grow", 0))
    if grow > 0:
        # 腐蚀保留区 = 洞向外扩 grow 像素
        keep = keep.filter(ImageFilter.MinFilter(size=2 * grow + 1))
    base.putalpha(ImageChops.darker(base.getchannel("A"), keep))
    fill_rgb = payload.get("fill_rgb")
    if fill_rgb:
        hole_mask = keep.point(lambda v: 255 if v == 0 else 0)
        base.paste((int(fill_rgb[0]), int(fill_rgb[1]), int(fill_rgb[2]), 0),
                   mask=hole_mask)
    output_path = run_dir / (payload.get("output") or "mid_hole.png")
    base.save(output_path, format="PNG")
    return {
        "output_path": str(output_path),
        "sources": used,
        "elapsed_sec": round(time.time() - started, 1),
    }


def handle_omnipsd(payload: dict) -> dict:
    # TODO: 加载 OmniPSD 模型推理,输出写到 get_run_dir(payload["run_id"]) / "omnipsd"
    raise NotImplementedError("omnipsd not implemented yet")


def handle_yolo(payload: dict) -> dict:
    """YOLO UI 元素检测(常驻 daemon):返回 save_txt 同款格式的标注行。

    payload:
        run_id / dir     图片所在目录(二选一)
        image            输入图片名,默认 origin.png
        imgsz / conf / iou   默认 1333 / 0.1 / 0.7
    结果同时写 <run_dir>/yolo.txt 留档。
    """
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    image = payload.get("image") or "origin.png"
    if not (run_dir / image).is_file():
        raise FileNotFoundError(f"input image not found: {run_dir / image}")

    started = time.time()
    result = yoloc.detect({
        "dir": str(run_dir),
        "image": image,
        "imgsz": payload.get("imgsz", 1600),
        "conf": payload.get("conf", 0.05),
        "iou": payload.get("iou", 0.7),
        "augment": payload.get("augment", False),
        "slice": payload.get("slice", False),
        "slice_size": payload.get("slice_size", 640),
    })
    txt_path = run_dir / "yolo.txt"
    txt_path.write_text("\n".join(result.get("lines", [])) + "\n", encoding="utf-8")
    result["txt_path"] = str(txt_path)
    result["elapsed_sec"] = round(time.time() - started, 1)
    return result


def handle_sam2(payload: dict) -> dict:
    """SAM2 抠图(常驻 daemon):border 数组作为 box+正负点提示,输出透明 PNG。

    payload:
        run_id / dir       图片所在目录(二选一)
        image              输入图片名,默认 text_back.png
        output             输出图片名,默认 cutout.png(写回同目录)
        borders            必填,结构化检测结果格式:
                           [{"bbox":[cx,cy,w,h] 归一化, "positive_points":[[x,y],...],
                             "negative_points":[[x,y],...]}, ...]
        padding_ratio      默认 0.02,每边按框尺寸比例外扩
        min_padding        默认 1,每边最小外扩像素
        mask_threshold     默认 0.5
        feather_radius     默认 0(硬边)
        multimask          默认 False
        crop_scale         默认 1.5,每个 icon 裁 bbox 的 N 倍切片单独分割;<=1 整图模式
        refine             默认 True,首轮 mask 作为 mask_input 再精化一轮
        alt_image          可选备选源图名(如 origin.png)。提供后 border 可带
                           source 字段:primary/alt/auto(auto=双源按 SAM2 自评分择优);
                           结果的 sources 字段记录逐 border 择源情况
    """
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    image = payload.get("image") or "text_back.png"
    if not (run_dir / image).is_file():
        raise FileNotFoundError(f"input image not found: {run_dir / image}")
    borders = payload.get("borders")
    if not isinstance(borders, list) or not borders:
        raise ValueError("payload.borders must be a non-empty array")
    alt_image = payload.get("alt_image")
    if alt_image and not (run_dir / alt_image).is_file():
        raise FileNotFoundError(f"alt image not found: {run_dir / alt_image}")

    started = time.time()
    result = sam2c.cutout({
        "dir": str(run_dir),
        "image": image,
        "alt_image": alt_image,
        "output": payload.get("output") or "cutout.png",
        "borders": borders,
        "padding_ratio": payload.get("padding_ratio", 0.02),
        "min_padding": payload.get("min_padding", 1),
        "mask_threshold": payload.get("mask_threshold", 0.5),
        "feather_radius": payload.get("feather_radius", 0),
        "multimask": payload.get("multimask", False),
        "crop_scale": payload.get("crop_scale", 1.5),
        "refine": payload.get("refine", True),
        "fill_holes": payload.get("fill_holes", True),
        "size_rules": payload.get("size_rules") or [],
    })
    result["elapsed_sec"] = round(time.time() - started, 1)
    return result


def register_all() -> None:
    worker.register("hello", handle_hello)
    worker.register("text_back", handle_text_back)
    worker.register("text_back_cold", handle_text_back_cold)
    worker.register("comfy_workflow", handle_comfy_workflow)
    worker.register("flux_fill", handle_flux_fill)
    worker.register("mid_hole", handle_mid_hole)
    worker.register("omnipsd", handle_omnipsd)
    worker.register("yolo", handle_yolo)
    worker.register("sam2", handle_sam2)
