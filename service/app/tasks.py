"""任务 handler 注册处。

每个 handler 在 GPU worker 线程里串行执行,签名: handler(payload: dict) -> Any(可 JSON 序列化)。
payload 里约定带 run_id,handler 用 storage.get_run_dir(run_id) 拿到目录,把输出写回去。

后续在这里实现 omnipsd / yolo / sam2 的真实逻辑,模型权重建议模块级加载一次、常驻显存。
"""
import csv
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


DEFAULT_RESIDUAL_FILL_PROMPT = (
    "Fill the masked regions with clean UI background, seamlessly matching "
    "the surrounding colors, textures and patterns. No text, no letters, "
    "no numbers, no symbols."
)


def _remove_residual_text(run_dir: Path, image_path: Path, size: tuple,
                          conf_min: float, grow: int, seed: int, steps: int,
                          lora_name, guidance: float) -> dict:
    """残字复检与清除:对合成完的去字图再跑一次 YOLO(结果不落盘),
    若仍检出 text,用 SAM2 抠出文字 mask,flux_fill 补洞后同尺寸 mask 内回贴。

    整个过程只改 mask 内像素:回贴与输入同尺寸,零重采样,不糊不色偏。
    """
    det = yoloc.detect({
        "dir": str(run_dir), "image": image_path.name,
        "imgsz": 1600, "conf": 0.05, "iou": 0.7,
        "augment": False, "slice": False, "slice_size": 640,
    })
    boxes = []
    for line in det.get("lines", []):
        parts = line.split()
        if len(parts) < 5 or parts[0] != "0":
            continue
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        if conf >= conf_min:
            boxes.append([float(v) for v in parts[1:5]])
    if not boxes:
        return {"found": 0, "filled": False}

    tw, th = size
    # SAM2 抠残字的精确 mask;失败或抠空时退化为整框矩形
    mask = None
    tmp_name = "_residual_text.png"
    try:
        sam2c.cutout({
            "dir": str(run_dir), "image": image_path.name, "output": tmp_name,
            "borders": [{"bbox": b, "positive_points": [],
                         "negative_points": []} for b in boxes],
            "padding_ratio": 0.05, "min_padding": 2,
            "mask_threshold": 0.5, "feather_radius": 0,
            "multimask": False, "crop_scale": 1.5, "refine": True,
            "fill_holes": True, "size_rules": [],
        })
        tmp_path = run_dir / tmp_name
        if tmp_path.is_file():
            with Image.open(tmp_path) as c:
                alpha = c.convert("RGBA").getchannel("A")
            m = alpha.point(lambda a: 255 if a > 0 else 0)
            if m.getbbox():
                mask = m
            tmp_path.unlink(missing_ok=True)
    except RuntimeError:
        pass
    if mask is None:
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        for cx, cy, w, h in boxes:
            draw.rectangle([(cx - w / 2) * tw - 2, (cy - h / 2) * th - 2,
                            (cx + w / 2) * tw + 2, (cy + h / 2) * th + 2],
                           fill=255)
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    if grow > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=2 * grow + 1))
    soft = mask.filter(ImageFilter.GaussianBlur(2))
    mask_rgb = soft.convert("RGB")  # ImageToMask 读红通道

    image_name = comfy.place_input_image(image_path, prefix="residual_")
    mask_name = comfy.place_input_pil(mask_rgb, prefix="residual_mask_")
    entry = comfy.run_workflow(build_flux_fill_workflow(
        image_name=image_name, mask_name=mask_name,
        prompt=DEFAULT_RESIDUAL_FILL_PROMPT, seed=seed, steps=steps,
        width=tw, height=th, guidance=guidance, lora_name=lora_name))
    images = comfy.output_image_paths(entry)
    if not images:
        return {"found": len(boxes), "filled": False,
                "error": "fill produced no output"}
    with Image.open(images[0]) as g:
        gen = g.convert("RGB")
        if gen.size != size:
            gen = gen.resize(size, Image.Resampling.LANCZOS)
    with Image.open(image_path) as b:
        base = b.convert("RGB")
    # 补洞区先向底图对齐色彩,再仅在 mask 内回贴(同尺寸,零重采样)
    gen, _ = _match_colors_to_input(gen, base, mask_rgb)
    Image.composite(gen, base, soft).save(image_path)
    return {"found": len(boxes), "filled": True,
            "prompt_id": entry.get("prompt_id")}


def handle_text_back(payload: dict) -> dict:
    """去字模型(常驻 ComfyUI 版):读 <run_dir>/origin.png,输出 <run_dir>/text_back.png。

    payload: run_id(必填,或用 dir 直接指定目录)、seed、steps、prompt、
             max_pixels、guidance、lora(均可选)。
             protect(默认 True):保护合成——只有 YOLO text 框内取重生成像素,
             其余保留原图,icon 物理上不可能被误删;
             protect_grow(默认 8)文字框外扩 px、protect_feather(默认 4)边缘羽化、
             protect_conf(默认 0.2)text 框置信度门槛(太低的框不给重生成权,
             防止误检的"假文字"框让 icon 失去保护)。
             residual_check(默认 True):合成后残字复检——再跑一次 YOLO(不落盘),
             若仍有 text(conf>=residual_conf,默认 0.3),SAM2 抠出文字 mask
             外扩 residual_grow(默认 4)px,flux_fill 补洞(默认挂 icon_back LoRA,
             residual_lora 传空串禁用),补完仅 mask 内回贴,得到最终 text_back。
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
    # 中间态写暂存名,残字处理完再原子替换:text_back.png 这个名字
    # 从头到尾只出现最终图,杜绝前端在补洞窗口期缓存到中间态
    stage_path = run_dir / "_text_back_stage.png"
    if protect:
        # mask 白区取重生成像素,黑区保留原图像素
        Image.composite(gen, base, mask).save(stage_path)
    else:
        shutil.copyfile(images[0], stage_path)

    # 残字复检:合成完再 YOLO 一遍(不落盘),还有 text 就 SAM2 抠掉 + Fill 补洞
    residual = None
    if bool(payload.get("residual_check", True)):
        residual = _remove_residual_text(
            run_dir, stage_path, (tw, th),
            conf_min=float(payload.get("residual_conf", 0.3)),
            grow=int(payload.get("residual_grow", 4)),
            seed=seed,
            steps=int(payload.get("residual_steps", 20)),
            lora_name=payload.get("residual_lora", ICON_BACK_LORA_NAME) or None,
            guidance=float(payload.get("residual_guidance", 30.0)),
        )
    os.replace(stage_path, output_path)
    return {
        "output_path": str(output_path),
        "size": [tw, th],
        "protect": protect,
        "protected_text_boxes": protected_boxes,
        "residual_text": residual,
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


def _match_colors_to_input(gen: Image.Image, ref: Image.Image,
                           mask_rgb: Image.Image):
    """外域色彩匹配:对冲 VAE 往返的系统性色偏(表现为洞外整体变淡/去饱和)。

    以洞外区域为样本,统计生成图 vs 输入图的逐通道均值/方差偏移,求逆仿射
    应用到整张生成图——洞外拉回原色,洞内跟随同一变换与周围保持一致。
    纯逐像素查表,零重采样,不会引入任何模糊。
    返回 (校正后图像, 逐通道均值偏移) ;洞外占比过小时不校正。
    """
    from PIL import ImageStat
    outside = mask_rgb.getchannel("R").point(lambda v: 255 if v < 16 else 0)
    if ImageStat.Stat(outside).mean[0] < 2:  # 洞外不足 ~1%,统计不可靠
        return gen, None
    ref_stat = ImageStat.Stat(ref, outside)
    gen_stat = ImageStat.Stat(gen, outside)
    bands = []
    shifts = []
    for i, band in enumerate(gen.split()):
        mg, mr = gen_stat.mean[i], ref_stat.mean[i]
        sg, sr = gen_stat.stddev[i], ref_stat.stddev[i]
        # 方差比夹在温和区间,防止病态放大
        scale = max(0.8, min(1.3, (sr / sg) if sg > 1e-3 else 1.0))
        lut = [min(255, max(0, round((v - mg) * scale + mr))) for v in range(256)]
        bands.append(band.point(lut))
        shifts.append(round(mr - mg, 2))
    return Image.merge("RGB", bands), shifts


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
        color_match      默认 True:外域色彩匹配——按洞外区域统计并对冲 VAE 往返
                         的整体色偏/去饱和;纯查表零重采样,不会引入模糊
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
    color_shift = None
    if bool(payload.get("color_match", True)):
        with Image.open(images[0]) as g:
            gen = g.convert("RGB")
        with Image.open(origin) as o:
            ref = o.convert("RGB").resize(gen.size, Image.Resampling.LANCZOS)
        mask_for_stat = (mask_img if mask_img.size == gen.size
                         else mask_img.resize(gen.size, Image.Resampling.NEAREST))
        corrected, color_shift = _match_colors_to_input(gen, ref, mask_for_stat)
        corrected.save(output_path)
    else:
        shutil.copyfile(images[0], output_path)
    result = {
        "output_path": str(output_path),
        "size": [tw, th],
        "color_shift": color_shift,
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
        model            可选,按 key 选权重:game0804_11m(默认)/ game0804_p2 / game0728_p2
        refine_bbox      默认 True:SAM2 几何回投——每个框跑一次分割,用 mask 的
                         紧致外接框替换 YOLO 框,治"检测框小了一截"的系统性偏差
        refine_classes   默认 [1,2,3,4](icon/assets/button/bar);text 不回投
                         (字形稀疏 mask 会把行框改小),panel 不回投(大框风险高)
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
        "model": payload.get("model"),
        "imgsz": payload.get("imgsz", 1600),
        "conf": payload.get("conf", 0.05),
        "iou": payload.get("iou", 0.7),
        "augment": payload.get("augment", False),
        "slice": payload.get("slice", False),
        "slice_size": payload.get("slice_size", 640),
    })

    lines = result.get("lines", [])
    if payload.get("refine_bbox", True) and lines:
        refine_classes = set(payload.get("refine_classes", [1, 2, 3, 4]))
        parsed = [line.split() for line in lines]
        idxs = [i for i, p in enumerate(parsed)
                if len(p) >= 5 and int(p[0]) in refine_classes]
        if idxs:
            refined = sam2c.refine_bboxes({
                "dir": str(run_dir), "image": image,
                "borders": [{"bbox": [float(v) for v in parsed[i][1:5]]}
                            for i in idxs],
            })
            n_refined = 0
            for i, rb in zip(idxs, refined.get("bboxes", [])):
                if rb.get("refined"):
                    parsed[i][1:5] = [f"{v:.6f}" for v in rb["bbox"]]
                    n_refined += 1
            lines = [" ".join(p) for p in parsed]
            result["lines"] = lines
            result["bbox_refined"] = n_refined

    txt_path = run_dir / "yolo.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


DEFAULT_ICON_REPAIR_PROMPT = (
    "修复这个游戏图标:图中可能存在残缺、涂抹混乱、无逻辑的破损区域,"
    "把它们恢复为完整、合理、符合图标语义的图形。"
    "严格保持图标原有的风格、配色、构图、轮廓和朝向不变,背景保持原样,"
    "不要添加任何文字。输出高清、边缘锐利的版本。"
)

# 备选提示词(payload.prompt 传入):在默认版基础上追加去马赛克指令,
# 对极小 icon(放大后块状化)效果显著,但对中等 icon 有诱发重绘幻觉的风险,
# 建议只对块状化的个别 icon 重跑时使用(2026-08-07 A/B 实测)。
ICON_REPAIR_DEPIXEL_CLAUSE = (
    "如果图像因放大而出现马赛克、像素块、锯齿或模糊,"
    "将其重绘为线条平滑、色块干净、细节清晰的高清版本。"
)


def build_qwen_edit_workflow(image_name: str, prompt: str, seed: int, steps: int,
                             cfg: float = 2.5, denoise: float = 1.0,
                             megapixels: float = 1.0) -> dict:
    """Qwen-Image-Edit 2511 指令编辑 workflow(ComfyUI API 格式)。

    输入图统一缩放到 megapixels 总像素再编辑:小 icon 裁块相当于先放大后重生成,
    修复顺手完成高清化。
    """
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "qwen_image_edit.safetensors",
                            "weight_dtype": "default"}},
        "shift": {"class_type": "ModelSamplingAuraFlow",
                  "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "cfgnorm": {"class_type": "CFGNorm",
                    "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                            "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader",
                "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "load": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "scale": {"class_type": "ImageScaleToTotalPixels",
                  "inputs": {"image": ["load", 0], "upscale_method": "lanczos",
                             "megapixels": megapixels, "resolution_steps": 16}},
        "pos": {"class_type": "TextEncodeQwenImageEdit",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0],
                           "image": ["scale", 0], "prompt": prompt}},
        "neg": {"class_type": "TextEncodeQwenImageEdit",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0],
                           "image": ["scale", 0], "prompt": ""}},
        "encode": {"class_type": "VAEEncode",
                   "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
        "sample": {"class_type": "KSampler",
                   "inputs": {"model": ["cfgnorm", 0], "positive": ["pos", 0],
                              "negative": ["neg", 0], "latent_image": ["encode", 0],
                              "seed": seed, "steps": steps, "cfg": cfg,
                              "sampler_name": "euler", "scheduler": "simple",
                              "denoise": denoise}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0],
                            "filename_prefix": "icon_repair"}},
    }


def handle_icon_repair(payload: dict) -> dict:
    """icon 批量修复:逐 icon 裁块 → Qwen-Image-Edit 修复+高清 → SAM2 抠图
    → 最小透明 PNG 存 <run_dir>/icon/,manifest.csv 记录精确回贴位置。

    payload:
        run_id / dir   二选一
        image          裁块源图,默认 text_back.png
        icons          可选 [{"bbox":[cx,cy,w,h] 归一化, "conf":可选}, ...];
                       不传则读 yolo.txt 里 class=1(icon)且 conf>=conf_min 的行
        conf_min       默认 0.25(仅 yolo.txt 来源生效)
        pad_ratio      默认 0.25,裁块上下文外扩(相对 icon 长边);min_pad 默认 16px
        prompt / seed / steps / cfg / denoise / megapixels   Qwen 编辑参数
        color_match    默认 True,编辑结果向原裁块整体色彩匹配(对冲 VAE 色偏)
        max_icons      可选,只处理前 N 个(调试用)
        keep_debug     默认 True,修复后的完整裁块留在 icon/_debug/ 供检查

    输出:
        icon/icon_NN.png    最小透明 PNG(高清尺寸,通常大于原 icon)
        icon/manifest.csv   回贴表:把 png 缩放到 paste_w×paste_h、
                            贴到源图 (paste_x,paste_y) 即精确拼回
        icon/recompose.png  按 manifest 拼回的整层透明预览
    """
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    src_name = payload.get("image") or "text_back.png"
    src_path = run_dir / src_name
    if not src_path.is_file():
        raise FileNotFoundError(f"input image not found: {src_path}")
    with Image.open(src_path) as im:
        src = im.convert("RGB")
    sw, sh = src.size

    icons = payload.get("icons")
    if not icons:
        txt = run_dir / "yolo.txt"
        if not txt.is_file():
            raise ValueError("payload.icons missing and yolo.txt not found")
        conf_min = float(payload.get("conf_min", 0.25))
        icons = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5 and p[0] == "1":
                conf = float(p[5]) if len(p) > 5 else 1.0
                if conf >= conf_min:
                    icons.append({"bbox": [float(v) for v in p[1:5]],
                                  "conf": conf})
    if not icons:
        raise ValueError("no icons to repair")
    if payload.get("max_icons"):
        icons = icons[: int(payload["max_icons"])]

    pad_ratio = float(payload.get("pad_ratio", 0.25))
    min_pad = int(payload.get("min_pad", 16))
    prompt = payload.get("prompt") or DEFAULT_ICON_REPAIR_PROMPT
    seed = int(payload.get("seed", 5))
    steps = int(payload.get("steps", 20))
    cfg = float(payload.get("cfg", 2.5))
    denoise = float(payload.get("denoise", 1.0))
    megapixels = float(payload.get("megapixels", 1.0))
    color_match = bool(payload.get("color_match", True))
    keep_debug = bool(payload.get("keep_debug", True))

    out_dir = run_dir / "icon"
    out_dir.mkdir(exist_ok=True)
    if keep_debug:
        (out_dir / "_debug").mkdir(exist_ok=True)

    def _one(i: int, ic: dict) -> dict:
        cx, cy, w, h = ic["bbox"]
        row = {"index": i, "file": "", "status": "ok",
               "cx": cx, "cy": cy, "w": w, "h": h,
               "conf": round(float(ic.get("conf", 1.0)), 4),
               "paste_x": 0, "paste_y": 0, "paste_w": 0, "paste_h": 0,
               "png_w": 0, "png_h": 0}
        bx0, by0 = (cx - w / 2) * sw, (cy - h / 2) * sh
        bx1, by1 = (cx + w / 2) * sw, (cy + h / 2) * sh
        pad = max(min_pad, round(max(bx1 - bx0, by1 - by0) * pad_ratio))
        x0, y0 = max(0, int(bx0 - pad)), max(0, int(by0 - pad))
        x1, y1 = min(sw, int(math.ceil(bx1 + pad))), min(sh, int(math.ceil(by1 + pad)))
        cw, ch = x1 - x0, y1 - y0
        if cw < 8 or ch < 8:
            row["status"] = "too_small"
            return row

        crop = src.crop((x0, y0, x1, y1))
        # 注意:megapixels 不要低于 ~1.0,Qwen-Image-Edit 按 1MP 训练,
        # 低分辨率采样会直接出糊块与幻觉(2026-08-07 实测)
        input_name = comfy.place_input_pil(crop, prefix=f"icon_repair_{i:02d}_")
        entry = comfy.run_workflow(build_qwen_edit_workflow(
            input_name, prompt, seed, steps, cfg, denoise, megapixels))
        images = comfy.output_image_paths(entry)
        if not images:
            row["status"] = "comfy_no_output"
            return row
        with Image.open(images[0]) as g:
            gen = g.convert("RGB")
        if color_match:
            ref = crop.resize(gen.size, Image.Resampling.LANCZOS)
            # 全黑 mask = 整图都算"洞外",按整块裁块统计色偏
            gen, _ = _match_colors_to_input(
                gen, ref, Image.new("RGB", gen.size, (0, 0, 0)))
        ow, oh = gen.size

        # SAM2 在修复后的高清裁块上抠图(归一化坐标不受缩放影响)
        work_name = f"icon/_work_{i:02d}.png"
        cut_name = f"icon/_cut_{i:02d}.png"
        gen.save(run_dir / work_name)
        nb = [((bx0 + bx1) / 2 - x0) / cw, ((by0 + by1) / 2 - y0) / ch,
              (bx1 - bx0) / cw, (by1 - by0) / ch]
        # 裁块四角必在 icon 框外(pad>0 保证),自动做负点,压制背景粘连
        neg_pts = [[0.02, 0.02], [0.98, 0.02], [0.02, 0.98], [0.98, 0.98]]
        tight = None
        cut = None
        try:
            sam2c.cutout({
                "dir": str(run_dir), "image": work_name, "output": cut_name,
                "borders": [{"bbox": nb, "positive_points": [],
                             "negative_points": neg_pts}],
                "padding_ratio": 0.02, "min_padding": 1,
                "mask_threshold": 0.5, "feather_radius": 0,
                "multimask": False, "crop_scale": 1.5, "refine": True,
                "fill_holes": True, "size_rules": [],
            })
            cut_path = run_dir / cut_name
            if cut_path.is_file():
                with Image.open(cut_path) as c:
                    cut = c.convert("RGBA")
                tight = cut.getchannel("A").getbbox()
        except RuntimeError:
            row["status"] = "sam2_error"

        if tight and cut is not None:
            icon_img = cut.crop(tight)
        else:
            # 兜底:抠图失败时用 bbox 区域不透明输出,保证拼回不缺件
            if row["status"] == "ok":
                row["status"] = "sam2_empty"
            fx, fy = ow / cw, oh / ch
            tight = (int((bx0 - x0) * fx), int((by0 - y0) * fy),
                     min(ow, int(math.ceil((bx1 - x0) * fx))),
                     min(oh, int(math.ceil((by1 - y0) * fy))))
            icon_img = gen.crop(tight).convert("RGBA")

        fname = f"icon_{i:02d}.png"
        icon_img.save(out_dir / fname)
        if keep_debug:
            gen.save(out_dir / "_debug" / f"repair_{i:02d}.png")
        for tmp in (run_dir / work_name, run_dir / cut_name):
            Path(tmp).unlink(missing_ok=True)

        # 高清输出坐标 → 源图坐标
        sx, sy = cw / ow, ch / oh
        row.update({
            "file": fname, "png_w": icon_img.width, "png_h": icon_img.height,
            "paste_x": int(round(x0 + tight[0] * sx)),
            "paste_y": int(round(y0 + tight[1] * sy)),
            "paste_w": max(1, int(round((tight[2] - tight[0]) * sx))),
            "paste_h": max(1, int(round((tight[3] - tight[1]) * sy))),
        })
        return row

    started = time.time()
    rows = []
    for i, ic in enumerate(icons):
        try:
            rows.append(_one(i, ic))
        except Exception as e:  # 单个失败不拖垮整批
            rows.append({"index": i, "file": "", "status": f"error: {e}",
                         "cx": ic["bbox"][0], "cy": ic["bbox"][1],
                         "w": ic["bbox"][2], "h": ic["bbox"][3],
                         "conf": round(float(ic.get("conf", 1.0)), 4),
                         "paste_x": 0, "paste_y": 0, "paste_w": 0,
                         "paste_h": 0, "png_w": 0, "png_h": 0})

    # Qwen 用完即卸,不常驻显存(free_after 传 false 可保留缓存连跑多批)
    if bool(payload.get("free_after", True)):
        comfy.free_models()

    fields = ["index", "file", "status", "cx", "cy", "w", "h", "conf",
              "paste_x", "paste_y", "paste_w", "paste_h", "png_w", "png_h"]
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    canvas = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    for row in rows:
        if row["file"] and row["paste_w"] > 0:
            with Image.open(out_dir / row["file"]) as p:
                piece = p.convert("RGBA").resize(
                    (row["paste_w"], row["paste_h"]), Image.Resampling.LANCZOS)
            canvas.alpha_composite(piece, (row["paste_x"], row["paste_y"]))
    canvas.save(out_dir / "recompose.png")

    n_ok = sum(1 for r in rows if r["file"])
    return {
        "count": len(rows), "ok": n_ok,
        "statuses": {s: sum(1 for r in rows if r["status"] == s)
                     for s in {r["status"] for r in rows}},
        "manifest": str(manifest_path),
        "recompose": str(out_dir / "recompose.png"),
        "source_size": [sw, sh],
        "elapsed_sec": round(time.time() - started, 1),
    }


# 单图版(默认,use_ref=False):无原图参照,定位成"忠实修复"而非创作,
# 防止模型在没有真相参照时自由发挥(实测会乱生成)
DEFAULT_ICON_ASSET_PROMPT_SINGLE = (
    "这是一张游戏UI图标\"{name}\"的抠图,放在纯{bg}色背景上。"
    "这是一个图像修复任务,不是创作任务。"
    "请对图中已有的图形做忠实的高清修复:提高清晰度、平整色块、锐化边缘,"
    "清除边缘残留的杂色、背景色块和碎片,修补明显的涂抹破损。"
    "必须保持图形的形状、比例、结构、配色、朝向与输入完全一致:"
    "不允许重新设计、简化、美化或替换任何部分,"
    "不要根据名称想象输入中不存在的细节。"
    "严禁添加输入中没有的任何元素——底座、托盘、支架、平台、"
    "阴影、倒影、光效、描边、装饰、文字一律不允许。"
    "背景保持纯{bg}色,纯净均匀,不得出现任何图形、色带或渐变。"
)

DEFAULT_ICON_ASSET_PROMPT = (
    "图2是从游戏UI(图1)中抠出来的图标\"{name}\",放在纯色背景上。"
    "它可能因为去文字处理导致局部纹理涂抹混乱,也可能因为抠图导致边缘粘连了背景杂色。"
    "请参考图1中它的原始形态,重新绘制这个图标:修复破损与混乱区域、清除边缘杂色,"
    "严格保持图标原有的风格、配色、构图、轮廓和朝向不变,"
    "只绘制这个图标本体(含其专属底座),忽略并清除图2中残留的其它图形碎片。"
    "输出高清、边缘锐利的版本。背景必须保持与图2完全相同的纯{bg}色,"
    "纯净均匀,不要在背景上添加任何图形、色带或阴影,不要添加任何文字。"
)


def build_qwen_edit_plus_workflow(image1_name: str, image2_name: str,
                                  prompt: str, seed: int, steps: int,
                                  cfg: float = 2.5, denoise: float = 1.0,
                                  megapixels: float = 1.0) -> dict:
    """Qwen-Image-Edit 2511 双图参照编辑(Plus 节点,ComfyUI API 格式)。

    image1 为参照(UI 上下文),image2 为编辑底图——采样 latent 取自 image2,
    输出尺寸跟随 image2(缩放到 megapixels)。
    """
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "qwen_image_edit.safetensors",
                            "weight_dtype": "default"}},
        "shift": {"class_type": "ModelSamplingAuraFlow",
                  "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "cfgnorm": {"class_type": "CFGNorm",
                    "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                            "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader",
                "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "load1": {"class_type": "LoadImage", "inputs": {"image": image1_name}},
        "load2": {"class_type": "LoadImage", "inputs": {"image": image2_name}},
        "scale2": {"class_type": "ImageScaleToTotalPixels",
                   "inputs": {"image": ["load2", 0], "upscale_method": "lanczos",
                              "megapixels": megapixels, "resolution_steps": 16}},
        "pos": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0],
                           "image1": ["load1", 0], "image2": ["scale2", 0],
                           "prompt": prompt}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0],
                           "image1": ["load1", 0], "image2": ["scale2", 0],
                           "prompt": ""}},
        "encode": {"class_type": "VAEEncode",
                   "inputs": {"pixels": ["scale2", 0], "vae": ["vae", 0]}},
        "sample": {"class_type": "KSampler",
                   "inputs": {"model": ["cfgnorm", 0], "positive": ["pos", 0],
                              "negative": ["neg", 0], "latent_image": ["encode", 0],
                              "seed": seed, "steps": steps, "cfg": cfg,
                              "sampler_name": "euler", "scheduler": "simple",
                              "denoise": denoise}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0],
                            "filename_prefix": "icon_asset"}},
    }


CHROMA_CANDIDATES = [(255, 0, 255), (0, 255, 0)]  # 品红 / 绿


def _pick_chroma(icon_rgba: Image.Image):
    """选一个离 icon 调色板最远的底色(防止图标撞色被泛洪击穿)。"""
    thumb = icon_rgba.convert("RGBA").resize((64, 64))
    pixels = [(r, g, b) for r, g, b, a in thumb.getdata() if a > 128]
    if not pixels:
        return CHROMA_CANDIDATES[0]

    def worst_dist(c):
        return min((r - c[0]) ** 2 + (g - c[1]) ** 2 + (b - c[2]) ** 2
                   for r, g, b in pixels)

    return max(CHROMA_CANDIDATES, key=worst_dist)


def _border_median_color(rgb: Image.Image) -> tuple:
    """输出图边框一圈像素的逐通道中位数——Qwen 会把底色"和谐"跑偏,
    以它实际画出来的底色为准,而不是我们发送时的底色。"""
    small = rgb.copy()
    small.thumbnail((256, 256))
    w, h = small.size
    data = small.load()
    px = []
    for x in range(w):
        px.append(data[x, 0])
        px.append(data[x, h - 1])
    for y in range(h):
        px.append(data[0, y])
        px.append(data[w - 1, y])
    mid = len(px) // 2
    return tuple(sorted(c[i] for c in px)[mid] for i in range(3))


def _unkey_border(img: Image.Image, tol: int = 60,
                  defringe: int = 1) -> Image.Image:
    """全局色键去底(底色自适应):
    1. 以"边框中位色"为键色(Qwen 会把底色画跑偏,以实际输出为准),
       全图凡接近键色的一律透明——底色本就按"离图标调色板最远"自适应挑选,
       全局键安全,且能清掉镂空 icon 内部的封闭底色区;
    2. 清除仍与边界连通的不透明残留(模型偶发的黑边条/杂色带),
       真图标居中且有 pad 不会贴边;若清完全空则回退不清;
    3. alpha 收缩 defringe 像素,消掉边缘 1px 混色晕。"""
    rgb = img.convert("RGB")
    key = _border_median_color(rgb)
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, key))
    # 每通道差值取最大近似色距;point 后 0=接近底色,255=图标内容
    r, g, b = diff.split()
    dist = ImageChops.lighter(ImageChops.lighter(r, g), b)
    binary = dist.point(lambda v: 0 if v <= tol else 255).convert("L")
    # 色相键补刀:模型常拿底色画投影/描边(同色相但明暗不同,RGB 距离抓不住),
    # 与键色同色相且饱和度高的一律并入底色。底色是按"远离图标调色板"自适应
    # 挑的,图标本体几乎不会撞到这个色相。
    hsv = rgb.convert("HSV")
    hch, sch, vch = hsv.getchannel("H"), hsv.getchannel("S"), hsv.getchannel("V")
    key_h = Image.new("RGB", (1, 1), key).convert("HSV").getpixel((0, 0))[0]
    hd = hch.point(lambda h: min(abs(h - key_h), 256 - abs(h - key_h)))
    hue_close = hd.point(lambda v: 255 if v <= 18 else 0)
    saturated = sch.point(lambda v: 255 if v >= 90 else 0)
    # 明度下限:深色像素的色相值不稳定(黑色描边会被误伤),V<60 一律豁免
    bright = vch.point(lambda v: 255 if v >= 60 else 0)
    hue_key = ImageChops.multiply(ImageChops.multiply(hue_close, saturated),
                                  bright)  # 255=命中色相键
    binary = ImageChops.subtract(binary, hue_key)  # 命中处归 0(视作底色)
    w, h = binary.size
    border = ([(x, y) for x in range(w) for y in (0, h - 1)]
              + [(x, y) for y in range(h) for x in (0, w - 1)])
    # 贴边的"内容"残留染 64(候删)
    for x, y in border:
        if binary.getpixel((x, y)) == 255:
            ImageDraw.floodfill(binary, (x, y), 64)
    hist = binary.histogram()
    junk_ok = hist[255] > w * h * 0.005  # 清完还有足量内容才生效
    alpha = binary.point(
        lambda v: 255 if (v == 255 or (not junk_ok and v == 64)) else 0)
    if defringe > 0:
        alpha = alpha.filter(ImageFilter.MinFilter(2 * defringe + 1))
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def handle_icon_asset(payload: dict) -> dict:
    """第 8 步素材化:每组 icon 出一张高清透明素材(组内成员共用)。

    链路:选组内最大成员 → 上下文裁块(图1) + SAM2 抠图合成纯色底(图2)
    → Qwen-Image-Edit 双图参照重绘 → 边界泛洪去底 → 最小透明 PNG。
    manifest.csv 记录组内每个成员的回贴矩形(fit-inside 居中,等比不拉伸)。

    payload:
        run_id / dir   二选一
        groups         必填 [{"name","slug","bbox":[[cx,cy,w,h],...]}, ...](归一化)
        image          上下文源图,默认 text_back.png
        cutout         SAM2 提取层,默认 icons.png(需先跑完提icon步骤)
        use_ref        (Qwen 重绘通道停用中,此参数暂不生效)
                       原语义:False=单图重绘,True=带上下文裁块双图参照
        pad_ratio / min_pad    裁块外扩,默认 0.3 / 16
        prompt         可选,{name} 占位符会被组名替换
        seed / steps / cfg / denoise / megapixels    Qwen 参数
        tol            泛洪容差,默认 60
        keep_debug     默认 True,重绘原图留在 icon_assets/_debug/

    输出:icon_assets/<slug>.png + manifest.csv + recompose.png
    """
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("payload.groups must be a non-empty array")

    src_path = run_dir / (payload.get("image") or "text_back.png")
    cut_path = run_dir / (payload.get("cutout") or "icons.png")
    for p in (src_path, cut_path):
        if not p.is_file():
            raise FileNotFoundError(f"input not found: {p}")
    with Image.open(src_path) as im:
        src = im.convert("RGB")
    with Image.open(cut_path) as im:
        cut_layer = im.convert("RGBA")
    sw, sh = src.size
    if cut_layer.size != src.size:
        cut_layer = cut_layer.resize(src.size, Image.Resampling.LANCZOS)

    use_ref = bool(payload.get("use_ref", False))
    pad_ratio = float(payload.get("pad_ratio", 0.3))
    min_pad = int(payload.get("min_pad", 16))
    seed = int(payload.get("seed", 5))
    steps = int(payload.get("steps", 20))
    cfg = float(payload.get("cfg", 2.5))
    denoise = float(payload.get("denoise", 1.0))
    megapixels = float(payload.get("megapixels", 1.0))
    tol = int(payload.get("tol", 60))
    keep_debug = bool(payload.get("keep_debug", True))

    out_dir = run_dir / "icon_assets"
    out_dir.mkdir(exist_ok=True)
    if keep_debug:
        (out_dir / "_debug").mkdir(exist_ok=True)

    def _px_rect(bbox):
        cx, cy, w, h = bbox
        return ((cx - w / 2) * sw, (cy - h / 2) * sh,
                (cx + w / 2) * sw, (cy + h / 2) * sh)

    def _one(group: dict) -> dict:
        slug = group["slug"]
        name = group.get("name", slug)
        bboxes = group["bbox"]
        # 选组内面积最大的成员当源(有效分辨率最高)
        rects = [_px_rect(b) for b in bboxes]
        bi = max(range(len(rects)),
                 key=lambda i: (rects[i][2] - rects[i][0]) * (rects[i][3] - rects[i][1]))
        bx0, by0, bx1, by1 = rects[bi]
        pad = max(min_pad, round(max(bx1 - bx0, by1 - by0) * pad_ratio))
        x0, y0 = max(0, int(bx0 - pad)), max(0, int(by0 - pad))
        x1 = min(sw, int(math.ceil(bx1 + pad)))
        y1 = min(sh, int(math.ceil(by1 + pad)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return {"slug": slug, "status": "too_small"}

        ctx_crop = src.crop((x0, y0, x1, y1))
        cut_crop = cut_layer.crop((x0, y0, x1, y1))
        # 只保留成员 bbox(外扩 15%)内的抠图像素:裁块 pad 里常混入邻近 icon
        # 的抠图,不隔离会被 Qwen 一起画进素材
        mm = max(8, round(max(bx1 - bx0, by1 - by0) * 0.15))
        keep = Image.new("L", cut_crop.size, 0)
        ImageDraw.Draw(keep).rectangle(
            [bx0 - x0 - mm, by0 - y0 - mm, bx1 - x0 + mm, by1 - y0 + mm],
            fill=255)
        cut_crop.putalpha(
            ImageChops.multiply(cut_crop.getchannel("A"), keep))

        # —— 当前通道:直接从抠图层(icons.png)裁块落库,Qwen 重绘已停用 ——
        tight = cut_crop.getchannel("A").getbbox()
        if not tight:
            return {"slug": slug, "status": "empty_cutout"}
        asset = cut_crop.crop(tight)
        status = "direct"
        bg = (0, 0, 0)
        fname = f"{slug}.png"
        asset.save(out_dir / fname)

        # —— Qwen 重绘通道(停用中,恢复时取消下面整段注释并删除上面的直裁段) ——
        # bg = _pick_chroma(cut_crop)
        # base = Image.new("RGB", cut_crop.size, bg)
        # base.paste(cut_crop, (0, 0), cut_crop)

        # img2 = comfy.place_input_pil(base, prefix=f"asset_cut_{slug[:24]}_")
        # bg_name = "品红" if bg == (255, 0, 255) else "绿"
        # if use_ref:
        #     img1 = comfy.place_input_pil(
        #         ctx_crop, prefix=f"asset_ctx_{slug[:24]}_")
        #     prompt = (payload.get("prompt") or DEFAULT_ICON_ASSET_PROMPT)
        #     workflow = build_qwen_edit_plus_workflow(
        #         img1, img2,
        #         prompt.replace("{name}", name).replace("{bg}", bg_name),
        #         seed, steps, cfg, denoise, megapixels)
        # else:
        #     prompt = (payload.get("prompt")
        #               or DEFAULT_ICON_ASSET_PROMPT_SINGLE)
        #     workflow = build_qwen_edit_workflow(
        #         img2, prompt.replace("{name}", name).replace("{bg}", bg_name),
        #         seed, steps, cfg, denoise, megapixels)
        # entry = comfy.run_workflow(workflow)
        # images = comfy.output_image_paths(entry)
        # if not images:
        #     return {"slug": slug, "status": "comfy_no_output"}
        # with Image.open(images[0]) as g:
        #     gen = g.convert("RGB")
        # if keep_debug:
        #     gen.save(out_dir / "_debug" / f"{slug}_raw.png")

        # rgba = _unkey_border(gen, tol, int(payload.get("defringe", 1)))
        # tight = rgba.getchannel("A").getbbox()
        # status = "ok"
        # if not tight:
        #     # 泛洪全键掉了(生成图整体接近底色)——退化为整图不透明
        #     status = "unkey_empty"
        #     rgba = gen.convert("RGBA")
        #     tight = (0, 0, rgba.width, rgba.height)
        # else:
        #     # 底色渗入检查:不透明占比过高说明没抠动,标记供人工复核
        #     area = (tight[2] - tight[0]) * (tight[3] - tight[1])
        #     opaque = sum(1 for a in rgba.crop(tight).getchannel("A").getdata()
        #                  if a > 0)
        #     if area > 0 and opaque / area > 0.98:
        #         status = "unkey_suspect"
        # asset = rgba.crop(tight)
        # asset.save(out_dir / f"{slug}.png")
        # 组内每个成员一条回贴记录:素材等比 fit 进各自 bbox,居中
        members = []
        aw, ah = asset.size
        for mi, (mx0, my0, mx1, my1) in enumerate(rects):
            bw, bh = mx1 - mx0, my1 - my0
            s = min(bw / aw, bh / ah) if aw and ah else 0
            pw, ph = max(1, round(aw * s)), max(1, round(ah * s))
            members.append({
                "member": mi, "bbox": bboxes[mi],
                "paste_x": int(round(mx0 + (bw - pw) / 2)),
                "paste_y": int(round(my0 + (bh - ph) / 2)),
                "paste_w": pw, "paste_h": ph,
            })
        return {"slug": slug, "name": name, "status": status,
                "file": f"{slug}.png", "png_w": aw, "png_h": ah,
                "source_member": bi, "bg": f"{bg[0]},{bg[1]},{bg[2]}",
                "members": members}

    started = time.time()
    # 先清掉本批将要生成的旧素材:前端靠"文件出现"做逐个即时显示,
    # 旧文件残留会被误认为新结果
    for group in groups:
        slug = group.get("slug")
        if slug:
            (out_dir / f"{slug}.png").unlink(missing_ok=True)
    results = []
    for group in groups:
        try:
            results.append(_one(group))
        except Exception as e:  # 单组失败不拖垮整批
            results.append({"slug": group.get("slug", "?"),
                            "status": f"error: {e}"})

    fields = ["slug", "name", "status", "file", "png_w", "png_h", "bg",
              "member", "cx", "cy", "w", "h",
              "paste_x", "paste_y", "paste_w", "paste_h"]
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            base_row = {k: r.get(k, "") for k in
                        ("slug", "name", "status", "file", "png_w", "png_h", "bg")}
            for m in r.get("members", [{}]):
                row = dict(base_row)
                if m:
                    row.update({
                        "member": m["member"],
                        "cx": m["bbox"][0], "cy": m["bbox"][1],
                        "w": m["bbox"][2], "h": m["bbox"][3],
                        "paste_x": m["paste_x"], "paste_y": m["paste_y"],
                        "paste_w": m["paste_w"], "paste_h": m["paste_h"],
                    })
                writer.writerow(row)

    n_ok = sum(1 for r in results if r.get("file"))
    return {
        "count": len(results), "ok": n_ok,
        "statuses": {s: sum(1 for r in results if r["status"] == s)
                     for s in {r["status"] for r in results}},
        "manifest": str(manifest_path),
        # 叠放显示数据:前端按 members 的回贴矩形绝对定位,不再服务端拼图
        "assets": [{"slug": r["slug"], "file": r["file"],
                    "status": r["status"], "members": r.get("members", [])}
                   for r in results if r.get("file")],
        "source_size": [sw, sh],
        "elapsed_sec": round(time.time() - started, 1),
    }


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
    worker.register("icon_repair", handle_icon_repair)
    worker.register("icon_asset", handle_icon_asset)
