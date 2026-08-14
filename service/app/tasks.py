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
        # mask_from 支持单文件名或数组:数组时逐层取 alpha 并集
        # (如第 9 步 icons.png + panel_f.png 一起挖洞)
        mask_from = payload.get("mask_from") or "icons.png"
        names = mask_from if isinstance(mask_from, list) else [mask_from]
        mask = None
        for name in names:
            src = run_dir / name
            if not src.is_file():
                raise FileNotFoundError(f"mask_from file not found: {src}")
            with Image.open(src) as m:
                if "A" not in m.getbands():
                    raise ValueError(f"{name} has no alpha channel, "
                                     "pass an explicit grayscale mask via payload.mask")
                alpha = m.getchannel("A").resize(size, Image.Resampling.BILINEAR)
            layer = alpha.point(lambda a: 255 if a > 0 else 0)
            mask = layer if mask is None else ImageChops.lighter(mask, layer)

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
        mask_from        默认 icons.png,取其 alpha>0 的区域作为修补区;
                         可传数组(如 ["icons.png","panel_f.png"]),逐层 alpha 并集
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
        # 与 _build_mask 同口径:mask_from 支持单文件名或数组(alpha 并集)
        mask_from = payload.get("mask_from") or "icons.png"
        names = mask_from if isinstance(mask_from, list) else [mask_from]
        region = None
        for name in names:
            with Image.open(run_dir / name) as m:
                if "A" not in m.getbands():
                    raise ValueError(f"{name} has no alpha channel")
                layer = m.getchannel("A").point(lambda a: 255 if a > 0 else 0)
            if region is None:
                region = layer
            elif layer.size != region.size:
                region = ImageChops.lighter(
                    region, layer.resize(region.size, Image.Resampling.NEAREST))
            else:
                region = ImageChops.lighter(region, layer)

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
        txt_output       留档文件名,默认 yolo.txt;探测类调用传独立名避免覆盖存档
    结果同时写 <run_dir>/<txt_output> 留档。
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

    # txt_output:探测类调用(如 12+ bar 裁块二次检测)传入独立文件名,
    # 避免覆盖 run 的 yolo.txt 检测存档
    txt_path = run_dir / (payload.get("txt_output") or "yolo.txt")
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
        restore_from       可选源图名(如 text_back.png):提取前在每个 border 框
                           邻域内把该图像素贴回提取源,恢复被前序挖洞破坏的自然
                           外观(串行提 bar 时,压在 bar 上的 icon/assets/panel
                           还原回去再分割,SAM2 不再面对黑洞残图)
        restore_margin_ratio  默认 0.08,还原区按框长边比例外扩
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
    restore_from = payload.get("restore_from")
    restore_tmp = None
    if restore_from:
        rf_path = run_dir / restore_from
        if not rf_path.is_file():
            raise FileNotFoundError(f"restore_from not found: {rf_path}")
        with Image.open(run_dir / image) as im:
            base_img = im.convert("RGB")
        with Image.open(rf_path) as im:
            rest = im.convert("RGB")
        if rest.size != base_img.size:
            rest = rest.resize(base_img.size, Image.Resampling.LANCZOS)
        bw_, bh_ = base_img.size
        mr = float(payload.get("restore_margin_ratio", 0.08))
        for b in borders:
            cx, cy, w, h = (float(v) for v in b["bbox"][:4])
            mm = max(4, round(max(w * bw_, h * bh_) * mr))
            rx0 = max(0, int((cx - w / 2) * bw_) - mm)
            ry0 = max(0, int((cy - h / 2) * bh_) - mm)
            rx1 = min(bw_, int(math.ceil((cx + w / 2) * bw_)) + mm)
            ry1 = min(bh_, int(math.ceil((cy + h / 2) * bh_)) + mm)
            base_img.paste(rest.crop((rx0, ry0, rx1, ry1)), (rx0, ry0))
        restore_tmp = "_sam2_restore_src.png"
        base_img.save(run_dir / restore_tmp)
        image = restore_tmp

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
    if restore_tmp:
        (run_dir / restore_tmp).unlink(missing_ok=True)
        result["restored_from"] = restore_from
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
                  defringe: int = 1, key: tuple = None,
                  junk: bool = True) -> Image.Image:
    """全局色键去底(底色自适应):
    1. 以"边框中位色"为键色(Qwen 会把底色画跑偏,以实际输出为准),
       全图凡接近键色的一律透明——底色本就按"离图标调色板最远"自适应挑选,
       全局键安全,且能清掉镂空 icon 内部的封闭底色区;
    2. 清除仍与边界连通的不透明残留(模型偶发的黑边条/杂色带),
       真图标居中且有 pad 不会贴边;若清完全空则回退不清;
    3. alpha 收缩 defringe 像素,消掉边缘 1px 混色晕。"""
    rgb = img.convert("RGB")
    if key is None:
        key = _border_median_color(rgb)
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, key))
    # 每通道差值取最大近似色距;point 后 0=接近底色,255=图标内容
    r, g, b = diff.split()
    dist = ImageChops.lighter(ImageChops.lighter(r, g), b)
    # 软阈值 alpha:tol*0.7 ~ tol*1.3 线性过渡——抗锯齿边缘拿到渐变透明度,
    # 不再"非0即255"地留绿毛刺或啃出锯齿
    lo, hi = tol * 0.7, tol * 1.3
    ramp = [0 if v <= lo else 255 if v >= hi
            else int((v - lo) / (hi - lo) * 255) for v in range(256)]
    alpha_soft = dist.point(ramp).convert("L")
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
    # 贴边的"内容"残留染 64(候删);junk=False 时跳过(槽位直裁场景
    # 素材本来就可能贴着裁剪框)
    if junk:
        for x, y in border:
            if binary.getpixel((x, y)) == 255:
                ImageDraw.floodfill(binary, (x, y), 64)
    hist = binary.histogram()
    junk_ok = hist[255] > w * h * 0.005  # 清完还有足量内容才生效
    # 硬判定得到的"删除区"(底色/色相键命中/贴边残留)在软 alpha 上清零,
    # 其余位置保留软 alpha 的渐变边缘
    removed = binary.point(
        lambda v: 0 if (v == 255 or (not junk_ok and v == 64)) else 255)
    alpha = ImageChops.subtract(alpha_soft, removed.convert("L"))
    if defringe > 0:
        alpha = alpha.filter(ImageFilter.MinFilter(2 * defringe + 1))

    out = _despill_edges(rgb, alpha, key).convert("RGBA")
    out.putalpha(alpha)
    return out


def _despill_edges(rgb: Image.Image, alpha: Image.Image, key: tuple) -> Image.Image:
    """边缘退底色(despill):只在 alpha 过渡带内压掉渗入的键色,
    带外原样保留,不误伤与底色同色系的图形本体。"""
    band = alpha.point(lambda v: 255 if 0 < v < 255 else 0)
    band = band.filter(ImageFilter.MaxFilter(3))
    r0, g0, b0 = rgb.split()
    kr, kg, kb = key
    if kg >= kr and kg >= kb:
        # 绿键:G 超出 max(R,B) 的部分视为绿溢出,压平
        despilled = Image.merge(
            "RGB", (r0, ImageChops.darker(g0, ImageChops.lighter(r0, b0)), b0))
    else:
        # 品红键:min(R,B) 超出 G 的部分视为品红溢出,从 R、B 同时扣除
        excess = ImageChops.subtract(ImageChops.darker(r0, b0), g0)
        despilled = Image.merge(
            "RGB", (ImageChops.subtract(r0, excess), g0,
                    ImageChops.subtract(b0, excess)))
    return Image.composite(despilled, rgb, band)


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


def _connected_tiles(rgba: Image.Image, min_area_ratio: float = 0.0005):
    """绿底切割后的连通域分析:返回每个素材块的全分辨率紧致框列表。

    在 ≤512px 的缩略 alpha 上做 floodfill 标记(纯 PIL,C 实现,快),
    再映射回全分辨率并按 alpha 收紧。
    """
    w, h = rgba.size
    scale = min(1.0, 512 / max(w, h))
    sw_, sh_ = max(1, round(w * scale)), max(1, round(h * scale))
    small = rgba.getchannel("A").resize((sw_, sh_), Image.Resampling.NEAREST)
    binary = small.point(lambda v: 255 if v > 0 else 0)
    px = binary.load()
    label = 1
    for y in range(sh_):
        for x in range(sw_):
            if px[x, y] == 255 and label < 250:
                ImageDraw.floodfill(binary, (x, y), label)
                label += 1
    boxes = {}
    areas = {}
    for y in range(sh_):
        for x in range(sw_):
            v = px[x, y]
            if 0 < v < 255:
                b = boxes.get(v)
                if b is None:
                    boxes[v] = [x, y, x, y]
                else:
                    b[0], b[1] = min(b[0], x), min(b[1], y)
                    b[2], b[3] = max(b[2], x), max(b[3], y)
                areas[v] = areas.get(v, 0) + 1
    tiles = []
    alpha = rgba.getchannel("A")
    for v, (x0, y0, x1, y1) in boxes.items():
        if areas[v] < sw_ * sh_ * min_area_ratio:
            continue  # 噪点碎屑
        # 映射回全分辨率,外扩 2 个缩略像素再按 alpha 收紧
        pad = 2
        fx0 = max(0, int((x0 - pad) / scale))
        fy0 = max(0, int((y0 - pad) / scale))
        fx1 = min(w, int(math.ceil((x1 + 1 + pad) / scale)))
        fy1 = min(h, int(math.ceil((y1 + 1 + pad) / scale)))
        tight = alpha.crop((fx0, fy0, fx1, fy1)).getbbox()
        if not tight:
            continue
        tiles.append((fx0 + tight[0], fy0 + tight[1],
                      fx0 + tight[2], fy0 + tight[3]))
    return tiles


def _thumb_vec(img: Image.Image, size: int = 16):
    """16×16 RGB 色彩网格向量(RGBA 先合成到中性灰上)。"""
    if img.mode == "RGBA":
        base = Image.new("RGB", img.size, (128, 128, 128))
        base.paste(img, (0, 0), img)
        img = base
    return list(img.convert("RGB").resize((size, size)).getdata())


def _vec_dist(a, b) -> float:
    """平均每通道绝对差,归一到 0~1。"""
    total = sum(abs(pa[c] - pb[c]) for pa, pb in zip(a, b) for c in range(3))
    return total / (len(a) * 3 * 255)


def _panel_z_order(panel_rects, tiles_by_panel, src: Image.Image):
    """计算 panel 前后叠放次序(z 越大越靠上)。

    规则:
    1. 一个 panel 的边界完全在另一个里面 → 内者在上;
    2. 部分重合 → 取色裁决:取源图重叠区像素,分别与两个候选素材在
       该区域的对应画面比色,谁更像谁在上(重叠区显示的是上层的像素)。
    以两两裁决建有向边(下→上),Kahn 拓扑排序出 z;有环时按面积降序兜底
    (大的一般在下)。
    """
    n = len(panel_rects)

    def _inter(a, b):
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        return (x0, y0, x1, y1) if x1 - x0 > 2 and y1 - y0 > 2 else None

    def _contains(a, b):
        return (a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3])

    def _region_vec(tile, paste, region, size=8):
        """素材 tile 在源图坐标 region 处的对应画面(经回贴矩形映射)。"""
        px0, py0, pw, ph = paste
        if pw <= 0 or ph <= 0:
            return None
        rx0 = (region[0] - px0) / pw * tile.width
        ry0 = (region[1] - py0) / ph * tile.height
        rx1 = (region[2] - px0) / pw * tile.width
        ry1 = (region[3] - py0) / ph * tile.height
        rx0, ry0 = max(0, int(rx0)), max(0, int(ry0))
        rx1 = min(tile.width, int(math.ceil(rx1)))
        ry1 = min(tile.height, int(math.ceil(ry1)))
        if rx1 - rx0 < 1 or ry1 - ry0 < 1:
            return None
        return _thumb_vec(tile.crop((rx0, ry0, rx1, ry1)), size)

    edges = []  # (下, 上)
    for i in range(n):
        for j in range(i + 1, n):
            ov = _inter(panel_rects[i], panel_rects[j])
            if not ov:
                continue
            if _contains(panel_rects[i], panel_rects[j]):
                edges.append((i, j))  # j 完全在 i 内 → j 在上
                continue
            if _contains(panel_rects[j], panel_rects[i]):
                edges.append((j, i))
                continue
            # 部分重合:重叠区取色
            ti, tj = tiles_by_panel.get(i), tiles_by_panel.get(j)
            if not ti or not tj:
                continue
            region = (int(ov[0]), int(ov[1]), int(math.ceil(ov[2])),
                      int(math.ceil(ov[3])))
            src_vec = _thumb_vec(src.crop(region), 8)
            vi = _region_vec(ti[0], ti[1], region)
            vj = _region_vec(tj[0], tj[1], region)
            if vi is None or vj is None:
                continue
            di, dj = _vec_dist(src_vec, vi), _vec_dist(src_vec, vj)
            # 更像源图重叠区的在上
            edges.append((j, i) if di < dj else (i, j))

    # Kahn 拓扑;环则退化为面积降序
    from collections import defaultdict, deque
    indeg = [0] * n
    adj = defaultdict(set)
    for lo, hi in edges:
        if hi not in adj[lo]:
            adj[lo].add(hi)
            indeg[hi] += 1
    q = deque(sorted((i for i in range(n) if indeg[i] == 0),
                     key=lambda i: -((panel_rects[i][2] - panel_rects[i][0])
                                     * (panel_rects[i][3] - panel_rects[i][1]))))
    z = {}
    order = 0
    while q:
        u = q.popleft()
        z[u] = order
        order += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(z) < n:  # 有环,剩余按面积降序垫底后排
        rest = sorted((i for i in range(n) if i not in z),
                      key=lambda i: -((panel_rects[i][2] - panel_rects[i][0])
                                      * (panel_rects[i][3] - panel_rects[i][1])))
        for u in rest:
            z[u] = order
            order += 1
    return z


def handle_panel_asset(payload: dict) -> dict:
    """第 16+ 步:把 nano banana 的绿底平铺图切割成 panel 素材并匹配回原位。

    链路:自适应色键去绿底 → 连通域切块 → 与检测 panel 做
    "长宽比 + 16×16 色彩网格"双特征代价匹配(贪心全局最小)→
    素材存 panel_assets/p<idx>.png(idx=原 panel 下标),manifest 记录回贴矩形。

    payload:
        run_id / dir   二选一
        image          绿底平铺图,默认 panels_green.png
        source         视觉比对源图,默认 mid_fill.png
        panels         可选 [[cx,cy,w,h],...](归一化);缺省读 structure1.json
                       的 panel 字段(过滤 yolo_detect=discard)
        tol            色键容差,默认 60;defringe 默认 1
        supersample    默认 2:平铺图放大 N 倍后再色键切割,边缘更平滑;1=关闭
        layers         可选 [[panel下标,...],...]:分层模式——第 16 步逐层原位
                       生成的 panels_green_L<k>.png 按 bbox 直裁,无匹配环节
        canvas         分层模式配套 {size:[cw,ch], scale, offset:[ox,oy]}:
                       源图垫绿到 GPT 预设画布的变换,裁切在画布空间进行
        mask_mode      默认 chroma:纯色键软阈值出边;
                       传 "sam2" 改用 SAM2 按块出轮廓(实测圆角/边缘行为不合预期,已弃用为默认)
        sam2           可选,SAM2 出边参数覆盖:{padding_ratio, min_padding,
                       mask_threshold, feather_radius, crop_scale, refine,
                       multimask, fill_holes, grow(角部外扩px,默认2),
                       guard_grow(护栏外扩px,默认1——治四边被色键削掉的问题)}
    结果:count_panels/count_tiles(数量审计)、assets(含匹配代价,cost
    越小越可信,>0.5 标 uncertain)、source_size。
    """
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    sheet_path = run_dir / (payload.get("image") or "panels_green.png")
    src_path = run_dir / (payload.get("source") or "mid_fill.png")
    for p in (sheet_path, src_path):
        if not p.is_file():
            raise FileNotFoundError(f"input not found: {p}")

    panels = payload.get("panels")
    if not panels:
        import json as _json
        sj = run_dir / "structure1.json"
        if not sj.is_file():
            raise ValueError("payload.panels missing and structure1.json not found")
        structured = _json.loads(sj.read_text(encoding="utf-8"))
        panels = [item["bbox"] for item in structured.get("panel", [])
                  if item.get("yolo_detect") != "discard"]
    if not panels:
        raise ValueError("no panels to match")

    started = time.time()
    with Image.open(sheet_path) as im:
        sheet = im.convert("RGB")
    with Image.open(src_path) as im:
        src = im.convert("RGB")
    sw, sh = src.size

    # 切割侧超采样:把平铺图放大 N 倍后再做色键——软阈值 alpha 在更细的
    # 网格上计算,defringe 的有效收缩减为 1/N px,边缘量化更平滑;
    # 素材落盘不缩回,天然多带一层分辨率余量(回贴时等比缩小)
    ss = max(1, int(payload.get("supersample", 2)))
    if ss > 1:
        sheet = sheet.resize((sheet.width * ss, sheet.height * ss),
                             Image.Resampling.LANCZOS)

    tol = int(payload.get("tol", 60))
    defringe = int(payload.get("defringe", 1))

    # 槽位模式:第 16 步用"对号入座"模板生成时,槽位表已随图上传——
    # 直接按槽位矩形裁切,对应关系由构造保证,匹配环节整体消失
    # 分层模式:第 16 步逐层"原位保留+剥离"生成时,每层是与源图同布局的
    # 绿底图(panels_green_L<k>.png)——位置已知,按 bbox 直裁,无匹配环节
    layers = payload.get("layers")
    canvas_fit = payload.get("canvas")  # {size:[cw,ch], scale, offset:[ox,oy]}
    if layers:
        out_dir = run_dir / "panel_assets"
        out_dir.mkdir(exist_ok=True)
        for old_f in out_dir.glob("p*.png"):
            old_f.unlink()

        def _panel_rect(b):
            return ((b[0] - b[2] / 2) * sw, (b[1] - b[3] / 2) * sh,
                    (b[0] + b[2] / 2) * sw, (b[1] + b[3] / 2) * sh)

        panel_rects = [_panel_rect(b) for b in panels]
        assets = []
        rows = []
        tiles_by_panel = {}
        margin = 6 * ss
        for k, layer in enumerate(layers):
            lp = run_dir / f"panels_green_L{k}.png"
            if not lp.is_file():
                for pi in layer:
                    rows.append({"panel_index": pi, "file": "",
                                 "cost": 1.0, "uncertain": True,
                                 "low_res": False, "res_ratio": 0,
                                 "verify": "warn", "vdist": 1.0,
                                 "ar_diff": 0, "z": 0, "png_w": 0,
                                 "png_h": 0, "paste_x": 0, "paste_y": 0,
                                 "paste_w": 0, "paste_h": 0})
                continue
            with Image.open(lp) as im:
                lsheet = im.convert("RGB")
            # 画布空间裁切:层图归一化到"预设画布尺寸×ss"(与 GPT 输出同比例,
            # 只有分辨率缩放没有比例拉伸);panel 矩形经 scale/offset 变换定位
            if canvas_fit:
                cw_, ch_ = canvas_fit["size"]
                c_scale = float(canvas_fit["scale"])
                ox, oy = canvas_fit["offset"]
            else:
                cw_, ch_, c_scale, ox, oy = sw, sh, 1.0, 0, 0
            lsheet = lsheet.resize((int(cw_) * ss, int(ch_) * ss),
                                   Image.Resampling.LANCZOS)
            key = _border_median_color(lsheet)
            for pi in layer:
                if pi >= len(panels):
                    continue
                x0, y0, x1, y1 = panel_rects[pi]
                # 源图坐标 → 画布坐标
                x0c, y0c = x0 * c_scale + ox, y0 * c_scale + oy
                x1c, y1c = x1 * c_scale + ox, y1 * c_scale + oy
                cx0 = max(0, int(x0c * ss - margin))
                cy0 = max(0, int(y0c * ss - margin))
                cx1 = min(lsheet.width, int(math.ceil(x1c * ss + margin)))
                cy1 = min(lsheet.height, int(math.ceil(y1c * ss + margin)))
                crop = lsheet.crop((cx0, cy0, cx1, cy1))
                cut = _unkey_border(crop, tol, defringe, key=key, junk=False)
                tight = cut.getchannel("A").getbbox()
                if not tight:
                    rows.append({"panel_index": pi, "file": "",
                                 "cost": 1.0, "uncertain": True,
                                 "low_res": False, "res_ratio": 0,
                                 "verify": "warn", "vdist": 1.0,
                                 "ar_diff": 0, "z": 0, "png_w": 0,
                                 "png_h": 0, "paste_x": 0, "paste_y": 0,
                                 "paste_w": 0, "paste_h": 0})
                    continue
                tile = cut.crop(tight)
                fname = f"p{pi:02d}.png"
                tile.save(out_dir / fname)
                bw, bh = x1 - x0, y1 - y0
                src_crop = src.crop((max(0, int(x0)), max(0, int(y0)),
                                     min(sw, int(math.ceil(x1))),
                                     min(sh, int(math.ceil(y1)))))
                vdist = _vec_dist(_thumb_vec(tile, 32),
                                  _thumb_vec(src_crop, 32))
                ar_diff = abs(math.log(max(
                    1e-6, (tile.width / max(1, tile.height))
                    / max(1e-6, bw / max(1.0, bh)))))
                res_ratio = round(
                    min(tile.width / max(1.0, bw * c_scale * ss),
                        tile.height / max(1.0, bh * c_scale * ss)) * ss, 2)
                paste = (int(round(x0)), int(round(y0)),
                         max(1, int(round(bw))), max(1, int(round(bh))))
                tiles_by_panel[pi] = (tile, paste)
                rec = {
                    "panel_index": pi, "file": fname, "cost": 0.0,
                    "uncertain": vdist >= 0.25,
                    "low_res": res_ratio < 0.9, "res_ratio": res_ratio,
                    "verify": "ok" if vdist < 0.25 else "warn",
                    "vdist": round(vdist, 4), "ar_diff": round(ar_diff, 4),
                    "png_w": tile.width, "png_h": tile.height,
                    "bbox": panels[pi],
                    "paste_x": paste[0], "paste_y": paste[1],
                    "paste_w": paste[2], "paste_h": paste[3],
                }
                assets.append(rec)
                rows.append(rec)
        z_map = _panel_z_order(panel_rects, tiles_by_panel, src)
        for rec in assets:
            rec["z"] = z_map.get(rec["panel_index"], 0)
        fields = ["panel_index", "file", "cost", "uncertain", "verify",
                  "vdist", "ar_diff", "res_ratio", "low_res", "z",
                  "png_w", "png_h", "paste_x", "paste_y",
                  "paste_w", "paste_h"]
        manifest_path = out_dir / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return {
            "mask_mode_used": "layers",
            "sam2_error": None,
            "count_panels": len(panels),
            "count_tiles": len([r for r in rows if r["file"]]),
            "matched": len(assets),
            "count_ok": len(assets) == len(panels),
            "assets": sorted(assets, key=lambda a: a["panel_index"]),
            "manifest": str(manifest_path),
            "source_size": [sw, sh],
            "elapsed_sec": round(time.time() - started, 1),
        }

    # 槽位模式仅在显式传 payload.slots 时启用(实验通道;
    # 不自动读 run 目录残留的 panels_slots.json,防止劫持常规切割)
    slots_data = payload.get("slots")
    if slots_data and slots_data.get("slots"):
        tw_, th_ = (int(v) * ss for v in slots_data["size"])
        if sheet.size != (tw_, th_):
            sheet = sheet.resize((tw_, th_), Image.Resampling.LANCZOS)

        # 连通域切块(GPT 几何服从性弱,槽位只当"匹配锚点"不当裁剪边界):
        # 每块与每个槽位算 IoU,按 IoU 从大到小贪心配对;IoU 太低的槽位
        # 回落外观匹配(色彩网格)兜底
        rgba_sheet = _unkey_border(sheet, tol, defringe)
        comp_boxes = _connected_tiles(rgba_sheet)

        def _iou(a, b):
            ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
            ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
            if ix1 <= ix0 or iy1 <= iy0:
                return 0.0
            inter = (ix1 - ix0) * (iy1 - iy0)
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            return inter / max(1.0, area_a + area_b - inter)

        slot_rects = {}
        for slot in slots_data["slots"]:
            pi = int(slot["index"])
            if pi >= len(panels):
                continue
            sx, sy, sw2, sh2 = (int(v) * ss for v in slot["rect"])
            slot_rects[pi] = (sx, sy, sx + sw2, sy + sh2)

        # IoU 贪心配对
        cand = sorted(((_iou(comp_boxes[ci], slot_rects[pi]), ci, pi)
                       for ci in range(len(comp_boxes))
                       for pi in slot_rects),
                      reverse=True)
        comp_of = {}
        used_c = set()
        for v, ci, pi in cand:
            if v < 0.15 or ci in used_c or pi in comp_of:
                continue
            comp_of[pi] = (ci, v)
            used_c.add(ci)

        def _panel_rect(b):
            return ((b[0] - b[2] / 2) * sw, (b[1] - b[3] / 2) * sh,
                    (b[0] + b[2] / 2) * sw, (b[1] + b[3] / 2) * sh)

        panel_rects = [_panel_rect(b) for b in panels]

        # 外观兜底:没被 IoU 认领的槽位 × 没被认领的块
        rest_p = [pi for pi in slot_rects if pi not in comp_of]
        rest_c = [ci for ci in range(len(comp_boxes)) if ci not in used_c]
        if rest_p and rest_c:
            pv = {}
            for pi in rest_p:
                x0, y0, x1, y1 = panel_rects[pi]
                pv[pi] = _thumb_vec(src.crop(
                    (max(0, int(x0)), max(0, int(y0)),
                     min(sw, int(math.ceil(x1))), min(sh, int(math.ceil(y1))))))
            cv = {ci: _thumb_vec(rgba_sheet.crop(comp_boxes[ci]))
                  for ci in rest_c}
            flat2 = sorted(((_vec_dist(cv[ci], pv[pi]), ci, pi)
                            for ci in rest_c for pi in rest_p))
            for v, ci, pi in flat2:
                if ci in used_c or pi in comp_of:
                    continue
                comp_of[pi] = (ci, 0.0)
                used_c.add(ci)

        out_dir = run_dir / "panel_assets"
        out_dir.mkdir(exist_ok=True)
        for old_f in out_dir.glob("p*.png"):
            old_f.unlink()
        assets = []
        rows = []
        tiles_by_panel = {}
        for pi in sorted(slot_rects):
            if pi not in comp_of:
                rows.append({"panel_index": pi, "file": "", "cost": 1.0,
                             "uncertain": True, "low_res": False,
                             "res_ratio": 0, "verify": "warn", "vdist": 1.0,
                             "ar_diff": 0, "z": 0, "png_w": 0, "png_h": 0,
                             "paste_x": 0, "paste_y": 0,
                             "paste_w": 0, "paste_h": 0})
                continue
            ci, iou_v = comp_of[pi]
            tile = rgba_sheet.crop(comp_boxes[ci])
            fname = f"p{pi:02d}.png"
            tile.save(out_dir / fname)
            x0, y0, x1, y1 = panel_rects[pi]
            bw, bh = x1 - x0, y1 - y0
            src_crop = src.crop((max(0, int(x0)), max(0, int(y0)),
                                 min(sw, int(math.ceil(x1))),
                                 min(sh, int(math.ceil(y1)))))
            vdist = _vec_dist(_thumb_vec(tile, 32), _thumb_vec(src_crop, 32))
            ar_diff = abs(math.log(max(1e-6, (tile.width / max(1, tile.height))
                                       / max(1e-6, bw / max(1.0, bh)))))
            res_ratio = round(min(tile.width / max(1, bw),
                                  tile.height / max(1, bh)), 2)
            paste = (int(round(x0)), int(round(y0)),
                     max(1, int(round(bw))), max(1, int(round(bh))))
            tiles_by_panel[pi] = (tile, paste)
            rec = {
                "panel_index": pi, "file": fname,
                "cost": round(1 - iou_v, 4),
                "uncertain": vdist >= 0.25,
                "low_res": res_ratio < 0.9, "res_ratio": res_ratio,
                "verify": "ok" if vdist < 0.25 else "warn",
                "vdist": round(vdist, 4), "ar_diff": round(ar_diff, 4),
                "png_w": tile.width, "png_h": tile.height,
                "bbox": panels[pi],
                "paste_x": paste[0], "paste_y": paste[1],
                "paste_w": paste[2], "paste_h": paste[3],
            }
            assets.append(rec)
            rows.append(rec)
        z_map = _panel_z_order(panel_rects, tiles_by_panel, src)
        for rec in assets:
            rec["z"] = z_map.get(rec["panel_index"], 0)
        fields = ["panel_index", "file", "cost", "uncertain", "verify",
                  "vdist", "ar_diff", "res_ratio", "low_res", "z",
                  "png_w", "png_h", "paste_x", "paste_y", "paste_w", "paste_h"]
        manifest_path = out_dir / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return {
            "mask_mode_used": "slots",
            "sam2_error": None,
            "count_panels": len(panels),
            "count_tiles": len([r for r in rows if r["file"]]),
            "matched": len(assets),
            "count_ok": len(assets) == len(panels),
            "assets": sorted(assets, key=lambda a: a["panel_index"]),
            "manifest": str(manifest_path),
            "source_size": [sw, sh],
            "elapsed_sec": round(time.time() - started, 1),
        }

    rgba = _unkey_border(sheet, tol, defringe)
    tile_boxes = _connected_tiles(rgba)

    # 出边模式:默认 sam2——色键只负责找块和计数,轮廓交给 SAM2(按物体边缘
    # 而非颜色距离,GPT 边缘混色像素/淡影对它无感);失败自动回落色键结果
    mask_mode = payload.get("mask_mode", "chroma")
    sam_cfg = payload.get("sam2") or {}
    mask_mode_used = "chroma"
    sam2_error = None
    if mask_mode == "sam2" and tile_boxes:
        try:
            gw, gh = sheet.size
            cut_name = "_panel_sam_cut.png"
            sam2c.cutout({
                "dir": str(run_dir), "image": sheet_path.name,
                "output": cut_name,
                "borders": [{"bbox": [((b[0] + b[2]) / 2) / gw,
                                      ((b[1] + b[3]) / 2) / gh,
                                      (b[2] - b[0]) / gw,
                                      (b[3] - b[1]) / gh],
                             "positive_points": [], "negative_points": []}
                            for b in tile_boxes],
                "padding_ratio": float(sam_cfg.get("padding_ratio", 0.02)),
                "min_padding": float(sam_cfg.get("min_padding", 2)),
                "mask_threshold": float(sam_cfg.get("mask_threshold", 0.5)),
                "feather_radius": float(sam_cfg.get("feather_radius", 0)),
                "multimask": bool(sam_cfg.get("multimask", False)),
                "crop_scale": float(sam_cfg.get("crop_scale", 1.5)),
                "refine": bool(sam_cfg.get("refine", True)),
                "fill_holes": bool(sam_cfg.get("fill_holes", True)),
                "size_rules": [],
            })
            cut_path = run_dir / cut_name
            with Image.open(cut_path) as c:
                sam_rgba = c.convert("RGBA")
            cut_path.unlink(missing_ok=True)
            if sam_rgba.size != sheet.size:
                sam_rgba = sam_rgba.resize(sheet.size, Image.Resampling.LANCZOS)
            # SAM2 在角部置信度低,掩码会系统性削角:先外扩 2px 把角"长回来",
            # 再用色键软 alpha(不收缩版)做颜色护栏——只允许长回非绿像素,
            # 角恢复到真实颜色边界为止,绿像素一个进不来
            hard = sam_rgba.getchannel("A").point(
                lambda v: 255 if v >= 128 else 0)
            grow_px = int(sam_cfg.get("grow", 2))
            grown = (hard.filter(ImageFilter.MaxFilter(2 * grow_px + 1))
                     if grow_px > 0 else hard)
            guard = _unkey_border(sheet, tol, 0)
            guard_alpha = guard.getchannel("A")
            # 护栏外扩:色键会把边缘抗锯齿混合带切掉(四边削像素的真正来源),
            # 允许最终边界越过颜色边界 guard_grow px,混合带以半透明回归,
            # despill 负责压掉其中的绿色成分
            gg = int(sam_cfg.get("guard_grow", 1))
            if gg > 0:
                guard_alpha = guard_alpha.filter(
                    ImageFilter.MaxFilter(2 * gg + 1))
            final_alpha = ImageChops.darker(grown, guard_alpha)
            key = _border_median_color(sheet)
            out = _despill_edges(sheet, final_alpha, key).convert("RGBA")
            out.putalpha(final_alpha)
            rgba = out
            mask_mode_used = "sam2"
        except Exception as e:  # 回落色键 rgba,但把原因透出去
            sam2_error = f"{type(e).__name__}: {e}"[-300:]

    tiles = [rgba.crop(b) for b in tile_boxes]

    # 双特征代价矩阵:长宽比 + 色彩网格
    def _panel_rect(b):
        return ((b[0] - b[2] / 2) * sw, (b[1] - b[3] / 2) * sh,
                (b[0] + b[2] / 2) * sw, (b[1] + b[3] / 2) * sh)

    panel_rects = [_panel_rect(b) for b in panels]
    panel_vecs = []
    panel_ars = []
    for (x0, y0, x1, y1) in panel_rects:
        crop = src.crop((max(0, int(x0)), max(0, int(y0)),
                         min(sw, int(math.ceil(x1))), min(sh, int(math.ceil(y1)))))
        panel_vecs.append(_thumb_vec(crop))
        panel_ars.append((x1 - x0) / max(1e-6, (y1 - y0)))
    tile_vecs = [_thumb_vec(t) for t in tiles]
    tile_ars = [t.width / max(1, t.height) for t in tiles]

    cost = [[abs(math.log(max(1e-6, tile_ars[ti] / panel_ars[pi])))
             + 2.0 * _vec_dist(tile_vecs[ti], panel_vecs[pi])
             for pi in range(len(panels))] for ti in range(len(tiles))]

    # 贪心全局最小分配
    pairs = []
    used_t, used_p = set(), set()
    flat = sorted(((cost[ti][pi], ti, pi)
                   for ti in range(len(tiles)) for pi in range(len(panels))))
    for c, ti, pi in flat:
        if ti in used_t or pi in used_p:
            continue
        pairs.append((ti, pi, c))
        used_t.add(ti)
        used_p.add(pi)

    # 2-opt 交换优化:修正贪心的次优配对,直至总代价不再下降
    improved = True
    while improved:
        improved = False
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                ta, pa, ca = pairs[a]
                tb, pb, cb = pairs[b]
                if cost[ta][pb] + cost[tb][pa] + 1e-9 < ca + cb:
                    pairs[a] = (ta, pb, cost[ta][pb])
                    pairs[b] = (tb, pa, cost[tb][pa])
                    improved = True

    out_dir = run_dir / "panel_assets"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("p*.png"):
        old.unlink()
    assets = []
    rows = []
    tiles_by_panel = {}
    for ti, pi, c in pairs:
        tile = tiles[ti]
        fname = f"p{pi:02d}.png"
        tile.save(out_dir / fname)
        x0, y0, x1, y1 = panel_rects[pi]
        bw, bh = x1 - x0, y1 - y0
        # 拉伸填满 bbox(几何权威):GPT 比例略偏时在拼回环节强行矫正;
        # 比例偏差大小由 ar_diff 复核记录,超阈标 warn 提示生成没守约
        paste = (int(round(x0)), int(round(y0)),
                 max(1, int(round(bw))), max(1, int(round(bh))))
        # 回贴前加强复核:32×32 高分辨率比色 + 长宽比偏差,双阈值确认位置与大小
        src_crop = src.crop((max(0, int(x0)), max(0, int(y0)),
                             min(sw, int(math.ceil(x1))),
                             min(sh, int(math.ceil(y1)))))
        vdist = _vec_dist(_thumb_vec(tile, 32), _thumb_vec(src_crop, 32))
        ar_diff = abs(math.log(max(1e-6,
                                   (tile.width / max(1, tile.height))
                                   / panel_ars[pi])))
        verify = "ok" if (vdist < 0.18 and ar_diff < 0.25) else "warn"
        # 分辨率比:素材像素 ÷ 回贴目标像素,<0.9 说明画小了、拉回会糊
        res_ratio = round(min(tile.width / max(1, bw),
                              tile.height / max(1, bh)), 2)
        tiles_by_panel[pi] = (tile, paste)
        rec = {
            "panel_index": pi, "file": fname, "cost": round(c, 4),
            "uncertain": c > 0.5 or verify == "warn",
            "low_res": res_ratio < 0.9,
            "res_ratio": res_ratio,
            "verify": verify, "vdist": round(vdist, 4),
            "ar_diff": round(ar_diff, 4),
            "png_w": tile.width, "png_h": tile.height,
            "bbox": panels[pi],
            "paste_x": paste[0], "paste_y": paste[1],
            "paste_w": paste[2], "paste_h": paste[3],
        }
        assets.append(rec)
        rows.append(rec)

    # 前后叠放次序(z 越大越靠上):包含→内者在上;相交→重叠区取色裁决
    z_map = _panel_z_order(panel_rects, tiles_by_panel, src)
    for rec in assets:
        rec["z"] = z_map.get(rec["panel_index"], 0)

    fields = ["panel_index", "file", "cost", "uncertain", "verify", "vdist",
              "ar_diff", "res_ratio", "low_res", "z", "png_w", "png_h",
              "paste_x", "paste_y", "paste_w", "paste_h"]
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "mask_mode_used": mask_mode_used,
        "sam2_error": sam2_error,
        "count_panels": len(panels), "count_tiles": len(tiles),
        "matched": len(pairs),
        "count_ok": len(tiles) == len(panels),
        "assets": sorted(assets, key=lambda a: a["panel_index"]),
        "manifest": str(manifest_path),
        "source_size": [sw, sh],
        "elapsed_sec": round(time.time() - started, 1),
    }


def _z_order_extract(rects):
    """提取场景的 z 序(合成图上无法取色裁决):
    包含 → 内者在上;部分相交 → 面积小者在上(UI 常识);Kahn 拓扑,环按面积降序兜底。"""
    n = len(rects)

    def _inter(a, b):
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        return x1 - x0 > 2 and y1 - y0 > 2

    def _contains(a, b):
        return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]

    def _area(r):
        return (r[2] - r[0]) * (r[3] - r[1])

    from collections import defaultdict, deque
    adj = defaultdict(set)
    indeg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not _inter(rects[i], rects[j]):
                continue
            if _contains(rects[i], rects[j]):
                lo, hi = i, j
            elif _contains(rects[j], rects[i]):
                lo, hi = j, i
            else:
                lo, hi = ((i, j) if _area(rects[i]) >= _area(rects[j])
                          else (j, i))
            if hi not in adj[lo]:
                adj[lo].add(hi)
                indeg[hi] += 1
    q = deque(sorted((i for i in range(n) if indeg[i] == 0),
                     key=lambda i: -_area(rects[i])))
    z = {}
    order = 0
    while q:
        u = q.popleft()
        z[u] = order
        order += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    for u in sorted((i for i in range(n) if i not in z),
                    key=lambda i: -_area(rects[i])):
        z[u] = order
        order += 1
    return z


def _geodesic_connected(candidate: Image.Image, seed: Image.Image,
                        iters: int = 200) -> Image.Image:
    """保留 candidate 中与 seed 相连的部分(测地膨胀,512 缩略网格上做)。

    用途:候选隐藏区只认"与可见本体相邻"的遮挡,隔空的杂物剔除。"""
    w, h = candidate.size
    scale = min(1.0, 512 / max(w, h))
    sw_, sh_ = max(1, round(w * scale)), max(1, round(h * scale))
    cand = candidate.resize((sw_, sh_), Image.Resampling.NEAREST) \
        .point(lambda v: 255 if v > 0 else 0)
    marker = seed.resize((sw_, sh_), Image.Resampling.NEAREST) \
        .point(lambda v: 255 if v > 0 else 0)
    allowed = ImageChops.lighter(cand, marker)
    prev = None
    for _ in range(iters):
        marker = ImageChops.darker(
            marker.filter(ImageFilter.MaxFilter(3)), allowed)
        cur = marker.tobytes()
        if cur == prev:
            break
        prev = cur
    keep = ImageChops.darker(marker, cand)
    return keep.resize((w, h), Image.Resampling.NEAREST)


def handle_panel_extract(payload: dict) -> dict:
    """第 16/17 替代:确定性 panel 提取(零生成式排版,几何零漂移)。

    逐 panel:SAM2 在 mid_fill 上按 bbox 抠可见部分 → z 序(包含→内者在上,
    相交→面积小者在上)→ 被上层 panel 压住的区域(上层 mask ∩ 本框)用
    flux_fill 原位补全(panel_fill LoRA)→ 输出逐 panel RGBA + manifest。

    payload:
        run_id / dir   二选一
        panels         必填 [[cx,cy,w,h],...](归一化)
        z_order        可选,与 panels 对齐的层级数组(越大越上层,如第 16 步
                       审核结果);缺席时按几何规则内算
        image          源图,默认 mid_fill.png
        lora           默认 panel_fill.safetensors(传空串禁用)
        seed/steps/guidance      fill 参数,默认 9/20/30
        grow/blur      洞外扩/羽化 px,默认 4/2
        margin_ratio   fill 裁块上下文外扩,默认 0.25
    输出:panel_extract/p<idx>.png + manifest.csv(paste 矩形、z、是否补过洞)
    """
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    panels = payload.get("panels")
    if not panels:
        raise ValueError("payload.panels must be a non-empty array")
    src_path = run_dir / (payload.get("image") or "mid_fill.png")
    if not src_path.is_file():
        raise FileNotFoundError(f"input not found: {src_path}")
    with Image.open(src_path) as im:
        src = im.convert("RGB")
    sw, sh = src.size

    lora = payload.get("lora", "panel_fill.safetensors") or None
    seed = int(payload.get("seed", 9))
    steps = int(payload.get("steps", 20))
    guidance = float(payload.get("guidance", 30.0))
    grow = int(payload.get("grow", 4))
    blur = float(payload.get("blur", 2))
    margin_ratio = float(payload.get("margin_ratio", 0.25))

    def _rect(b):
        return (max(0, int((b[0] - b[2] / 2) * sw)),
                max(0, int((b[1] - b[3] / 2) * sh)),
                min(sw, int(math.ceil((b[0] + b[2] / 2) * sw))),
                min(sh, int(math.ceil((b[1] + b[3] / 2) * sh))))

    rects = [_rect(b) for b in panels]
    started = time.time()

    # ① 逐 panel SAM2 抠可见部分(全画布 alpha)
    alphas = []
    for i, b in enumerate(panels):
        tmp = f"_pe_cut_{i:02d}.png"
        try:
            sam2c.cutout({
                "dir": str(run_dir), "image": src_path.name, "output": tmp,
                "borders": [{"bbox": b, "positive_points": [],
                             "negative_points": []}],
                "padding_ratio": 0.02, "min_padding": 2,
                "mask_threshold": 0.5, "feather_radius": 0,
                "multimask": False, "crop_scale": 1.5, "refine": True,
                "fill_holes": True, "size_rules": [],
            })
            tmp_path = run_dir / tmp
            with Image.open(tmp_path) as c:
                a = c.convert("RGBA").getchannel("A").point(
                    lambda v: 255 if v > 0 else 0)
            tmp_path.unlink(missing_ok=True)
        except RuntimeError:
            a = Image.new("L", (sw, sh), 0)
        if a.size != (sw, sh):
            a = a.resize((sw, sh), Image.Resampling.NEAREST)
        alphas.append(a)

    # ② z 序与上层遮挡:payload.z_order(如第 16 步 VL 审核的层级)优先,
    # 缺席时按几何规则内算(包含→内者在上,相交→面积小者在上)
    z_payload = payload.get("z_order")
    if z_payload and len(z_payload) == len(rects):
        z_map = [int(v) for v in z_payload]
    else:
        z_map = _z_order_extract(rects)
    out_dir = run_dir / "panel_extract"
    out_dir.mkdir(exist_ok=True)
    for old_f in out_dir.glob("p*.png"):
        old_f.unlink()

    assets = []
    rows = []
    for i, b in enumerate(panels):
        x0, y0, x1, y1 = rects[i]
        # 工作区放宽:真实边界可能藏在遮挡物下、超出 YOLO 框——
        # 候选隐藏区 = 上层 mask ∩ "bbox 外扩 40% 长边"的矩形(隐藏部分只会
        # 紧邻可见部分,放宽有界)
        d = max(24, round(max(x1 - x0, y1 - y0) * 0.4))
        ex0, ey0 = max(0, x0 - d), max(0, y0 - d)
        ex1, ey1 = min(sw, x1 + d), min(sh, y1 + d)
        exp_mask = Image.new("L", (sw, sh), 0)
        ImageDraw.Draw(exp_mask).rectangle([ex0, ey0, ex1 - 1, ey1 - 1],
                                           fill=255)
        upper = Image.new("L", (sw, sh), 0)
        for j in range(len(panels)):
            if z_map[j] > z_map[i]:
                upper = ImageChops.lighter(upper, alphas[j])
        hole = ImageChops.multiply(upper, exp_mask)
        # 洞不含可见本体(可见处不重绘)
        hole = ImageChops.subtract(hole, alphas[i])
        # 连通性约束:只认与可见本体相连的遮挡区,隔空杂物剔除
        if hole.getbbox():
            hole = _geodesic_connected(hole, alphas[i])
        filled = bool(hole.getbbox())
        vis_area = sum(1 for v in alphas[i].getdata() if v)
        hole_area = sum(1 for v in hole.getdata() if v) if filled else 0
        hidden_ratio = round(hole_area / max(1, vis_area), 3)

        # 裁块窗口覆盖 可见∪候选隐藏区
        ub = ImageChops.lighter(alphas[i], hole).getbbox() or (x0, y0, x1, y1)
        m = max(16, round(max(ub[2] - ub[0], ub[3] - ub[1]) * margin_ratio))
        cx0, cy0 = max(0, ub[0] - m), max(0, ub[1] - m)
        cx1, cy1 = min(sw, ub[2] + m), min(sh, ub[3] + m)
        crop = src.crop((cx0, cy0, cx1, cy1))
        if filled:
            # ③ 原位 inpaint 候选隐藏区(panel_fill 训练目标=完整下层 panel,
            # 模型会画出被压住的延续和真实收边)
            hole_crop = hole.crop((cx0, cy0, cx1, cy1))
            if grow > 0:
                hole_crop = hole_crop.filter(
                    ImageFilter.MaxFilter(2 * grow + 1))
            soft = hole_crop.filter(ImageFilter.GaussianBlur(blur)) \
                if blur > 0 else hole_crop
            img_name = comfy.place_input_pil(crop, prefix=f"pe_{i:02d}_")
            mask_name = comfy.place_input_pil(
                soft.convert("RGB"), prefix=f"pe_mask_{i:02d}_")
            entry = comfy.run_workflow(build_flux_fill_workflow(
                image_name=img_name, mask_name=mask_name,
                prompt=DEFAULT_RESIDUAL_FILL_PROMPT,
                seed=seed, steps=steps,
                width=crop.width, height=crop.height,
                guidance=guidance, lora_name=lora))
            images = comfy.output_image_paths(entry)
            if images:
                with Image.open(images[0]) as g:
                    gen = g.convert("RGB")
                    if gen.size != crop.size:
                        gen = gen.resize(crop.size, Image.Resampling.LANCZOS)
                gen, _ = _match_colors_to_input(gen, crop, soft.convert("RGB"))
                crop = Image.composite(gen, crop, soft)

        # ④ 定界:补全后的裁块上重新 SAM2 测真实边界(补完再定界,
        # 不再信被遮挡的视觉边界/YOLO 框);未补洞的直接用可见 mask
        region = ImageChops.lighter(alphas[i], hole).crop((cx0, cy0, cx1, cy1))
        remeasured = False
        if filled:
            tmp_img = f"_pe_fill_{i:02d}.png"
            tmp_cut = f"_pe_recut_{i:02d}.png"
            crop.save(run_dir / tmp_img)
            rb = region.getbbox()
            if rb:
                nb = [((rb[0] + rb[2]) / 2) / crop.width,
                      ((rb[1] + rb[3]) / 2) / crop.height,
                      (rb[2] - rb[0]) / crop.width,
                      (rb[3] - rb[1]) / crop.height]
                try:
                    sam2c.cutout({
                        "dir": str(run_dir), "image": tmp_img,
                        "output": tmp_cut,
                        "borders": [{"bbox": nb, "positive_points": [],
                                     "negative_points": []}],
                        "padding_ratio": 0.02, "min_padding": 2,
                        "mask_threshold": 0.5, "feather_radius": 0,
                        "multimask": False, "crop_scale": 1.5,
                        "refine": True, "fill_holes": True,
                        "size_rules": [],
                    })
                    cut_path = run_dir / tmp_cut
                    if cut_path.is_file():
                        with Image.open(cut_path) as c:
                            a2 = c.convert("RGBA").getchannel("A").point(
                                lambda v: 255 if v > 0 else 0)
                        if a2.size == crop.size and a2.getbbox():
                            # 守门:复测 mask 必须罩住绝大部分可见本体,
                            # 否则说明抓错目标,弃用复测结果
                            vis_crop = alphas[i].crop((cx0, cy0, cx1, cy1))
                            inter = sum(1 for v in ImageChops.darker(
                                a2, vis_crop).getdata() if v)
                            vis_ct = sum(1 for v in vis_crop.getdata() if v)
                            if vis_ct and inter / vis_ct >= 0.7:
                                region = a2
                                remeasured = True
                except RuntimeError:
                    pass
            (run_dir / tmp_img).unlink(missing_ok=True)
            (run_dir / tmp_cut).unlink(missing_ok=True)

        tile = crop.convert("RGBA")
        tile.putalpha(region)
        tight = region.getbbox()
        if not tight:
            rows.append({"panel_index": i, "file": "", "z": z_map[i],
                         "filled": filled, "remeasured": remeasured,
                         "hidden_ratio": hidden_ratio,
                         "needs_review": True, "png_w": 0, "png_h": 0,
                         "paste_x": 0, "paste_y": 0,
                         "paste_w": 0, "paste_h": 0})
            continue
        tile = tile.crop(tight)
        fname = f"p{i:02d}.png"
        tile.save(out_dir / fname)
        rec = {
            "panel_index": i, "file": fname, "z": z_map[i],
            "filled": filled, "remeasured": remeasured,
            "hidden_ratio": hidden_ratio,
            "needs_review": hidden_ratio > 0.6,
            "png_w": tile.width, "png_h": tile.height,
            "bbox": panels[i],
            "paste_x": cx0 + tight[0], "paste_y": cy0 + tight[1],
            "paste_w": tight[2] - tight[0], "paste_h": tight[3] - tight[1],
        }
        assets.append(rec)
        rows.append(rec)

    fields = ["panel_index", "file", "z", "filled", "remeasured",
              "hidden_ratio", "needs_review", "png_w", "png_h",
              "paste_x", "paste_y", "paste_w", "paste_h"]
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "count": len(panels),
        "ok": len(assets),
        "filled_count": sum(1 for r in assets if r["filled"]),
        "assets": sorted(assets, key=lambda a: a["panel_index"]),
        "manifest": str(manifest_path),
        "source_size": [sw, sh],
        "elapsed_sec": round(time.time() - started, 1),
    }


def handle_panel_peel(payload: dict) -> dict:
    """第 17 步"分层提取":拆-补-拆-补 剥洋葱,按 z 层整层出图。

    以第 16 步审核的层级为准,从最上层开始:
      拆 z_k :SAM2 按该层全部 panel 框在**当前工作图**上抠出
              → panel_layers/z<k>.png(整层一张 RGBA);
      补下层 :工作图上把 z_k 的 alpha 挖洞,flux_fill(panel_fill LoRA)
              原位补全 → 新工作图(下层被压住的部分被补完整),继续拆下一层。
    最底层拆完不再补。每个 panel 的细拆后续另做。

    payload:
        run_id / dir   二选一
        panels         必填 [[cx,cy,w,h],...](归一化)
        z_order        与 panels 对齐的层级数组(越大越上层);缺席时几何内算
        image          源图,默认 mid_fill.png
        lora           默认 panel_fill.safetensors(空串禁用)
        seed/steps/guidance   fill 参数,默认 9/20/30
        grow/blur      洞外扩/羽化 px,默认 4/2
        prompt         补洞提示词,默认 DEFAULT_RESIDUAL_FILL_PROMPT
        padding_ratio/min_padding/mask_threshold/feather_radius/
        crop_scale/refine/multimask/fill_holes   拆(SAM2)参数,默认同第 17 步旧版
    输出:panel_layers/z<k>.png 逐层 RGBA、stage_after_z<k>.png 补后工作图、
         manifest.json(levels: [{z, file, count}])
    """
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    panels = payload.get("panels")
    if not panels:
        raise ValueError("payload.panels must be a non-empty array")
    src_path = run_dir / (payload.get("image") or "mid_fill.png")
    if not src_path.is_file():
        raise FileNotFoundError(f"input not found: {src_path}")
    lora = payload.get("lora", "panel_fill.safetensors") or None
    seed = int(payload.get("seed", 9))
    steps = int(payload.get("steps", 20))
    guidance = float(payload.get("guidance", 30.0))
    grow = int(payload.get("grow", 4))
    blur = float(payload.get("blur", 2))
    prompt = payload.get("prompt") or DEFAULT_RESIDUAL_FILL_PROMPT
    # 拆(SAM2)参数,全部可配
    sam_params = {
        "padding_ratio": float(payload.get("padding_ratio", 0.02)),
        "min_padding": float(payload.get("min_padding", 2)),
        "mask_threshold": float(payload.get("mask_threshold", 0.5)),
        "feather_radius": float(payload.get("feather_radius", 0)),
        "multimask": bool(payload.get("multimask", False)),
        "crop_scale": float(payload.get("crop_scale", 1.5)),
        "refine": bool(payload.get("refine", True)),
        "fill_holes": bool(payload.get("fill_holes", True)),
    }

    with Image.open(src_path) as im:
        work = im.convert("RGB")
    sw, sh = work.size

    def _rect(b):
        return (max(0, int((b[0] - b[2] / 2) * sw)),
                max(0, int((b[1] - b[3] / 2) * sh)),
                min(sw, int(math.ceil((b[0] + b[2] / 2) * sw))),
                min(sh, int(math.ceil((b[1] + b[3] / 2) * sh))))

    z_payload = payload.get("z_order")
    if z_payload and len(z_payload) == len(panels):
        z_map = [int(v) for v in z_payload]
    else:
        z_map = _z_order_extract([_rect(b) for b in panels])

    out_dir = run_dir / "panel_layers"
    out_dir.mkdir(exist_ok=True)
    for old_f in out_dir.glob("*.png"):
        old_f.unlink()

    started = time.time()
    levels = sorted(set(z_map), reverse=True)  # 顶层在前
    results = []
    for pos, level in enumerate(levels):
        idxs = [i for i, z in enumerate(z_map) if z == level]
        # 拆:当前工作图落盘,整层 SAM2(该层全部框合并输出一张 RGBA)
        work_name = f"_peel_work_{level}.png"
        work.save(run_dir / work_name)
        layer_rel = f"panel_layers/z{level}.png"
        sam2c.cutout({
            "dir": str(run_dir), "image": work_name, "output": layer_rel,
            "borders": [{"bbox": panels[i], "positive_points": [],
                         "negative_points": []} for i in idxs],
            **sam_params, "size_rules": [],
        })
        (run_dir / work_name).unlink(missing_ok=True)
        with Image.open(run_dir / layer_rel) as im:
            alpha = im.convert("RGBA").getchannel("A").point(
                lambda v: 255 if v > 0 else 0)
        if alpha.size != (sw, sh):
            alpha = alpha.resize((sw, sh), Image.Resampling.NEAREST)
        results.append({"z": level, "file": layer_rel, "count": len(idxs)})

        # 补:非最底层时,把本层挖掉的洞 fill 补全,得到下一轮工作图
        if pos != len(levels) - 1 and alpha.getbbox():
            hole = alpha
            if grow > 0:
                hole = hole.filter(ImageFilter.MaxFilter(2 * grow + 1))
            soft = hole.filter(ImageFilter.GaussianBlur(blur)) \
                if blur > 0 else hole
            img_name = comfy.place_input_pil(work, prefix=f"peel_{level}_")
            mask_name = comfy.place_input_pil(
                soft.convert("RGB"), prefix=f"peel_mask_{level}_")
            entry = comfy.run_workflow(build_flux_fill_workflow(
                image_name=img_name, mask_name=mask_name,
                prompt=prompt,
                seed=seed, steps=steps,
                width=sw, height=sh,
                guidance=guidance, lora_name=lora))
            images = comfy.output_image_paths(entry)
            if not images:
                raise RuntimeError(f"flux_fill 无输出(补 z{level} 下层时)")
            with Image.open(images[0]) as g:
                gen = g.convert("RGB")
                if gen.size != (sw, sh):
                    gen = gen.resize((sw, sh), Image.Resampling.LANCZOS)
            gen, _ = _match_colors_to_input(gen, work, soft.convert("RGB"))
            work = Image.composite(gen, work, soft)
            work.save(out_dir / f"stage_after_z{level}.png")

    import json as _json
    manifest = {"levels": results, "count": len(panels),
                "elapsed_sec": round(time.time() - started, 1)}
    (out_dir / "manifest.json").write_text(
        _json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def handle_qwen_layered(payload: dict) -> dict:
    """第 16 步(新):Qwen-Image-Layered 一步分层(代理 127.0.0.1:8195 daemon)。

    payload:
        run_id / dir   二选一
        image          输入图,默认 mid_fill.png
        output_dir     输出目录名,默认 panel_layers_qwen
        layers         层数(含 bg),默认 6(= 训练口径 bg+5)
        steps/seed/true_cfg   采样参数,默认 40/7/4.0
    返回 daemon 的 manifest:{files:[...], elapsed_sec}
    仅双卡布局可用(daemon 常驻 GPU1);单卡日会连接失败。
    """
    if payload.get("dir"):
        run_dir = Path(payload["dir"])
    else:
        run_dir = storage.get_run_dir(payload["run_id"])
    import urllib.request as _ur
    import urllib.error as _ue
    import json as _json
    body = _json.dumps({
        "dir": str(run_dir),
        "image": payload.get("image") or "mid_fill.png",
        "output_dir": payload.get("output_dir") or "panel_layers_qwen",
        "layers": int(payload.get("layers", 6)),
        "steps": int(payload.get("steps", 40)),
        "seed": int(payload.get("seed", 7)),
        "true_cfg": float(payload.get("true_cfg", 4.0)),
    }).encode()
    req = _ur.Request("http://127.0.0.1:8195/decompose", data=body,
                      headers={"Content-Type": "application/json"})
    try:
        with _ur.urlopen(req, timeout=900) as resp:
            return _json.loads(resp.read())
    except _ue.HTTPError as e:
        detail = ""
        try:
            detail = _json.loads(e.read().decode()).get("error", "")
        except Exception:
            pass
        raise RuntimeError(f"qwen layered daemon HTTP {e.code}: {detail[-2000:]}")
    except _ue.URLError as e:
        raise RuntimeError(
            f"qwen layered daemon 不可达({e.reason});该功能需要双卡布局") from None


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
    worker.register("panel_asset", handle_panel_asset)
    worker.register("panel_extract", handle_panel_extract)
    worker.register("panel_peel", handle_panel_peel)
    worker.register("qwen_layered", handle_qwen_layered)
