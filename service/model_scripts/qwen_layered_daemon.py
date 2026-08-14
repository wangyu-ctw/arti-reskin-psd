"""Qwen-Image-Layered panel 分层常驻 daemon(GPU_PLAN 布局 B,钉 GPU1)。

加载 LoRA 已合并的 transformer(载入时 torchao fp8 权重量化,~20G 常驻)
+ RGBA VAE + 预计算的正/负提示词嵌入(不载 7B 文本编码器)。

监听 127.0.0.1:8195:
    GET  /health      就绪探针(模型加载完成后 ok)
    POST /decompose   {"dir": run目录, "image": "mid_fill.png",
                       "output_dir": "panel_layers_qwen",
                       "layers": 6, "steps": 40, "seed": 7, "true_cfg": 4.0}
    返回 {"files": ["panel_layers_qwen/bg.png", ".../z0.png", ...],
          "elapsed_sec": ...}
    输出帧序 = 训练口径:bg, panel z0..z{layers-2}(combined 帧管线内部丢弃)
"""
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

HOST = "127.0.0.1"
PORT = int(os.environ.get("QWENLD_PORT", "8195"))
BASE = os.environ.get("QWEN_LAYERED_BASE",
                      "/workspace/hf_models/Qwen-Image-Layered")
MERGED = os.environ.get("QWEN_LAYERED_MERGED",
                        "/workspace/hf_models/Qwen-Layered-panel-merged")

pipe = None
prompt_cache = None
run_lock = threading.Lock()


def load_model() -> None:
    global pipe, prompt_cache
    from diffusers import (QwenImageLayeredPipeline,
                           QwenImageTransformer2DModel, TorchAoConfig)
    from torchao.quantization import Float8WeightOnlyConfig
    print("loading merged transformer (fp8 weight-only quantize on load)...",
          flush=True)
    tf = QwenImageTransformer2DModel.from_pretrained(
        MERGED, subfolder="transformer", torch_dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(Float8WeightOnlyConfig()),
        device_map="cuda")
    pipe = QwenImageLayeredPipeline.from_pretrained(
        BASE, transformer=tf, text_encoder=None, tokenizer=None,
        processor=None, torch_dtype=torch.bfloat16)
    # 管线内部会取 text_encoder.dtype(条件图 dtype 对齐);不载 7B 编码器,
    # 塞一个只带 dtype 的替身。绕过 DiffusionPipeline.__setattr__ 的注册逻辑
    class _Dummy:
        dtype = torch.bfloat16
    object.__setattr__(pipe, "text_encoder", _Dummy())
    pipe.vae.to("cuda")
    # 管线尾部把 (layers+1) 帧 reshape 成 batch 一次性 vae.decode,显存峰值
    # 会顶爆 GPU1(还驻着 SAM2/YOLO)。包一层逐帧分块解码
    from diffusers.models.autoencoders.vae import DecoderOutput
    orig_decode = pipe.vae.decode
    def chunked_decode(z, return_dict=True, **kw):
        outs = []
        for i in range(z.shape[0]):
            outs.append(orig_decode(z[i:i + 1], return_dict=False, **kw)[0])
            torch.cuda.empty_cache()
        out = torch.cat(outs, dim=0)
        return DecoderOutput(sample=out) if return_dict else (out,)
    pipe.vae.decode = chunked_decode
    prompt_cache = torch.load(Path(MERGED) / "prompt_cache.pt",
                              map_location="cuda", weights_only=False)
    print("qwen layered ready", flush=True)


def decompose(req: dict) -> dict:
    from PIL import Image
    run_dir = Path(req["dir"])
    image_path = run_dir / req.get("image", "mid_fill.png")
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    out_rel = req.get("output_dir", "panel_layers_qwen")
    layers = int(req.get("layers", 6))
    steps = int(req.get("steps", 40))
    seed = int(req.get("seed", 7))
    true_cfg = float(req.get("true_cfg", 4.0))

    started = time.time()
    img = Image.open(image_path).convert("RGBA")
    gen = torch.Generator("cuda").manual_seed(seed)
    pe = prompt_cache
    dev_pe = pe["prompt_embeds"].to("cuda")
    dev_pm = pe["prompt_mask"].to("cuda") if pe["prompt_mask"] is not None else None
    dev_ne = pe["neg_embeds"].to("cuda")
    dev_nm = pe["neg_mask"].to("cuda") if pe["neg_mask"] is not None else None
    # prompt 文本必须非空(空串会触发管线的自动看图配文,需要文本编码器);
    # 实际语义来自预计算 embeds,文本只是占位
    # 正向传文本仅为绕过空 prompt 的自动配文;负向只许传 embeds(管线校验)
    result = pipe(image=img,
                  prompt=pe["prompt"],
                  prompt_embeds=dev_pe, prompt_embeds_mask=dev_pm,
                  negative_prompt_embeds=dev_ne,
                  negative_prompt_embeds_mask=dev_nm,
                  true_cfg_scale=true_cfg, layers=layers,
                  num_inference_steps=steps, generator=gen,
                  resolution=640)
    out_dir = run_dir / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    files = []
    names = ["bg"] + [f"z{i}" for i in range(layers - 1)]
    for name, layer in zip(names, result.images[0]):
        f = f"{out_rel}/{name}.png"
        layer.save(run_dir / f)
        files.append(f)
    manifest = {"files": files, "layers": layers, "steps": steps,
                "seed": seed, "true_cfg": true_cfg,
                "elapsed_sec": round(time.time() - started, 1)}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": pipe is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/decompose":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length))
            with run_lock:
                result = decompose(req)
            self._send(200, result)
        except Exception:
            self._send(500, {"error": traceback.format_exc()[-3000:]})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    load_model()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"qwen layered daemon listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
