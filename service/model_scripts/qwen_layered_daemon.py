"""Qwen-Image-Layered 分层常驻 daemon(v2:基座 + 双 adapter 热切换)。

加载基座 transformer(默认 bf16;QWEN_LAYERED_QUANT=fp8 时 torchao 载入量化,
实验性)+ 两个 PEFT adapter(six_slot 六槽整图 / panelz panel z 分层)
+ 各自的预计算正/负提示词嵌入(不载 7B 文本编码器)。

显存:bf16 基座 ~39G + adapter 各 ~0.4G,需要 ≥60G 单卡(PRO 6000 级);
32G 卡请设 QWEN_LAYERED_QUANT=fp8(PEFT over torchao,未充分验证)。

监听 127.0.0.1:8195:
    GET  /health      {ok, modes}
    POST /decompose   {"dir": run目录, "image": "xxx.png",
                       "mode": "six_slot" | "panelz"(默认 panelz,兼容旧前端),
                       "output_dir": ..., "layers": N, "names": [...](可选),
                       "steps": 40, "seed": 7, "true_cfg": 4.0,
                       "resolution": 640|1024}
    返回 {"files": [...], "elapsed_sec": ...}
    帧序 = 训练口径:names[0](bg)在前,依序向上。
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
QUANT = os.environ.get("QWEN_LAYERED_QUANT", "bf16")
COMPILE = os.environ.get("QWEN_LAYERED_COMPILE", "1") != "0"
ADAPTERS = {
    "six_slot": os.environ.get(
        "QWEN_ADAPTER_SIX", "/workspace/outputs/six_slot_v3_lora/checkpoint-3000"),
    "panelz": os.environ.get(
        "QWEN_ADAPTER_PANELZ", "/workspace/outputs/panelz_v2_lora/checkpoint-3000"),
}
PROMPT_CACHES = {
    "six_slot": os.environ.get(
        "QWEN_PC_SIX",
        "/workspace/inputs/six_slot_cache_20260814/prompt_cache_infer.pt"),
    "panelz": os.environ.get(
        "QWEN_PC_PANELZ",
        "/workspace/inputs/layered_panel_v3_cache_20260818/prompt_cache_infer.pt"),
}
MODE_DEFAULTS = {
    "six_slot": {"layers": 7,
                 "names": ["bg", "panel", "controls", "assets",
                           "panel_f", "icon", "text"]},
    "panelz": {"layers": 6, "names": None},  # None -> bg,z0..z{n-2}
}

pipe = None
transformer = None
prompt_caches = {}
loaded_modes = []
run_lock = threading.Lock()


def load_model() -> None:
    global pipe, transformer, prompt_caches, loaded_modes
    from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel
    from peft import PeftModel
    kwargs = {"subfolder": "transformer", "torch_dtype": torch.bfloat16}
    if QUANT == "fp8":
        from diffusers import TorchAoConfig
        from torchao.quantization import Float8WeightOnlyConfig
        kwargs["quantization_config"] = TorchAoConfig(Float8WeightOnlyConfig())
        kwargs["device_map"] = "cuda"
        print("loading base transformer (fp8 weight-only, experimental with PEFT)...",
              flush=True)
    else:
        print("loading base transformer (bf16)...", flush=True)
    tf = QwenImageTransformer2DModel.from_pretrained(BASE, **kwargs)

    first = True
    for mode, path in ADAPTERS.items():
        if not Path(path).is_dir():
            print(f"[warn] adapter missing, skip mode {mode}: {path}", flush=True)
            continue
        if first:
            tf = PeftModel.from_pretrained(tf, path, adapter_name=mode)
            first = False
        else:
            tf.load_adapter(path, adapter_name=mode)
        loaded_modes.append(mode)
    if not loaded_modes:
        raise RuntimeError("no adapter loaded; check QWEN_ADAPTER_* paths")
    transformer = tf

    pipe = QwenImageLayeredPipeline.from_pretrained(
        BASE, transformer=tf, text_encoder=None, tokenizer=None,
        processor=None, torch_dtype=torch.bfloat16)
    # 管线内部会取 text_encoder.dtype;塞只带 dtype 的替身(不载 7B 编码器)
    class _Dummy:
        dtype = torch.bfloat16
    object.__setattr__(pipe, "text_encoder", _Dummy())
    if QUANT != "fp8":
        tf.to("cuda")
    pipe.vae.to("cuda")
    # 管线尾部 (layers+1) 帧一批 vae.decode 是显存峰值元凶,包成逐帧分块
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

    # torch.compile:常驻进程吃编译收益(~20-30% 提速)。dynamic=True 让
    # 一次编译覆盖不同纵横比/帧数的序列长度,避免逐形状重编;
    # 首次调用(每个 mode)会多花 2~5 分钟编译预热。QWEN_LAYERED_COMPILE=0 关闭
    if COMPILE:
        try:
            tf.compile(dynamic=True)
            print("transformer compiled (dynamic)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] torch.compile 失败,回退 eager: {e}", flush=True)

    for mode in loaded_modes:
        prompt_caches[mode] = torch.load(
            PROMPT_CACHES[mode], map_location="cpu", weights_only=False)
    print(f"qwen layered ready, modes: {loaded_modes}", flush=True)


def decompose(req: dict) -> dict:
    from PIL import Image
    mode = req.get("mode", "panelz")
    if mode not in loaded_modes:
        raise ValueError(f"mode {mode} 未加载(可用: {loaded_modes})")
    run_dir = Path(req["dir"])
    image_path = run_dir / req.get("image", "mid_fill.png")
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    out_rel = req.get("output_dir", "panel_layers_qwen")
    defaults = MODE_DEFAULTS[mode]
    layers = int(req.get("layers", defaults["layers"]))
    names = req.get("names") or defaults["names"] or (
        ["bg"] + [f"z{i}" for i in range(layers - 1)])
    if len(names) != layers:
        raise ValueError(f"names 长度 {len(names)} != layers {layers}")
    steps = int(req.get("steps", 40))
    seed = int(req.get("seed", 7))
    true_cfg = float(req.get("true_cfg", 4.0))
    resolution = int(req.get("resolution", 640))

    started = time.time()
    transformer.set_adapter(mode)
    img = Image.open(image_path).convert("RGBA")
    gen = torch.Generator("cuda").manual_seed(seed)
    pe = prompt_caches[mode]
    dev = lambda t: t.to("cuda") if t is not None else None  # noqa: E731
    # 正向传文本仅为绕过空 prompt 的自动配文(需编码器);负向只许传 embeds
    result = pipe(image=img,
                  prompt=pe["prompt"],
                  prompt_embeds=dev(pe["prompt_embeds"]),
                  prompt_embeds_mask=dev(pe["prompt_mask"]),
                  negative_prompt_embeds=dev(pe["neg_embeds"]),
                  negative_prompt_embeds_mask=dev(pe["neg_mask"]),
                  true_cfg_scale=true_cfg, layers=layers,
                  num_inference_steps=steps, generator=gen,
                  resolution=resolution)
    out_dir = run_dir / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    files = []
    for name, layer in zip(names, result.images[0]):
        f = f"{out_rel}/{name}.png"
        layer.save(run_dir / f)
        files.append(f)
    manifest = {"files": files, "mode": mode, "layers": layers,
                "names": names, "steps": steps, "seed": seed,
                "true_cfg": true_cfg, "resolution": resolution,
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
            self._send(200, {"ok": pipe is not None, "modes": loaded_modes})
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
