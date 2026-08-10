"""YOLO 常驻检测服务(跑在 /workspace/ui_skin/.venv,由 yolod.sh 启动)。

模型只加载一次,监听 127.0.0.1:8190(仅本机,由 service 代理调用):
    GET  /health   就绪探针(模型加载完成后才返回 ok)
    POST /detect   检测,请求体:
        {
          "dir": "/workspace/servData/<run_id>",   图片所在目录
          "image": "origin.png",                   输入图片名
          "imgsz": 1600,
          "conf": 0.1,
          "iou": 0.7,
          "augment": false,                        TTA:翻转+多尺度融合,慢 2~3 倍
          "slice": false,                          SAHI 切片推理,小目标检出率显著提升
          "slice_size": 640                        切片边长(slice=true 时生效)
        }
    返回:
        {
          "lines": ["<class_id> <cx> <cy> <w> <h> <conf>", ...],   YOLO save_txt 同款格式(归一化)
          "names": {"0": "text", ...},                              类别表
          "count": N
        }
"""
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = int(os.environ.get("YOLOD_PORT", "8190"))
# 多模型注册表:请求可用 "model" 字段按 key 选择;默认模型启动即载,其余懒加载缓存
MODEL_PATHS = {
    "game0804_11m": "/workspace/ui_skin/pretrained/yolo/yolo_game0804_best.pt",
    "game0804_p2": "/workspace/ui_skin/pretrained/yolo/yolo_game0804_p2_best.pt",
    "game0728_p2": "/workspace/ui_skin/pretrained/yolo/yolo_game0728_p2_best.pt",
}
# 兼容旧环境变量:YOLO_MODEL 指定的路径注册为 "env" 并作为默认
if os.environ.get("YOLO_MODEL"):
    MODEL_PATHS["env"] = os.environ["YOLO_MODEL"]
DEFAULT_MODEL = os.environ.get(
    "YOLO_DEFAULT_MODEL", "env" if "env" in MODEL_PATHS else "game0804_11m")

_models: dict = {}
sahi_model = None
predict_lock = threading.Lock()


def get_model(key: str):
    """按 key 取模型,未加载则现场加载并缓存(每个 ~40M,常驻显存无压力)。"""
    if key not in MODEL_PATHS:
        raise ValueError(f"unknown model {key!r}, available: {sorted(MODEL_PATHS)}")
    if key not in _models:
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        from ultralytics import YOLO
        print(f"loading YOLO[{key}] from {MODEL_PATHS[key]} ...", flush=True)
        _models[key] = YOLO(MODEL_PATHS[key])
        print(f"YOLO[{key}] ready", flush=True)
    return _models[key]


def load_model() -> None:
    get_model(DEFAULT_MODEL)


def _get_sahi_model(conf: float):
    """惰性加载 SAHI 包装的检测模型(切片推理用)。"""
    global sahi_model
    from sahi import AutoDetectionModel
    if sahi_model is None:
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=MODEL_PATHS[DEFAULT_MODEL],
            confidence_threshold=conf, device="cuda:0",
        )
    else:
        sahi_model.confidence_threshold = conf
    return sahi_model


def _detect_sliced(image_path: Path, conf: float, slice_size: int) -> dict:
    """SAHI 切片推理:大图切成带重叠的小块分别检测再合并,小目标检出率更高。"""
    from PIL import Image as PILImage
    from sahi.predict import get_sliced_prediction
    with PILImage.open(image_path) as im:
        width, height = im.size
    result = get_sliced_prediction(
        str(image_path), _get_sahi_model(conf),
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=0.25, overlap_width_ratio=0.25,
        verbose=0,
    )
    lines, names = [], {}
    for pred in result.object_prediction_list:
        b = pred.bbox
        cx = (b.minx + b.maxx) / 2 / width
        cy = (b.miny + b.maxy) / 2 / height
        w = (b.maxx - b.minx) / width
        h = (b.maxy - b.miny) / height
        cls = int(pred.category.id)
        names[str(cls)] = pred.category.name
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {pred.score.value:.4f}")
    return {"lines": lines, "names": names, "count": len(lines)}


def detect(req: dict) -> dict:
    image_path = Path(req["dir"]) / req.get("image", "origin.png")
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    imgsz = int(req.get("imgsz", 1600))
    conf = float(req.get("conf", 0.05))
    iou = float(req.get("iou", 0.7))
    augment = bool(req.get("augment", False))

    if req.get("slice"):
        return _detect_sliced(image_path, conf, int(req.get("slice_size", 640)))

    model = get_model(str(req.get("model") or DEFAULT_MODEL))
    results = model.predict(
        source=str(image_path), imgsz=imgsz, conf=conf, iou=iou,
        augment=augment, device=0, verbose=False,
    )
    r = results[0]
    lines = []
    for box in r.boxes:
        cls = int(box.cls.item())
        c = float(box.conf.item())
        cx, cy, w, h = (float(v) for v in box.xywhn[0].tolist())
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {c:.4f}")
    names = {str(k): v for k, v in (r.names or {}).items()}
    return {"lines": lines, "names": names, "count": len(lines)}


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
            self._send(200, {"ok": DEFAULT_MODEL in _models,
                             "loaded": sorted(_models),
                             "available": sorted(MODEL_PATHS),
                             "default": DEFAULT_MODEL})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/detect":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length))
            with predict_lock:
                result = detect(req)
            self._send(200, result)
        except Exception:
            self._send(500, {"error": traceback.format_exc()[-3000:]})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    load_model()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"yolo daemon listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
