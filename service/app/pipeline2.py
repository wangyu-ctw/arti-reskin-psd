"""新管线(/pipeline2)服务端实现:全部编排与像素操作在 Python 完成,
前端只提交 p2_* 任务并读取 run 目录的 p2_state.json 渲染。

步骤任务(均注册到 worker,GPU1 泳道):
    p2_sixslot    六槽分层(qwen daemon mode=six_slot)+ 覆盖率过滤
    p2_panelz     bg+panel 合成 → panelz 分层 + 覆盖率过滤
    p2_yolo       非 panel 层逐层贴中性底 YOLO 清点(记录来源层)
    p2_gpt        GPT 审核(类型/cover/过拆/缺拆),缺拆走 SAM2 补拆
    p2_extract    元素抠取:孤立连通域直裁 + 粘连 SAM2 + 跨层并回合成
    p2_recompose  拼回对比图 p2_recompose.png

状态文件 p2_state.json 由本模块独占读写(键名用 camelCase,前端直读)。
与旧管线完全隔离:不改旧 handler 行为,只做在-process 复用调用。
"""
import base64
import io
import json
import math
import shutil
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

from PIL import Image, ImageDraw

from . import storage

SIX_SLOT_NAMES = ["bg", "panel", "controls", "assets", "panel_f", "icon", "text"]
DETECT_SLOTS = ["controls", "assets", "panel_f", "icon", "text"]
YOLO_CLASSES = ["text", "icon", "assets", "button", "bar", "panel"]
EMPTY_COVERAGE = 0.002
SIXSLOT_DIR = "p2_sixslot"
PANELZ_DIR = "p2_panelz"
ELEMENTS_DIR = "p2_elements"

SAM2_PARAMS = {
    "padding_ratio": 0.02, "min_padding": 2, "mask_threshold": 0.55,
    "feather_radius": 1, "crop_scale": 4.5, "refine": False,
    "multimask": True, "fill_holes": False,
}

GPT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["elements", "missing"],
    "properties": {
        "elements": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "type", "cover", "verdict", "merge_into"],
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": [
                    "text", "icon", "button", "assets", "bar", "panel",
                    "panel_f", "unknown"]},
                "cover": {"type": "string", "enum": [
                    "none", "selected", "highlight", "disabled", "claimed", "locked"]},
                "verdict": {"type": "string", "enum": ["ok", "over_split", "mis_type"]},
                "merge_into": {"type": "string"},
            }}},
        "missing": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["layer", "type", "bbox", "note"],
            "properties": {
                "layer": {"type": "string", "enum": DETECT_SLOTS},
                "type": {"type": "string", "enum": [
                    "text", "icon", "button", "assets", "bar"]},
                "bbox": {"type": "array", "items": {"type": "number"},
                         "minItems": 4, "maxItems": 4},
                "note": {"type": "string"},
            }}},
    },
}


# ---------- 基础工具 ----------

def _run_dir(payload: dict) -> Path:
    if payload.get("dir"):
        return Path(payload["dir"])
    return storage.get_run_dir(payload["run_id"])


def _state_path(run_dir: Path) -> Path:
    return run_dir / "p2_state.json"


def _load_state(run_dir: Path) -> dict:
    p = _state_path(run_dir)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _save_state(run_dir: Path, state: dict) -> None:
    _state_path(run_dir).write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


# 双泳道后 p2_detect 与 p2_sixslot 可能并行读写 p2_state.json,
# 所有写入统一走"锁内重读-合并-落盘",避免互相覆盖
_STATE_LOCK = threading.Lock()


def _mutate_state(run_dir: Path, patch: dict) -> dict:
    with _STATE_LOCK:
        state = _load_state(run_dir)
        state.update(patch)
        _save_state(run_dir, state)
        return state


def _coverage(img: Image.Image) -> float:
    a = img.getchannel("A")
    hist = a.histogram()
    solid = sum(hist[9:])
    return solid / (img.width * img.height)


def _layer_infos(run_dir: Path, out_dir: str, names: list) -> list:
    infos = []
    for name in names:
        f = f"{out_dir}/{name}.png"
        with Image.open(run_dir / f) as im:
            cov = _coverage(im.convert("RGBA"))
        infos.append({"name": name, "file": f,
                      "coverage": round(cov, 4),
                      "keep": cov >= EMPTY_COVERAGE})
    return infos


def _bbox_px(bbox, w, h):
    cx, cy, bw, bh = (float(v) for v in bbox[:4])
    return (int((cx - bw / 2) * w), int((cy - bh / 2) * h),
            int(math.ceil((cx + bw / 2) * w)), int(math.ceil((cy + bh / 2) * h)))


def _union_bbox(bboxes: list) -> list:
    x0 = min(b[0] - b[2] / 2 for b in bboxes)
    y0 = min(b[1] - b[3] / 2 for b in bboxes)
    x1 = max(b[0] + b[2] / 2 for b in bboxes)
    y1 = max(b[1] + b[3] / 2 for b in bboxes)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


# ---------- 步骤 1:六槽分层 ----------

def handle_p2_sixslot(payload: dict) -> dict:
    from . import tasks
    run_dir = _run_dir(payload)
    result = tasks.handle_qwen_layered({
        "dir": str(run_dir),
        "image": payload.get("image") or "origin.png",
        "mode": "six_slot",
        "layers": len(SIX_SLOT_NAMES),
        "names": SIX_SLOT_NAMES,
        "output_dir": SIXSLOT_DIR,
        "steps": payload.get("steps", 40),
        "seed": payload.get("seed", 7),
        "true_cfg": payload.get("true_cfg", 4.0),
        "resolution": payload.get("resolution", 1024),
    })
    with Image.open(run_dir / (payload.get("image") or "origin.png")) as im:
        size = {"w": im.width, "h": im.height}
    layers = _layer_infos(run_dir, SIXSLOT_DIR, SIX_SLOT_NAMES)
    # 生成完立刻绿底逐层 YOLO(text/icon/panel_f/assets/controls)
    layer_yolo = []
    kept = {l["name"]: l for l in layers if l["keep"]}
    for nm in LAYER_DETECT_SLOTS:
        if nm in kept:
            layer_yolo += _cc_on_layer(run_dir, nm, kept[nm]["file"], payload)
    state = _mutate_state(run_dir, {
        "imageSize": size,
        "slotLayers": layers,
        "layerYolo": layer_yolo,
        "panelzLayers": [], "elements": [], "missing": [],
        "originInventory": [], "inventoryStats": None,
        "gptSummary": "", "extractStats": "",
        "yoloDropped": 0, "yoloRescued": 0,
    })  # originYolo(⓪⁺ 并行产出)保留
    return {"elapsed_sec": result.get("elapsed_sec"),
            "slotLayers": state["slotLayers"]}


# ---------- 步骤 2:panelz ----------

def handle_p2_panelz(payload: dict) -> dict:
    from . import tasks
    run_dir = _run_dir(payload)
    state = _load_state(run_dir)
    layers = {l["name"]: l for l in state.get("slotLayers", [])}
    if "bg" not in layers or "panel" not in layers:
        raise RuntimeError("先跑 p2_sixslot")
    if not layers["panel"]["keep"]:
        raise RuntimeError("panel 层为空,无需 panelz")
    with Image.open(run_dir / layers["bg"]["file"]) as im:
        comp = Image.new("RGBA", im.size, (255, 255, 255, 255))
        comp.alpha_composite(im.convert("RGBA"))
    with Image.open(run_dir / layers["panel"]["file"]) as im:
        comp.alpha_composite(im.convert("RGBA"))
    comp.convert("RGB").save(run_dir / "p2_panelz_input.png")
    n_layers = int(payload.get("layers", 6))
    result = tasks.handle_qwen_layered({
        "dir": str(run_dir),
        "image": "p2_panelz_input.png",
        "mode": "panelz",
        "layers": n_layers,
        "output_dir": PANELZ_DIR,
        "steps": payload.get("steps", 40),
        "seed": payload.get("seed", 7),
        "true_cfg": payload.get("true_cfg", 4.0),
        "resolution": payload.get("resolution", 1024),
    })
    names = ["bg"] + [f"z{i}" for i in range(n_layers - 1)]
    zlayers = _layer_infos(run_dir, PANELZ_DIR, names)
    # z 层(除 bg 与弃层)绿底 YOLO
    z_yolo = []
    for z in zlayers:
        if z["name"] != "bg" and z["keep"]:
            z_yolo += _cc_on_layer(run_dir, z["name"], z["file"], payload)
    with _STATE_LOCK:
        st = _load_state(run_dir)
        old_ly = [e for e in st.get("layerYolo", [])
                  if not e["sourceLayer"].startswith("z")]
        st["panelzLayers"] = zlayers
        st["layerYolo"] = old_ly + z_yolo
        _save_state(run_dir, st)
        state = st
    return {"elapsed_sec": result.get("elapsed_sec"),
            "panelzLayers": state["panelzLayers"]}


# ---------- 步骤 1′:原图检测(老管线 yolo+gpt 移植,可与生成并行) ----------

_DET_ITEMS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "bbox": {"type": "array",
                     "items": {"type": "number", "minimum": 0, "maximum": 1},
                     "minItems": 4, "maxItems": 4},
            "conf": {"type": "number", "minimum": 0, "maximum": 1},
            "yolo_detect": {"type": "string",
                            "enum": ["remain", "discard", "missing"]},
        },
        "required": ["bbox", "conf", "yolo_detect"],
        "additionalProperties": False,
    },
}
DETECT_SCHEMA = {
    "type": "object",
    "properties": {c: _DET_ITEMS for c in
                   ["text", "icon", "assets", "button", "bar", "panel", "panel_f"]},
    "required": ["text", "icon", "assets", "button", "bar", "panel", "panel_f"],
    "additionalProperties": False,
}


def handle_p2_detect(payload: dict) -> dict:
    """⓪⁺ 原图检测(纯 YOLO,不走 VL):产出 originYolo,供汇总审核与查看。"""
    from . import tasks
    run_dir = _run_dir(payload)
    yolo = tasks.handle_yolo({
        "dir": str(run_dir), "image": "origin.png",
        "model": payload.get("yolo_model", "game0804_p2"),
        "imgsz": payload.get("imgsz", 1600),
        "conf": payload.get("conf", 0.1),
        "iou": payload.get("iou", 0.7),
        "txt_output": "p2_detect_yolo.txt",
    })
    boxes = []
    for line in yolo.get("lines", []):
        parts = line.split()
        if len(parts) < 5:
            continue
        ci = int(parts[0])
        boxes.append({
            "cls": YOLO_CLASSES[ci] if ci < len(YOLO_CLASSES) else "unknown",
            "bbox": [float(v) for v in parts[1:5]],
            "conf": float(parts[5]) if len(parts) > 5 else 0.0,
        })
    _mutate_state(run_dir, {"originYolo": boxes})
    return {"count": len(boxes),
            "byClass": {c: sum(1 for b in boxes if b["cls"] == c)
                        for c in set(b["cls"] for b in boxes)}}


GREEN_BG = (0, 255, 0, 255)
LAYER_DETECT_SLOTS = ["text", "icon", "panel_f", "assets", "controls"]


LAYER_CC_CLS = {"text": "text", "icon": "icon", "assets": "assets",
                "panel_f": "panel_f"}  # controls 按长宽比分 bar/button;z* → panel


def _cc_on_layer(run_dir, layer_name, layer_file, payload):
    """分离层 CV 连通域检测(不走 YOLO):层自带 alpha,连通域即元素,
    bbox 像素级贴合、零截断。text 层先做水平膨胀把同一行笔画连成行框。"""
    import numpy as _np
    with Image.open(run_dir / layer_file) as im:
        arr = _np.asarray(im.convert("RGBA"))
    alpha_thr = int(payload.get("alpha_thr", 8))
    min_area = int(payload.get("min_area", 30))
    mask = arr[..., 3] > alpha_thr
    H, W = mask.shape
    label_mask = mask
    if layer_name == "text":
        # 水平膨胀:同一行文字的碎笔画并成行
        k = max(4, W // 150)
        try:
            from scipy import ndimage
            label_mask = ndimage.binary_dilation(
                mask, structure=_np.ones((1, 2 * k + 1), dtype=bool))
        except Exception:  # noqa: BLE001
            label_mask = mask
    lab, n = _label_components(label_mask)
    out = []
    for kidx in range(1, n + 1):
        comp = (lab == kidx) & mask  # 紧致边界用原始 alpha,不吃膨胀量
        ys, xs = _np.nonzero(comp)
        if ys.size < min_area:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        w, h = x1 - x0, y1 - y0
        if layer_name in LAYER_CC_CLS:
            cls = LAYER_CC_CLS[layer_name]
        elif layer_name == "controls":
            ar = max(w / max(1, h), h / max(1, w))
            cls = "bar" if ar > 3.5 else "button"
        else:
            cls = "panel"
        out.append({
            "sourceLayer": layer_name, "sourceFile": layer_file,
            "cls": cls,
            "bbox": [((x0 + x1) / 2) / W, ((y0 + y1) / 2) / H, w / W, h / H],
            "conf": 0.9,
        })
    return out


def _yolo_on_layer(run_dir, layer_name, layer_file, payload):
    """分离层贴纯绿底后 YOLO,返回带来源层的检测清单。"""
    from . import tasks
    det_name = f"p2_det_{layer_name}.png"
    with Image.open(run_dir / layer_file) as im:
        base = Image.new("RGBA", im.size, GREEN_BG)
        base.alpha_composite(im.convert("RGBA"))
    base.convert("RGB").save(run_dir / det_name)
    r = tasks.handle_yolo({
        "dir": str(run_dir), "image": det_name,
        "model": payload.get("yolo_model", payload.get("model", "game0804_p2")),
        "imgsz": payload.get("imgsz", 1600),
        "conf": payload.get("conf", 0.1),
        "iou": payload.get("iou", 0.7),
        "refine_bbox": False,
        "txt_output": f"p2_yolo_{layer_name}.txt",
    })
    out = []
    for line in r.get("lines", []):
        parts = line.split()
        if len(parts) < 5:
            continue
        ci = int(parts[0])
        out.append({
            "sourceLayer": layer_name, "sourceFile": layer_file,
            "cls": YOLO_CLASSES[ci] if ci < len(YOLO_CLASSES) else "unknown",
            "bbox": [float(v) for v in parts[1:5]],
            "conf": float(parts[5]) if len(parts) > 5 else 0.0,
        })
    return out


# ---------- 步骤 3:逐层 YOLO 清点 + bbox 去重 ----------

# 各来源层的"本职类别":去重时本职检出优先于串槽检出
LAYER_EXPECTED = {
    "controls": {"button", "bar"}, "assets": {"assets"},
    "panel_f": {"panel"}, "icon": {"icon"}, "text": {"text"},
}


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def handle_p2_yolo(payload: dict) -> dict:
    """六槽的非 panel 层 + panelz 的 z 层逐层贴中性底检测;
    同一元素被画进多层(串槽泄漏)或近重框会产生重复 bbox——按
    "本职类别优先、置信度其次"贪心去重(IoU ≥ dedup_iou 视为重复)。"""
    from . import tasks
    run_dir = _run_dir(payload)
    state = _load_state(run_dir)
    slot_layers = {l["name"]: l for l in state.get("slotLayers", [])}
    detect_targets = [(s, slot_layers[s], LAYER_EXPECTED[s])
                      for s in DETECT_SLOTS
                      if s in slot_layers and slot_layers[s]["keep"]]
    # panelz 的 z 层也清点(panel 来源图与 bbox)
    for z in state.get("panelzLayers", []):
        if z["name"] != "bg" and z["keep"]:
            detect_targets.append((z["name"], z, {"panel"}))

    raw = []
    for slot, layer, expected in detect_targets:
        det_name = f"p2_det_{slot}.png"
        with Image.open(run_dir / layer["file"]) as im:
            base = Image.new("RGBA", im.size, (128, 128, 128, 255))
            base.alpha_composite(im.convert("RGBA"))
        base.convert("RGB").save(run_dir / det_name)
        result = tasks.handle_yolo({
            "dir": str(run_dir),
            "image": det_name,
            "model": payload.get("model", "game0804_p2"),
            "imgsz": payload.get("imgsz", 1600),
            "conf": payload.get("conf", 0.1),
            "iou": payload.get("iou", 0.7),
            "refine_bbox": False,
            "txt_output": f"p2_yolo_{slot}.txt",
        })
        for i, line in enumerate(result.get("lines", [])):
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_idx = int(parts[0])
            cls = YOLO_CLASSES[cls_idx] if cls_idx < len(YOLO_CLASSES) else "unknown"
            raw.append({
                "id": f"{slot}_{i}",
                "sourceLayer": slot,
                "sourceFile": layer["file"],
                "bbox": [float(v) for v in parts[1:5]],
                "cls": cls,
                "conf": float(parts[5]) if len(parts) > 5 else 0.0,
                "_native": cls in expected,
            })

    # 去重:本职优先、conf 其次,贪心保留与已留框 IoU < 阈值者
    dedup_iou = float(payload.get("dedup_iou", 0.8))
    raw.sort(key=lambda e: (e["_native"], e["conf"]), reverse=True)
    kept = []
    for e in raw:
        if all(_iou(e["bbox"], k["bbox"]) < dedup_iou for k in kept):
            kept.append(e)
    dropped = len(raw) - len(kept)
    for e in kept:
        e.pop("_native", None)

    # 原图对账兜底:生成侧丢掉的元素(尤其小 icon)在原图上仍可见。
    # 有 p2_detect 的审核清单(VL 纠错/补漏过)就用它;没有则裸 YOLO 扫一遍
    match_iou = float(payload.get("origin_match_iou", 0.45))
    origin_conf = float(payload.get("origin_conf", 0.25))
    inventory = state.get("originInventory") or []
    candidates = []
    if inventory:
        for it in inventory:
            if it["cls"] == "panel":
                continue  # panel 由 panelz 负责
            candidates.append((it["cls"], it["bbox"], it.get("conf", 1.0)))
    else:
        result = tasks.handle_yolo({
            "dir": str(run_dir), "image": "origin.png",
            "model": payload.get("model", "game0804_p2"),
            "imgsz": payload.get("imgsz", 1600),
            "conf": payload.get("conf", 0.1),
            "iou": payload.get("iou", 0.7),
            "refine_bbox": False,
            "txt_output": "p2_yolo_origin.txt",
        })
        for line in result.get("lines", []):
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_idx = int(parts[0])
            cls = YOLO_CLASSES[cls_idx] if cls_idx < len(YOLO_CLASSES) else "unknown"
            conf = float(parts[5]) if len(parts) > 5 else 0.0
            if cls == "panel" or conf < origin_conf:
                continue
            candidates.append((cls, [float(v) for v in parts[1:5]], conf))
    n_rescued = 0
    for i, (cls, bbox, conf) in enumerate(candidates):
        if any(_iou(bbox, k["bbox"]) >= match_iou for k in kept):
            continue
        kept.append({
            "id": f"origin_{i}", "sourceLayer": "origin",
            "sourceFile": "origin.png", "bbox": bbox,
            "cls": cls, "conf": conf, "fromOrigin": True,
        })
        n_rescued += 1

    _mutate_state(run_dir, {"elements": kept, "yoloDropped": dropped,
                            "yoloRescued": n_rescued})
    return {"count": len(kept), "dropped": dropped, "rescued": n_rescued}


# ---------- 步骤 4:GPT 审核 ----------

def _img_data_url(img: Image.Image, quality=90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _draw_overlay(origin: Image.Image, elements: list) -> Image.Image:
    colors = {"controls": "#ff4d4f", "assets": "#40a9ff", "panel_f": "#9254de",
              "icon": "#faad14", "text": "#52c41a"}
    img = origin.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    lw = max(2, w // 500)
    for el in elements:
        x0, y0, x1, y1 = _bbox_px(el["bbox"], w, h)
        color = colors.get(el["sourceLayer"], "#ff00ff")
        d.rectangle([x0, y0, x1, y1], outline=color, width=lw)
        d.text((x0 + 2, max(0, y0 - 14)), el["id"], fill=color)
    return img


def _call_openrouter(api_key: str, model: str, prompt: str,
                     user_text: str, images: list, schema: dict = None,
                     effort: str = None, speed: str = None) -> dict:
    """effort/speed 同旧管线第三步:reasoning.effort(none~max)、
    provider.sort(latency/throughput;balanced 走默认路由不传)"""
    schema = schema or GPT_SCHEMA
    is_gemini = "gemini" in model
    if is_gemini:
        url = "https://openrouter.ai/api/v1/chat/completions"
        content = [{"type": "text", "text": user_text}] + [
            {"type": "image_url", "image_url": {"url": u}} for u in images]
        body = {"model": model,
                "messages": [{"role": "system", "content": prompt},
                             {"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "p2_audit", "strict": True, "schema": schema}}}
    else:
        url = "https://openrouter.ai/api/v1/responses"
        content = [{"type": "input_text", "text": user_text}] + [
            {"type": "input_image", "image_url": u} for u in images]
        body = {"model": model, "instructions": prompt,
                "input": [{"role": "user", "content": content}],
                "text": {"format": {"type": "json_schema", "name": "p2_audit",
                                    "strict": True, "schema": schema}}}
    if effort:
        body["reasoning"] = {"effort": effort, "exclude": True}
    provider = {"require_parameters": True}
    if speed and speed != "balanced":
        provider["sort"] = speed
    body["provider"] = provider
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read().decode()[:300]}")
    # chat 与 responses 两种取文本
    text = None
    choices = data.get("choices")
    if choices and isinstance(choices[0].get("message", {}).get("content"), str):
        text = choices[0]["message"]["content"]
    if text is None:
        for o in data.get("output") or []:
            for p in o.get("content") or []:
                if isinstance(p.get("text"), str) and p["text"].strip():
                    text = p["text"]
                    break
            if text:
                break
    if text is None and isinstance(data.get("output_text"), str):
        text = data["output_text"]
    if text is None:
        raise RuntimeError("无法从模型响应中取出文本")
    return json.loads(text)


def handle_p2_gpt(payload: dict) -> dict:
    from . import tasks
    run_dir = _run_dir(payload)
    state = _load_state(run_dir)
    elements = state.get("elements", [])
    if not elements:
        raise RuntimeError("先跑 p2_yolo")
    # pop 而非 get:key 只在本次调用栈存活,任务对象内存里不留存,更不落盘
    api_key = payload.pop("api_key", "") or ""
    if not api_key.strip():
        raise RuntimeError("缺少 OpenRouter API Key(payload.api_key)")
    with Image.open(run_dir / "origin.png") as im:
        origin = im.convert("RGB")
        overlay = _draw_overlay(origin, elements)
        user_text = ("元素清单(JSON):\n" + json.dumps(
            [{"id": e["id"], "sourceLayer": e["sourceLayer"],
              "cls": e["cls"], "bbox": e["bbox"]} for e in elements],
            ensure_ascii=False) + "\n\n图1=原图,图2=编号框叠加图。请按系统要求输出。")
        result = _call_openrouter(
            api_key, payload.get("model", "openai/gpt-5.6-sol"),
            payload.get("prompt", ""), user_text,
            [_img_data_url(origin), _img_data_url(overlay)])

    by_id = {e["id"]: e for e in elements}
    for v in result.get("elements", []):
        el = by_id.get(v.get("id"))
        if not el:
            continue
        el["type"] = v.get("type")
        el["cover"] = v.get("cover")
        el["verdict"] = v.get("verdict")
        if v.get("cover") not in (None, "none"):
            el["skip"] = True
        host = by_id.get(v.get("merge_into") or "")
        if v.get("verdict") == "over_split" and host:
            el["mergedInto"] = host["id"]
            host["bbox"] = _union_bbox([host["bbox"], el["bbox"]])
    # 缺拆:对应层 SAM2 补拆
    slot_layers = {l["name"]: l for l in state.get("slotLayers", [])}
    missing = []
    for i, m in enumerate(result.get("missing", [])):
        layer = slot_layers.get(m.get("layer"))
        rec = dict(m)
        if layer:
            out = f"p2_missing_{i}.png"
            tasks.handle_sam2({
                "dir": str(run_dir), "image": layer["file"], "output": out,
                "borders": [{"bbox": m["bbox"]}], **SAM2_PARAMS})
            rec["file"] = out
            by_id[f"missing_{i}"] = {
                "id": f"missing_{i}", "sourceLayer": m["layer"],
                "sourceFile": layer["file"], "bbox": m["bbox"],
                "cls": m["type"], "conf": 1.0, "type": m["type"],
                "extract": {"method": "sam2_combined", "file": out,
                            "bbox": m["bbox"]},
            }
        missing.append(rec)
    merged = list(by_id.values())
    n_skip = sum(1 for e in merged if e.get("skip"))
    n_merge = sum(1 for e in merged if e.get("mergedInto"))
    summary = (f"审核 {len(result.get('elements', []))} 元素:cover 跳过 {n_skip}、"
               f"过拆并回 {n_merge}、缺拆补 {len(missing)}")
    _mutate_state(run_dir, {"elements": merged, "missing": missing,
                            "gptSummary": summary})
    return {"summary": summary}


# ---------- 步骤 5:元素抠取 ----------

def handle_p2_extract(payload: dict) -> dict:
    from . import tasks
    run_dir = _run_dir(payload)
    state = _load_state(run_dir)
    elements = state.get("elements", [])
    if not elements:
        raise RuntimeError("先跑 p2_yolo")
    by_id = {e["id"]: e for e in elements}

    def cross_child(e):
        host = by_id.get(e.get("mergedInto") or "")
        return bool(host and host["sourceFile"] != e["sourceFile"])

    active = [e for e in elements
              if not e.get("skip") and not e.get("extract")
              and (not e.get("mergedInto") or cross_child(e))]
    by_layer = {}
    for e in active:
        by_layer.setdefault(e["sourceFile"], []).append(e)

    n_crop = n_sam = 0
    for layer_file, els in by_layer.items():
        # 原图兜底元素:原图无 alpha,连通域判定不适用,直接整批 SAM2
        if layer_file == "origin.png":
            out = "p2_sam2cut_origin.png"
            tasks.handle_sam2({
                "dir": str(run_dir), "image": layer_file, "output": out,
                "borders": [{"bbox": e["bbox"]} for e in els],
                **SAM2_PARAMS})
            for e in els:
                by_id[e["id"]]["extract"] = {
                    "method": "sam2_combined", "file": out, "bbox": e["bbox"]}
                n_sam += 1
            continue
        result = tasks.handle_element_extract({
            "dir": str(run_dir), "layer": layer_file,
            "elements": [{"id": e["id"], "bbox": e["bbox"]} for e in els],
            "out_dir": ELEMENTS_DIR,
        })
        for s in result["saved"]:
            by_id[s["id"]]["extract"] = {
                "method": "crop", "file": s["file"], "bbox": s["bbox"]}
            n_crop += 1
        if result["needs_sam2"]:
            slot = els[0]["sourceLayer"]
            out = f"p2_sam2cut_{slot}.png"
            tasks.handle_sam2({
                "dir": str(run_dir), "image": layer_file, "output": out,
                "borders": [{"bbox": by_id[i]["bbox"]}
                            for i in result["needs_sam2"]],
                **SAM2_PARAMS})
            for i in result["needs_sam2"]:
                by_id[i]["extract"] = {"method": "sam2_combined", "file": out,
                                       "bbox": by_id[i]["bbox"]}
                n_sam += 1

    # 跨层过拆并回:宿主+跨层子块素材合成一张并集图
    size = state.get("imageSize") or {}
    W, H = size.get("w"), size.get("h")
    n_union = 0
    if W and H:
        for host in by_id.values():
            if host.get("mergedInto") or not host.get("extract"):
                continue
            children = [c for c in by_id.values()
                        if c.get("mergedInto") == host["id"]
                        and c["sourceFile"] != host["sourceFile"]
                        and c.get("extract")]
            if not children:
                continue
            parts = [host] + children
            ub = _union_bbox([p["extract"]["bbox"] for p in parts])
            ux0, uy0, ux1, uy1 = _bbox_px(ub, W, H)
            canvas = Image.new("RGBA", (max(1, ux1 - ux0), max(1, uy1 - uy0)),
                               (0, 0, 0, 0))
            for p in parts:
                ex = p["extract"]
                with Image.open(run_dir / ex["file"]) as im:
                    asset = im.convert("RGBA")
                bx0, by0, bx1, by1 = _bbox_px(ex["bbox"], W, H)
                if ex["method"] == "sam2_combined":
                    sw, sh = asset.size
                    sx0, sy0, sx1, sy1 = _bbox_px(ex["bbox"], sw, sh)
                    asset = asset.crop((sx0, sy0, sx1, sy1))
                if asset.size != (bx1 - bx0, by1 - by0):
                    asset = asset.resize((max(1, bx1 - bx0), max(1, by1 - by0)),
                                         Image.LANCZOS)
                canvas.alpha_composite(asset, (bx0 - ux0, by0 - uy0))
            fname = f"{ELEMENTS_DIR}/{host['id']}_merged.png"
            canvas.save(run_dir / fname)
            host["extract"] = {"method": "crop", "file": fname, "bbox": ub}
            n_union += 1

    stats = (f"直裁 {n_crop} 个、SAM2 抠 {n_sam} 个"
             + (f"、跨层并回合成 {n_union} 个" if n_union else ""))
    _mutate_state(run_dir, {"elements": list(by_id.values()),
                            "extractStats": stats})
    return {"stats": stats}


# ---------- 步骤 5ᵇ:全量素材化 ----------

# 素材 z 序:text > icon > panel_f > assets > controls > panelz z(大到小) > bg
ASSET_SWEEP_SLOTS = ["icon", "panel_f", "assets", "controls"]

TEXT_SAM2_PARAMS = {
    "padding_ratio": 0.01, "min_padding": 1, "mask_threshold": 0.7,
    "feather_radius": 1, "crop_scale": 5, "refine": False,
    "multimask": True, "fill_holes": False,
}


def _label_components(mask):
    """alpha 布尔图 → (标签图, 数量)。优先 scipy(C 速度),缺则 BFS 兜底。"""
    import numpy as _np
    try:
        from scipy import ndimage
        return ndimage.label(mask)
    except Exception:  # noqa: BLE001
        from collections import deque as _deque
        lab = _np.zeros(mask.shape, dtype=_np.int32)
        h, w = mask.shape
        n = 0
        for sy, sx in zip(*_np.nonzero(mask)):
            if lab[sy, sx]:
                continue
            n += 1
            dq = _deque([(int(sy), int(sx))])
            lab[sy, sx] = n
            while dq:
                y, x = dq.popleft()
                for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = n
                        dq.append((ny, nx))
        return lab, n


def handle_p2_assets(payload: dict) -> dict:
    """全量素材化:icon/panel_f/assets/controls + panelz 各 z 层,按 alpha
    连通域直裁成最小尺寸 PNG(不用 SAM2);text 层按 YOLO bbox + SAM2 mask
    在 text 层上切。账本写 p2_assets.json:{zOrder, background, assets[]},
    每条素材记 文件路径/归一化 bbox/来源层图/zIndex。

    payload: run_id|dir, alpha_thr=8, min_area=30(px², 过滤噪点)
    """
    import numpy as _np
    from . import tasks
    run_dir = _run_dir(payload)
    state = _load_state(run_dir)
    slot_layers = {l["name"]: l for l in state.get("slotLayers", [])}
    if not slot_layers:
        raise RuntimeError("先跑 p2_sixslot")
    alpha_thr = int(payload.get("alpha_thr", 8))
    min_area = int(payload.get("min_area", 30))

    out_dir = run_dir / "p2_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    # z 序表:bg=0 → panelz z0..zN → controls → assets → panel_f → icon → text
    z_layers = [z for z in state.get("panelzLayers", [])
                if z["name"] != "bg" and z["keep"]]
    zmap = {"bg": 0}
    for i, z in enumerate(z_layers):
        zmap[z["name"]] = 1 + i
    base = 1 + len(z_layers)
    for j, nm in enumerate(["controls", "assets", "panel_f", "icon", "text"]):
        zmap[nm] = base + j

    assets = []

    def sweep_layer(name: str, layer: dict) -> int:
        """连通域直裁一整层,返回产出数"""
        with Image.open(run_dir / layer["file"]) as im:
            rgba = im.convert("RGBA")
        arr = _np.asarray(rgba)
        mask = arr[..., 3] > alpha_thr
        lab, n = _label_components(mask)
        H, W = mask.shape
        count = 0
        for k in range(1, n + 1):
            ys, xs = _np.nonzero(lab == k)
            if ys.size < min_area:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            crop = arr[y0:y1, x0:x1].copy()
            crop[..., 3] = _np.where(lab[y0:y1, x0:x1] == k, crop[..., 3], 0)
            count += 1
            fname = f"p2_assets/{name}_{count:03d}.png"
            Image.fromarray(crop).save(run_dir / fname)
            assets.append({
                "id": f"{name}_{count:03d}", "file": fname,
                "sourceLayer": name, "sourceFile": layer["file"],
                "bbox": [((x0 + x1) / 2) / W, ((y0 + y1) / 2) / H,
                         (x1 - x0) / W, (y1 - y0) / H],
                "zIndex": zmap[name], "area": int(ys.size), "method": "cc",
            })
        return count

    by_layer = {}
    for name in ASSET_SWEEP_SLOTS:
        layer = slot_layers.get(name)
        if layer and layer["keep"]:
            by_layer[name] = sweep_layer(name, layer)
    for z in z_layers:
        by_layer[z["name"]] = sweep_layer(z["name"], z)

    # text:按 YOLO bbox + SAM2 mask 在 text 层上切
    text_layer = slot_layers.get("text")
    text_boxes = [e["bbox"] for e in state.get("elements", [])
                  if e.get("sourceLayer") == "text"]
    if not text_boxes:
        text_boxes = [it["bbox"] for it in state.get("originInventory", [])
                      if it.get("cls") == "text"]
    n_text = 0
    if text_layer and text_layer["keep"] and text_boxes:
        cut_name = "p2_assets_textcut.png"
        tasks.handle_sam2({
            "dir": str(run_dir), "image": text_layer["file"],
            "output": cut_name,
            "borders": [{"bbox": b} for b in text_boxes],
            **TEXT_SAM2_PARAMS})
        with Image.open(run_dir / cut_name) as im:
            cut = im.convert("RGBA")
        carr = _np.asarray(cut)
        H, W = carr.shape[:2]
        for b in text_boxes:
            x0, y0, x1, y1 = _bbox_px(b, W, H)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            region = carr[y0:y1, x0:x1]
            if region.size == 0:
                continue
            ys, xs = _np.nonzero(region[..., 3] > alpha_thr)
            if ys.size < min_area:
                continue
            ty0, ty1 = y0 + int(ys.min()), y0 + int(ys.max()) + 1
            tx0, tx1 = x0 + int(xs.min()), x0 + int(xs.max()) + 1
            n_text += 1
            fname = f"p2_assets/text_{n_text:03d}.png"
            Image.fromarray(carr[ty0:ty1, tx0:tx1]).save(run_dir / fname)
            assets.append({
                "id": f"text_{n_text:03d}", "file": fname,
                "sourceLayer": "text", "sourceFile": text_layer["file"],
                "bbox": [((tx0 + tx1) / 2) / W, ((ty0 + ty1) / 2) / H,
                         (tx1 - tx0) / W, (ty1 - ty0) / H],
                "zIndex": zmap["text"], "area": int(ys.size),
                "method": "sam2_text",
            })
        by_layer["text"] = n_text

    manifest = {
        "zOrder": zmap,
        "background": ({"file": slot_layers["bg"]["file"], "zIndex": 0}
                       if slot_layers.get("bg") else None),
        "assets": assets,
    }
    (run_dir / "p2_assets.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    _mutate_state(run_dir, {"assetsSummary": {
        "count": len(assets), "byLayer": by_layer, "file": "p2_assets.json"}})
    return {"count": len(assets), "byLayer": by_layer}


def handle_p2_layer_yolo(payload: dict) -> dict:
    """重跑层检测(不重新生成):scope=six(六槽五层)/panelz(z 层)。
    用现有层文件重新绿底 YOLO,替换 layerYolo 中对应来源的条目。"""
    run_dir = _run_dir(payload)
    state = _load_state(run_dir)
    scope = payload.get("scope", "six")
    dets = []
    if scope == "six":
        kept = {l["name"]: l for l in state.get("slotLayers", []) if l["keep"]}
        targets = [(n, kept[n]["file"]) for n in LAYER_DETECT_SLOTS if n in kept]
        drop = set(LAYER_DETECT_SLOTS)
    else:
        targets = [(z["name"], z["file"]) for z in state.get("panelzLayers", [])
                   if z["name"] != "bg" and z["keep"]]
        drop = {n for n, _ in targets} | {
            e["sourceLayer"] for e in state.get("layerYolo", [])
            if e["sourceLayer"].startswith("z")}
    if not targets:
        raise RuntimeError(f"scope={scope} 没有可检测的层(先跑生成)")
    for nm, lf in targets:
        dets += _cc_on_layer(run_dir, nm, lf, payload)
    with _STATE_LOCK:
        st = _load_state(run_dir)
        old = [e for e in st.get("layerYolo", []) if e["sourceLayer"] not in drop]
        st["layerYolo"] = old + dets
        _save_state(run_dir, st)
    return {"scope": scope, "count": len(dets)}


# ---------- 步骤 2:汇总审核(p2_inventory:合集去重 + VL 复审) ----------

INVENTORY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["elements", "missing"],
    "properties": {
        "elements": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "type", "action", "dup_of"],
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": [
                    "text", "icon", "button", "assets", "bar",
                    "panel", "panel_f", "unknown"]},
                "action": {"type": "string", "enum": ["keep", "dup", "discard"]},
                "dup_of": {"type": "string"},
            }}},
        "missing": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["type", "bbox", "note"],
            "properties": {
                "type": {"type": "string", "enum": [
                    "text", "icon", "button", "assets", "bar"]},
                "bbox": {"type": "array", "items": {"type": "number"},
                         "minItems": 4, "maxItems": 4},
                "note": {"type": "string"},
            }}},
    },
}


def handle_p2_inventory(payload: dict) -> dict:
    """② 汇总审核:原图 YOLO(⓪⁺)+ 各层绿底 YOLO 的合集 → 机械去重
    (来源优先级:原图 > 本职层 > 其它,conf 决胜)→ VL 复审
    (类别纠正 / 再去重 dup / 剔除 discard / 补漏 missing)。
    产物 originInventory(级联切取的裁切权威)。"""
    from . import tasks  # noqa: F401
    run_dir = _run_dir(payload)
    state = _load_state(run_dir)
    origin_yolo = state.get("originYolo") or []
    layer_yolo = state.get("layerYolo") or []
    if not origin_yolo and not layer_yolo:
        raise RuntimeError("先跑 ⓪⁺ 原图 YOLO 或 ① 分层(层检测)")
    # pop 而非 get:key 只在本次调用栈存活,任务对象内存里不留存,更不落盘
    api_key = payload.pop("api_key", "") or ""
    if not api_key.strip():
        raise RuntimeError("缺少 OpenRouter API Key(payload.api_key)")
    dedup_iou = float(payload.get("dedup_iou", 0.8))

    # 合集 + 机械去重
    # 优先级:层连通域(bbox 像素级贴合层内容,零截断)> 原图 YOLO
    union = []
    for b in origin_yolo:
        union.append({**b, "sourceLayer": "origin", "_prio": 1,
                      "_native": True})
    for e in layer_yolo:
        union.append({**e, "_prio": 2, "_native": True})
    union.sort(key=lambda x: (x["_prio"], x["_native"], x["conf"]), reverse=True)
    deduped = []
    for e in union:
        if all(_iou(e["bbox"], k["bbox"]) < dedup_iou for k in deduped):
            deduped.append(e)
    n_union, n_dedup = len(union), len(deduped)
    cand = []
    for i, e in enumerate(deduped):
        cand.append({"id": f"c_{i:03d}", "cls": e["cls"], "bbox": e["bbox"],
                     "conf": e.get("conf", 0), "sourceLayer": e["sourceLayer"]})

    # VL 复审
    with Image.open(run_dir / "origin.png") as im:
        origin = im.convert("RGB")
        overlay = _draw_overlay(origin, cand)
        user_text = (
            "候选清单(JSON,来自原图 YOLO 与各分离层 YOLO 的合集,已做一轮"
            "机械去重,仍可能残留跨层重复):\n"
            + json.dumps([{"id": c["id"], "cls": c["cls"],
                           "sourceLayer": c["sourceLayer"], "bbox": c["bbox"]}
                          for c in cand], ensure_ascii=False)
            + "\n\n图1=原图,图2=编号框叠加图。请按系统要求逐项输出"
              "(type 类别纠正;重复项 action=dup 且 dup_of 指向保留项;"
              "误检 action=discard;并补漏 missing)。")
        result = _call_openrouter(
            api_key, payload.get("model", "openai/gpt-5.6-sol"),
            payload.get("prompt", ""), user_text,
            [_img_data_url(origin), _img_data_url(overlay)],
            schema=INVENTORY_SCHEMA,
            effort=payload.get("effort", "high"),
            speed=payload.get("speed", "balanced"))

    by_id = {c["id"]: c for c in cand}
    inventory = []
    n_dup = n_discard = 0
    for v in result.get("elements", []):
        c = by_id.get(v.get("id"))
        if not c:
            continue
        if v.get("action") == "dup":
            n_dup += 1
            continue
        if v.get("action") == "discard":
            n_discard += 1
            continue
        cls = v.get("type") or c["cls"]
        if cls == "unknown":
            cls = c["cls"]
        inventory.append({"cls": cls, "bbox": c["bbox"],
                          "conf": c["conf"], "status": "keep"})
    n_missing = 0
    for m in result.get("missing", []):
        inventory.append({"cls": m["type"], "bbox": m["bbox"],
                          "conf": 1.0, "status": "missing"})
        n_missing += 1

    stats = {"union": n_union, "afterDedup": n_dedup, "vlDup": n_dup,
             "vlDiscard": n_discard, "vlMissing": n_missing,
             "final": len(inventory)}
    # 隐式落盘完整审计记录(排障用,前端不展示):合集→候选→VL 裁决→最终
    (run_dir / "p2_inventory.json").write_text(json.dumps({
        "stats": stats,
        "originYolo": origin_yolo,
        "layerYolo": layer_yolo,
        "candidates": cand,
        "vlVerdicts": result.get("elements", []),
        "vlMissing": result.get("missing", []),
        "final": inventory,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    _mutate_state(run_dir, {"originInventory": inventory,
                            "inventoryStats": stats})
    return stats


# ---------- 步骤 5:级联切取(p2_cascade) ----------
#
# 自顶向下逐层(text→icon→panel_f→assets→controls→zN..z0):
#   元素层:清单(⓪⁺ originInventory)为裁切权威——本层类别的 bbox 逐个裁
#          (孤立=连通域直裁,粘连=SAM2);没切到的进 miss 登记簿;
#          上层残留(temp)落在本层元素框上→单独成材+记叠压关系(不合并);
#   面板层:layered 为权威——先救援登记簿里的元素(粘连在 panel 上的
#          SAM2 提取+类型纠正),再连通域全裁成 panel 素材;temp 残留
#          完全落在某 panel 内或轮廓相近→合并进该 panel;清单 panel 框只软对账;
#   终局:temp=碎屑图(PSD 置顶临时层);登记簿余额走差分判定
#         (bg≈原图→烙在 bg:SAM2 提取+批量 flux fill 净化 bg;
#          否则→原图 SAM2 提取);全量重影审计。

CLS_HOME = {"text": "text", "icon": "icon", "assets": "assets",
            "button": "controls", "bar": "controls"}
CASCADE_DIR = "p2_cascade"          # 交付目录:只放切下的素材 + bg.png + debris.png
CASCADE_DBG_DIR = "p2_cascade_dbg"  # 过程目录:temp 快照、SAM2 批切、probe 等中间产物


def _rgba_arr(run_dir: Path, file: str, size=None):
    import numpy as _np
    with Image.open(run_dir / file) as im:
        img = im.convert("RGBA")
    if size and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return _np.asarray(img).copy()


def _tight_cut(arr, mask):
    """按掩码从 RGBA 数组紧裁,返回 (crop_rgba, [x0,y0,x1,y1]);mask 空返回 None"""
    import numpy as _np
    ys, xs = _np.nonzero(mask)
    if ys.size == 0:
        return None, None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = arr[y0:y1, x0:x1].copy()
    crop[..., 3] = _np.where(mask[y0:y1, x0:x1], crop[..., 3], 0)
    return crop, [x0, y0, x1, y1]


def _px_bbox_norm(px, W, H):
    x0, y0, x1, y1 = px
    return [((x0 + x1) / 2) / W, ((y0 + y1) / 2) / H, (x1 - x0) / W, (y1 - y0) / H]


def handle_p2_cascade(payload: dict) -> dict:  # noqa: C901
    import numpy as _np
    from . import tasks
    run_dir = _run_dir(payload)
    state = _load_state(run_dir)
    slot_layers = {l["name"]: l for l in state.get("slotLayers", [])}
    inventory = state.get("originInventory") or []
    if not slot_layers:
        raise RuntimeError("先跑 p2_sixslot")
    if not inventory:
        raise RuntimeError("先跑 p2_detect(级联以审核清单为裁切权威)")
    alpha_thr = int(payload.get("alpha_thr", 8))
    min_area = int(payload.get("min_area", 30))
    overflow_px = int(payload.get("overflow_px", 10))
    land_ratio = float(payload.get("land_ratio", 0.4))
    diff_bg = float(payload.get("diff_bg_thr", 14))     # < 此值 → 烙在 bg
    diff_origin = float(payload.get("diff_origin_thr", 34))  # > 此值 → 已蒸发

    # 重跑=整目录删除重建:交付目录里永远只有本次的素材,不残留旧文件
    out_dir = run_dir / CASCADE_DIR
    dbg_dir = run_dir / CASCADE_DBG_DIR
    for d in (out_dir, dbg_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    # 画布基准 = text 层尺寸(六槽各层同尺寸;panelz 层重采样对齐)
    ref_layer = next((slot_layers[n] for n in SIX_SLOT_NAMES if n in slot_layers), None)
    with Image.open(run_dir / ref_layer["file"]) as im:
        CANVAS = im.size
    W, H = CANVAS

    # 层序(高 z → 低 z)与 zIndex 表
    z_layers = [z for z in state.get("panelzLayers", [])
                if z["name"] != "bg" and z["keep"]]
    zmap = {"bg": 0}
    for i, z in enumerate(z_layers):
        zmap[z["name"]] = 1 + i
    base = 1 + len(z_layers)
    for j, nm in enumerate(["controls", "assets", "panel_f", "icon", "text"]):
        zmap[nm] = base + j
    seq = [("text", "element"), ("icon", "element"), ("panel_f", "panel"),
           ("assets", "element"), ("controls", "element")]
    seq += [(z["name"], "panel") for z in reversed(z_layers)]
    layer_files = {**{n: l["file"] for n, l in slot_layers.items()},
                   **{z["name"]: z["file"] for z in z_layers}}

    # 元素清单 → 待裁项;panel 类只留软对账
    items = []
    for i, it in enumerate(inventory):
        if it["cls"] in CLS_HOME:
            items.append({"id": f"{it['cls']}_{i:03d}", "cls": it["cls"],
                          "bbox": it["bbox"], "home": CLS_HOME[it["cls"]],
                          "resolved": False, "note": ""})
    panel_inv = [it for it in inventory if it["cls"] in ("panel", "panel_f")]

    assets = []
    asset_seq = {}

    def new_asset(prefix):
        asset_seq[prefix] = asset_seq.get(prefix, 0) + 1
        return f"{prefix}_{asset_seq[prefix]:03d}"

    def save_asset(crop, px, *, cls, layer_name, method, aid=None,
                   stacked_on=None, origin_layer=None, recovered=None):
        aid = aid or new_asset(cls)
        fname = f"{CASCADE_DIR}/{aid}.png"
        Image.fromarray(crop).save(run_dir / fname)
        rec = {"id": aid, "file": fname, "cls": cls,
               "bbox": _px_bbox_norm(px, W, H),
               "sourceLayer": layer_name, "zIndex": zmap.get(layer_name, 0),
               "method": method}
        if stacked_on:
            rec["stackedOn"] = stacked_on
        if origin_layer:
            rec["originLayer"] = origin_layer
        if recovered:
            rec["recoveredFrom"] = recovered
        assets.append(rec)
        return aid

    def sam2_batch(image_file, boxes, out_name):
        """一层一批 SAM2,返回 cutout RGBA 数组(画布尺寸)"""
        tasks.handle_sam2({
            "dir": str(run_dir), "image": image_file,
            "output": f"{CASCADE_DBG_DIR}/{out_name}",
            "borders": [{"bbox": b} for b in boxes], **SAM2_PARAMS})
        return _rgba_arr(run_dir, f"{CASCADE_DBG_DIR}/{out_name}", CANVAS)

    def bbox_px(b):
        x0, y0, x1, y1 = _bbox_px(b, W, H)
        return max(0, x0), max(0, y0), min(W, x1), min(H, y1)

    def analyze_bbox(arr, lab, b):
        """bbox 处的连通域:返回 (命中标签集合, 是否孤立)。孤立=命中域的
        紧致边界不超出 bbox overflow_px 以上。"""
        x0, y0, x1, y1 = bbox_px(b)
        if x1 <= x0 or y1 <= y0:
            return set(), True
        region = lab[y0:y1, x0:x1]
        ids = set(int(v) for v in _np.unique(region)) - {0}
        if not ids:
            return set(), True
        isolated = True
        for k in ids:
            ys, xs = _np.nonzero(lab == k)
            over = max(x0 - xs.min(), y0 - ys.min(),
                       xs.max() + 1 - x1, ys.max() + 1 - y1)
            if over > overflow_px:
                isolated = False
        return ids, isolated

    # 各元素 home 层的原始 alpha:销账防抢跑用——真身还在家(home 层 bbox 处
    # 有实质像素)时,不许上层把恰好同框的别层像素(如压在元素上的文字标签)
    # 当成该元素切走,否则轮到 home 层时 item 已 resolved,真身全数掉进 temp
    home_raw_alpha = {}
    for _nm in ("text", "icon", "assets", "controls"):
        _f = layer_files.get(_nm)
        home_raw_alpha[_nm] = (
            _rgba_arr(run_dir, _f, CANVAS)[..., 3] > alpha_thr) if _f else None
    done_homes = set()

    # temp 残留画布(RGBA,上层在上)
    temp = _np.zeros((H, W, 4), dtype=_np.uint8)
    relations = 0
    steps_log = []

    for step_idx, (layer_name, kind) in enumerate(seq):
        n_assets0 = len(assets)
        n_rel0 = relations
        rescued_ids = []
        claimed_ids = []
        absorbed = 0
        lf = layer_files.get(layer_name)
        arr = (_rgba_arr(run_dir, lf, CANVAS) if lf
               else _np.zeros((H, W, 4), dtype=_np.uint8))
        lmask = arr[..., 3] > alpha_thr
        lab, _n = _label_components(lmask)

        # -- temp 残留处理(仅面板层,先于全裁:吸进来的随 panel 一起裁;
        #    元素层的 temp 下沉挪到本层裁切之后,见下) --
        if kind == "panel":
            tmask = temp[..., 3] > alpha_thr
            tlab, tn = _label_components(tmask)
            for k in range(1, tn + 1):
                comp = tlab == k
                area = int(comp.sum())
                if area < min_area:
                    continue
                # 面板层归属吸收:完全在某 panel 内 / 轮廓相近 → 并入该 panel
                ys, xs = _np.nonzero(comp)
                ty0, ty1 = ys.min(), ys.max() + 1
                tx0, tx1 = xs.min(), xs.max() + 1
                region = lab[ty0:ty1, tx0:tx1]
                cand = set(int(v) for v in _np.unique(region)) - {0}
                for pk in cand:
                    pmask = lab == pk
                    inter = int((comp & pmask).sum())
                    pys, pxs = _np.nonzero(pmask)
                    same_shape = (abs(int(pys.min()) - ty0) + abs(int(pxs.min()) - tx0)
                                  + abs(int(pys.max()) + 1 - ty1)
                                  + abs(int(pxs.max()) + 1 - tx1)) <= 4 * overflow_px
                    fully_inside = inter >= area * 0.9 or (
                        ty0 >= pys.min() and tx0 >= pxs.min()
                        and ty1 <= pys.max() + 1 and tx1 <= pxs.max() + 1)
                    if fully_inside or same_shape:
                        # 像素并入层图(panel 全裁时自然带上),自 temp 移除
                        over = temp.copy()
                        over[..., 3] = _np.where(comp, over[..., 3], 0)
                        base_img = Image.fromarray(arr)
                        base_img.alpha_composite(Image.fromarray(over))
                        arr = _np.asarray(base_img).copy()
                        lmask = arr[..., 3] > alpha_thr
                        lab, _n = _label_components(lmask)
                        temp[..., 3] = _np.where(comp, 0, temp[..., 3])
                        absorbed += 1
                        break

        # -- 销账检查(元素类,先于本层裁切) --
        pending = [it for it in items if not it["resolved"]
                   and it["home"] != layer_name]
        sam2_rescue = []
        for it in pending:
            # 防抢跑:home 层还没轮到且真身还在家(home 层 bbox 处有实质像素)
            # → 留给 home 层自己裁,别把同框的别层像素(文字标签等)认领走
            if it["home"] not in done_homes:
                ha = home_raw_alpha.get(it["home"])
                if ha is not None:
                    hx0, hy0, hx1, hy1 = bbox_px(it["bbox"])
                    if int(ha[hy0:hy1, hx0:hx1].sum()) >= min_area:
                        continue
            ids, isolated = analyze_bbox(arr, lab, it["bbox"])
            if not ids:
                continue
            if isolated:
                # lab 是本轮开头算的,前面救援可能已把同一片像素切走——
                # 与当前 alpha 求交,空了就当没找到(留给后续步骤/miss 找回),
                # 否则会落一个全透明素材还把 item 销账
                m = _np.isin(lab, list(ids)) & (arr[..., 3] > alpha_thr)
                if int(m.sum()) < min_area:
                    continue
                crop, px = _tight_cut(arr, m)
                if crop is not None:
                    save_asset(crop, px, cls=it["cls"], layer_name=layer_name,
                               method="cc_rescue", aid=it["id"])
                    arr[..., 3] = _np.where(m, 0, arr[..., 3])
                    it["resolved"] = True
                    it["note"] = f"rescued@{layer_name}"
                    rescued_ids.append(it["id"])
            else:
                sam2_rescue.append(it)
        if sam2_rescue:
            cut = sam2_batch(lf, [it["bbox"] for it in sam2_rescue],
                             f"_rescue_{layer_name}.png")
            for it in sam2_rescue:
                x0, y0, x1, y1 = bbox_px(it["bbox"])
                sub = cut[y0:y1, x0:x1]
                m = _np.zeros((H, W), dtype=bool)
                m[y0:y1, x0:x1] = sub[..., 3] > alpha_thr
                crop, px = _tight_cut(cut, m)
                if crop is None:
                    continue
                save_asset(crop, px, cls=it["cls"], layer_name=layer_name,
                           method="sam2_rescue", aid=it["id"])
                arr[..., 3] = _np.where(m, 0, arr[..., 3])
                it["resolved"] = True
                it["note"] = f"rescued_adhered@{layer_name}"
                rescued_ids.append(it["id"])
            lmask = arr[..., 3] > alpha_thr
            lab, _n = _label_components(lmask)

        # -- 本层裁切 --
        if kind == "element":
            home_items = [it for it in items
                          if it["home"] == layer_name and not it["resolved"]]
            if layer_name == "text":
                boxes = [it["bbox"] for it in home_items]
                if boxes and lf:
                    cut = sam2_batch(lf, boxes, "_textcut.png")
                    for it in home_items:
                        x0, y0, x1, y1 = bbox_px(it["bbox"])
                        m = _np.zeros((H, W), dtype=bool)
                        m[y0:y1, x0:x1] = cut[y0:y1, x0:x1][..., 3] > alpha_thr
                        crop, px = _tight_cut(cut, m)
                        if crop is None or int(m.sum()) < min_area:
                            continue
                        save_asset(crop, px, cls="text", layer_name="text",
                                   method="sam2_text", aid=it["id"])
                        arr[..., 3] = _np.where(m, 0, arr[..., 3])
                        it["resolved"] = True
            else:
                sam2_home = []
                for it in home_items:
                    ids, isolated = analyze_bbox(arr, lab, it["bbox"])
                    if not ids:
                        continue
                    if isolated:
                        # 同救援:lab 可能过期,与当前 alpha 求交防切空
                        m = _np.isin(lab, list(ids)) & (arr[..., 3] > alpha_thr)
                        if int(m.sum()) < min_area:
                            continue
                        crop, px = _tight_cut(arr, m)
                        if crop is None:
                            continue
                        save_asset(crop, px, cls=it["cls"],
                                   layer_name=layer_name, method="cc",
                                   aid=it["id"])
                        arr[..., 3] = _np.where(m, 0, arr[..., 3])
                        it["resolved"] = True
                    else:
                        sam2_home.append(it)
                if sam2_home and lf:
                    cut = sam2_batch(lf, [it["bbox"] for it in sam2_home],
                                     f"_adh_{layer_name}.png")
                    for it in sam2_home:
                        x0, y0, x1, y1 = bbox_px(it["bbox"])
                        m = _np.zeros((H, W), dtype=bool)
                        m[y0:y1, x0:x1] = cut[y0:y1, x0:x1][..., 3] > alpha_thr
                        crop, px = _tight_cut(cut, m)
                        if crop is None or int(m.sum()) < min_area:
                            continue
                        save_asset(crop, px, cls=it["cls"],
                                   layer_name=layer_name, method="sam2_adhered",
                                   aid=it["id"])
                        arr[..., 3] = _np.where(m, 0, arr[..., 3])
                        it["resolved"] = True
        else:
            # 面板层:连通域全裁(layered 权威)
            lmask = arr[..., 3] > alpha_thr
            lab, n = _label_components(lmask)
            for k in range(1, n + 1):
                m = lab == k
                if int(m.sum()) < min_area:
                    continue
                crop, px = _tight_cut(arr, m)
                if crop is None:
                    continue
                cls = "panel_f" if layer_name == "panel_f" else "panel"
                save_asset(crop, px, cls=cls, layer_name=layer_name, method="cc")
                arr[..., 3] = _np.where(m, 0, arr[..., 3])

        # -- temp 残留下沉(元素层,后于本层裁切):
        #    落在未解决 item 框上 → 以该 item 身份认领(串槽内容归位);
        #    落在已解决 item 框上 → 独立成材 + 叠压关系(标签压在元素上) --
        if kind == "element":
            tmask = temp[..., 3] > alpha_thr
            tlab, tn = _label_components(tmask)
            layer_boxes = sorted((it for it in items if it["home"] == layer_name),
                                 key=lambda x: x["resolved"])  # 未解决优先认领
            for k in range(1, tn + 1):
                comp = tlab == k
                area = int(comp.sum())
                if area < min_area:
                    continue
                for it in layer_boxes:
                    x0, y0, x1, y1 = bbox_px(it["bbox"])
                    inside = int(comp[y0:y1, x0:x1].sum())
                    if inside < area * land_ratio:
                        continue
                    crop, px = _tight_cut(temp, comp)
                    if crop is None:
                        break
                    if not it["resolved"]:
                        save_asset(crop, px, cls=it["cls"], layer_name=layer_name,
                                   method="temp_claim", aid=it["id"],
                                   origin_layer="temp")
                        it["resolved"] = True
                        it["note"] = f"temp_claim@{layer_name}"
                        claimed_ids.append(it["id"])
                    else:
                        save_asset(crop, px, cls="overlay",
                                   layer_name=layer_name, method="cc",
                                   aid=new_asset("overlay"),
                                   stacked_on=it["id"], origin_layer="temp")
                        relations += 1
                    temp[..., 3] = _np.where(comp, 0, temp[..., 3])
                    break

        done_homes.add(layer_name)

        # -- 结转:本层剩余并入 temp(temp 在上) --
        rest = Image.fromarray(arr)
        over = Image.fromarray(temp)
        rest.alpha_composite(over)
        temp = _np.asarray(rest).copy()

        # 过程留痕:该层处理完的 temp 快照 + 统计
        temp_file = f"{CASCADE_DBG_DIR}/_temp_{step_idx:02d}_{layer_name}.png"
        Image.fromarray(temp).save(run_dir / temp_file)
        layer_assets = assets[n_assets0:]
        steps_log.append({
            "layer": layer_name, "kind": kind,
            "cut": sum(1 for a in layer_assets
                       if a["cls"] not in ("overlay",)
                       and a["id"] not in rescued_ids
                       and a["id"] not in claimed_ids),
            "rescued": rescued_ids,
            "claimed": claimed_ids,
            "overlays": relations - n_rel0,
            "absorbed": absorbed,
            "assets": [a["id"] for a in layer_assets],
            "tempFile": temp_file,
            "tempCoverage": round(float((temp[..., 3] > alpha_thr).mean()), 4),
        })

    # ---- 终局 ----
    debris_file = f"{CASCADE_DIR}/debris.png"  # PSD 置顶碎屑层,属交付物
    Image.fromarray(temp).save(run_dir / debris_file)

    # miss 找回:差分判定 → bg 路(SAM2+批量 fill)/原图路(SAM2)
    lost = [it for it in items if not it["resolved"]]
    origin_arr = _rgba_arr(run_dir, "origin.png", CANVAS)
    bg_file = slot_layers.get("bg", {}).get("file")
    bg_arr = _rgba_arr(run_dir, bg_file, CANVAS) if bg_file else None
    bg_path_items, origin_path_items = [], []
    for it in lost:
        if bg_arr is None:
            origin_path_items.append(it)
            continue
        x0, y0, x1, y1 = bbox_px(it["bbox"])
        if x1 <= x0 or y1 <= y0:
            origin_path_items.append(it)
            continue
        d = float(_np.abs(
            bg_arr[y0:y1, x0:x1, :3].astype(_np.int16)
            - origin_arr[y0:y1, x0:x1, :3].astype(_np.int16)).mean())
        it["bgDiff"] = round(d, 1)
        if d < diff_bg:
            bg_path_items.append(it)
        elif d > diff_origin:
            origin_path_items.append(it)
        else:
            # 模糊带:YOLO 复检 bg 裁块
            x0e = max(0, x0 - (x1 - x0)); y0e = max(0, y0 - (y1 - y0))
            x1e = min(W, x1 + (x1 - x0)); y1e = min(H, y1 + (y1 - y0))
            probe = Image.fromarray(bg_arr[y0e:y1e, x0e:x1e]).convert("RGB")
            probe_name = f"{CASCADE_DBG_DIR}/_probe.png"
            probe.save(run_dir / probe_name)
            r = tasks.handle_yolo({
                "dir": str(run_dir), "image": probe_name,
                "model": payload.get("yolo_model", "game0804_p2"),
                "imgsz": 640, "conf": 0.15, "refine_bbox": False,
                "txt_output": f"{CASCADE_DBG_DIR}/_probe.txt"})
            found = any(
                YOLO_CLASSES[int(l.split()[0])] == it["cls"]
                for l in r.get("lines", []) if l.split())
            (bg_path_items if found else origin_path_items).append(it)

    if bg_path_items:
        cut = sam2_batch(bg_file, [it["bbox"] for it in bg_path_items],
                         "_bgcut.png")
        for it in bg_path_items:
            x0, y0, x1, y1 = bbox_px(it["bbox"])
            m = _np.zeros((H, W), dtype=bool)
            m[y0:y1, x0:x1] = cut[y0:y1, x0:x1][..., 3] > alpha_thr
            crop, px = _tight_cut(cut, m)
            if crop is None:
                origin_path_items.append(it)
                continue
            save_asset(crop, px, cls=it["cls"], layer_name="bg",
                       method="sam2", aid=it["id"], recovered="bg")
            it["resolved"] = True
            it["note"] = "recovered@bg"
        # 批量净化 bg:一次 flux fill,直接产出交付目录的 bg.png
        tasks.handle_flux_fill({
            "dir": str(run_dir), "image": bg_file,
            "mask_from": f"{CASCADE_DBG_DIR}/_bgcut.png",
            "output": f"{CASCADE_DIR}/bg.png",
            "prompt": payload.get(
                "fill_prompt",
                "Clean game UI background. Repair the masked regions with "
                "surrounding background texture only. No new elements."),
            "steps": payload.get("fill_steps", 30),
            "guidance": payload.get("fill_guidance", 30),
            "grow_mask": 8, "mask_blur": 4,
        })
        bg_final = f"{CASCADE_DIR}/bg.png"
    elif bg_file:
        # 不需净化也把 bg 复制进交付目录:素材+背景=完整交付物
        bg_final = f"{CASCADE_DIR}/bg.png"
        Image.open(run_dir / bg_file).convert("RGBA").save(run_dir / bg_final)
    else:
        bg_final = bg_file

    if origin_path_items:
        cut = sam2_batch("origin.png",
                         [it["bbox"] for it in origin_path_items], "_origincut.png")
        for it in origin_path_items:
            if it["resolved"]:
                continue
            x0, y0, x1, y1 = bbox_px(it["bbox"])
            m = _np.zeros((H, W), dtype=bool)
            m[y0:y1, x0:x1] = cut[y0:y1, x0:x1][..., 3] > alpha_thr
            crop, px = _tight_cut(cut, m)
            if crop is None or int(m.sum()) < min_area:
                it["note"] = "lost"
                continue
            save_asset(crop, px, cls=it["cls"], layer_name="origin",
                       method="sam2", aid=it["id"], recovered="origin")
            it["resolved"] = True
            it["note"] = "recovered@origin"

    # 软对账(panel 类)与重影审计
    panel_assets = [a for a in assets if a["cls"] in ("panel", "panel_f")]
    soft = []
    for it in panel_inv:
        matched = any(_iou(it["bbox"], a["bbox"]) >= 0.4 for a in panel_assets)
        if not matched:
            soft.append({"cls": it["cls"], "bbox": it["bbox"]})
    ghosts = []
    if bg_arr is not None:
        for a in assets:
            if a.get("recoveredFrom") == "bg" or a["cls"] in ("panel", "panel_f"):
                continue
            x0, y0, x1, y1 = bbox_px(a["bbox"])
            if x1 <= x0 or y1 <= y0:
                continue
            d = float(_np.abs(
                bg_arr[y0:y1, x0:x1, :3].astype(_np.int16)
                - origin_arr[y0:y1, x0:x1, :3].astype(_np.int16)).mean())
            if d < diff_bg:
                ghosts.append({"id": a["id"], "bgDiff": round(d, 1)})

    still_lost = [{"id": it["id"], "cls": it["cls"], "bbox": it["bbox"]}
                  for it in items if not it["resolved"]]
    manifest = {
        "zOrder": zmap,
        # 版本戳:级联重跑后素材文件名不变,前端靠它给图片 URL 换版防缓存混读
        "generatedAt": int(time.time()),
        "canvas": {"w": W, "h": H},
        "steps": steps_log,
        "background": {"file": bg_final, "zIndex": 0},
        "debris": debris_file,
        "assets": assets,
        "report": {
            "lost": still_lost,
            "panelSoftCheck": soft,
            "ghostSuspects": ghosts,
            "recoveredBg": sum(1 for a in assets if a.get("recoveredFrom") == "bg"),
            "recoveredOrigin": sum(1 for a in assets
                                   if a.get("recoveredFrom") == "origin"),
            "stackRelations": relations,
        },
    }
    (run_dir / "p2_cascade.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    by_layer = {}
    for a in assets:
        by_layer[a["sourceLayer"]] = by_layer.get(a["sourceLayer"], 0) + 1
    summary = {"count": len(assets), "byLayer": by_layer,
               "lost": len(still_lost), "ghosts": len(ghosts),
               "relations": relations, "file": "p2_cascade.json",
               "debris": debris_file, "background": bg_final}
    # 级联重跑 → 下游 PSD 摘要作废
    _mutate_state(run_dir, {"cascadeSummary": summary, "psdSummary": None})
    return summary


# ---------- 步骤 6:拼回 ----------

def handle_p2_recompose(payload: dict) -> dict:
    run_dir = _run_dir(payload)
    state = _load_state(run_dir)
    size = state.get("imageSize") or {}
    W, H = size.get("w"), size.get("h")
    if not W:
        raise RuntimeError("先跑 p2_sixslot")
    started = time.time()
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    slot_layers = {l["name"]: l for l in state.get("slotLayers", [])}
    if "bg" in slot_layers:
        with Image.open(run_dir / slot_layers["bg"]["file"]) as im:
            canvas.alpha_composite(im.convert("RGBA").resize((W, H), Image.LANCZOS))
    for z in state.get("panelzLayers", []):
        if z["name"] != "bg" and z["keep"]:
            with Image.open(run_dir / z["file"]) as im:
                canvas.alpha_composite(im.convert("RGBA").resize((W, H), Image.LANCZOS))
    for e in state.get("elements", []):
        if e.get("mergedInto"):
            continue
        ex = e.get("extract")
        bbox = (ex or {}).get("bbox") or e["bbox"]
        x0, y0, x1, y1 = _bbox_px(bbox, W, H)
        dw, dh = max(1, x1 - x0), max(1, y1 - y0)
        src_file = ex["file"] if ex else e["sourceFile"]
        with Image.open(run_dir / src_file) as im:
            asset = im.convert("RGBA")
        if ex is None or ex["method"] == "sam2_combined":
            sw, sh = asset.size
            sx0, sy0, sx1, sy1 = _bbox_px(bbox, sw, sh)
            asset = asset.crop((sx0, sy0, sx1, sy1))
        if asset.size != (dw, dh):
            asset = asset.resize((dw, dh), Image.LANCZOS)
        canvas.alpha_composite(asset, (x0, y0))
    out = "p2_recompose.png"
    canvas.save(run_dir / out)
    return {"file": out, "elapsed_sec": round(time.time() - started, 1)}


# ---------- 步骤 4:生成 PSD(p2_psd) ----------

PSD_DIR = "p2_psd"


def handle_p2_psd(payload: dict) -> dict:
    """p2_cascade.json + 交付素材 → 分组分层 PSD(落网盘 run 目录)。

    PSD 画布 = 原图尺寸。级联素材的像素/bbox 在"生成画布"帧(原图经
    1024 桶 + 16 对齐的拉伸帧,两轴各差 ~2%),归一化 bbox 与帧无关,
    这里统一折算回原图帧:按 bbox×原图尺寸定位,素材像素等比缩放——
    否则 PSD 叠原图会出现越往右下越大的线性漂移。

    图层序(底→顶)= 账本 zOrder 升序:bg → z 小→大 → controls → assets →
    panel_f → icon → text,碎屑图置顶。素材按 sourceLayer 进组,组内保持
    账本顺序(叠压 overlay 追加在被叠元素之后,即在其上方)。
    同时拍平出 preview.png(原图尺寸),与原图算平均误差/差异像素比,写回 state。
    """
    import numpy as _np
    from pytoshop.user import nested_layers as nl
    from pytoshop import enums
    run_dir = _run_dir(payload)
    man_path = run_dir / "p2_cascade.json"
    if not man_path.exists():
        raise RuntimeError("先跑 p2_cascade(需要 p2_cascade.json)")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    with Image.open(run_dir / "origin.png") as _oim:
        W, H = _oim.size  # PSD/preview 帧 = 原图
    started = time.time()

    out_dir = run_dir / PSD_DIR
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    def _img_layer(name, arr, top=0, left=0):
        return nl.Image(
            name=name, top=int(top), left=int(left),
            channels={0: _np.ascontiguousarray(arr[..., 0]),
                      1: _np.ascontiguousarray(arr[..., 1]),
                      2: _np.ascontiguousarray(arr[..., 2]),
                      -1: _np.ascontiguousarray(arr[..., 3])})

    def _load_rgba(rel):
        with Image.open(run_dir / rel) as im:
            return _np.asarray(im.convert("RGBA")).copy()

    # 素材按 sourceLayer 分组(账本顺序);同时拍平合成 preview
    zmap = manifest["zOrder"]
    by_layer: dict = {}
    for a in manifest["assets"]:
        by_layer.setdefault(a["sourceLayer"], []).append(a)

    preview = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    n_layers = 0
    groups_top_first = []

    bg_file = (manifest.get("background") or {}).get("file")
    if bg_file:
        preview.alpha_composite(
            Image.open(run_dir / bg_file).convert("RGBA").resize((W, H)))

    # 底→顶合成 preview;组列表按 pytoshop 约定倒序(列表首个=顶层)
    n_skipped = 0
    for lname in sorted(by_layer, key=lambda n: zmap.get(n, 0)):
        members = by_layer[lname]
        layers_top_first = []
        for a in members:  # 账本顺序=切取顺序,底→顶
            arr = _load_rgba(a["file"])
            if not arr[..., 3].any():
                n_skipped += 1  # 全透明素材,pytoshop 也会丢,直接跳过并如实上报
                continue
            x, y, w, h = a["bbox"]
            # 原图帧定位:用边缘取整而非「左上取整+尺寸取整」,避免右/下边 ±1 累积
            left = round((x - w / 2) * W)
            top = round((y - h / 2) * H)
            tw = max(1, round((x + w / 2) * W) - left)
            th = max(1, round((y + h / 2) * H) - top)
            if (arr.shape[1], arr.shape[0]) != (tw, th):
                arr = _np.asarray(Image.fromarray(arr).resize(
                    (tw, th), Image.LANCZOS)).copy()
            preview.alpha_composite(Image.fromarray(arr), (left, top))
            layers_top_first.insert(0, _img_layer(a["id"], arr, top, left))
            n_layers += 1
        groups_top_first.insert(0, nl.Group(name=lname, layers=layers_top_first))

    debris_file = manifest.get("debris")
    if debris_file and (run_dir / debris_file).exists():
        arr = _load_rgba(debris_file)
        if arr[..., 3].any():
            # 碎屑图是画布帧全幅,同样缩回原图帧
            if (arr.shape[1], arr.shape[0]) != (W, H):
                arr = _np.asarray(Image.fromarray(arr).resize(
                    (W, H), Image.LANCZOS)).copy()
            preview.alpha_composite(Image.fromarray(arr))
            groups_top_first.insert(0, _img_layer("debris", arr))
            n_layers += 1
    if bg_file:
        bg_arr = _load_rgba(bg_file)
        if (bg_arr.shape[1], bg_arr.shape[0]) != (W, H):
            bg_arr = _np.asarray(Image.fromarray(bg_arr).resize(
                (W, H), Image.LANCZOS)).copy()
        groups_top_first.append(_img_layer("bg", bg_arr))
        n_layers += 1

    # 实证:pytoshop 的 size 参数是 (width, height)
    psd = nl.nested_layers_to_psd(
        groups_top_first, color_mode=enums.ColorMode.rgb,
        size=(W, H), compression=enums.Compression.raw)
    # pytoshop 不拍平,内嵌合并预览手工塞入(第三方缩略图用;PS 自己渲染图层)
    try:
        flat = _np.asarray(preview.convert("RGB")).copy()
        for i in range(3):
            psd.image_data.channels[i] = _np.ascontiguousarray(flat[..., i])
    except Exception:
        pass  # 塞不进就算了,不影响 PSD 图层本身
    psd_file = f"{PSD_DIR}/result.psd"
    with open(run_dir / psd_file, "wb") as f:
        psd.write(f)

    preview_file = f"{PSD_DIR}/preview.png"
    preview.save(run_dir / preview_file)

    # 与原图误差(两者已同帧同尺寸):RGB 平均绝对差 + 差异像素占比
    with Image.open(run_dir / "origin.png") as im:
        origin = im.convert("RGB")
    comp = preview.convert("RGB")
    if comp.size != origin.size:  # 理论不会走到,兜底
        comp = comp.resize(origin.size, Image.LANCZOS)
    a = _np.asarray(origin).astype(_np.int16)
    b = _np.asarray(comp).astype(_np.int16)
    d = _np.abs(a - b)
    diff_mean = float(d.mean())
    diff_pct = float((d.max(axis=-1) > 12).mean() * 100)

    summary = {
        "file": psd_file, "preview": preview_file,
        "layers": n_layers, "groups": len(groups_top_first),
        "skipped": n_skipped,
        "sizeMB": round((run_dir / psd_file).stat().st_size / 1024 / 1024, 1),
        "diffMean": round(diff_mean, 2), "diffPct": round(diff_pct, 2),
        "generatedAt": int(time.time()),
        "elapsed": round(time.time() - started, 1),
    }
    _mutate_state(run_dir, {"psdSummary": summary})
    return summary
