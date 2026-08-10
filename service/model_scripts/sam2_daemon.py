"""SAM2 常驻抠图服务(跑在 /workspace/sam2-env,由 sam2d.sh 启动)。

模型只加载一次,监听 127.0.0.1:8189(仅本机,由 service 代理调用):
    GET  /health   就绪探针(模型加载完成后才返回 ok)
    POST /cutout   抠图,请求体:
        {
          "dir": "/workspace/servData/<run_id>",   图片所在目录,输出也写这里
          "image": "text_back.png",                输入图片名
          "output": "cutout.png",                  输出图片名
          "borders": [                             border 对象数组(结构化检测结果格式)
            {"bbox": [cx, cy, w, h],               归一化 0-1,中心点+宽高
             "positive_points": [[x, y], ...],     归一化 0-1,可空
             "negative_points": [[x, y], ...]},
            ...
          ],
          "padding_ratio": 0.02,                   每边按框尺寸比例外扩
          "min_padding": 1,                        每边最小外扩像素
          "mask_threshold": 0.5,                   SAM logit 阈值
          "feather_radius": 0,                     alpha 边缘高斯羽化半径,0 为硬边
          "multimask": false,                      多候选 mask 取最高分
          "crop_scale": 1.5,                       每个 icon 裁 bbox 的 N 倍切片单独分割,
                                                   小图形有效分辨率大幅提升;<=1 退回整图模式
          "refine": true,                          首轮 mask logits 作为 mask_input 再精化一轮
          "alt_image": "origin.png"                可选备选源图(尺寸必须与 image 一致)。
                                                   提供后,border 可带 "source" 字段:
                                                   "primary"(默认,用 image)/"alt"(用 alt_image)/
                                                   "auto"(两源各抠一次,按 SAM2 自评分择优,
                                                   平手取 primary;空 mask 直接出局)
        }

    响应含 "sources":逐 border 的 {index, source, scores} 择源记录(未启用双源时为空)。

抠图逻辑移植自 sam2/scripts/sam2_yolo_cutout.py:每个 border 作为一次独立的
box+points 提示,所有 mask 合并成一个 alpha 通道,输出透明 PNG(尺寸同输入图)。
"""
import json
import os
import threading
import traceback
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

HOST = "127.0.0.1"
PORT = int(os.environ.get("SAM2D_PORT", "8189"))
CHECKPOINT = os.environ.get("SAM2_CHECKPOINT",
                            "/workspace/sam2/checkpoints/sam2.1_hiera_large.pt")
MODEL_CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")

predictor = None
predict_lock = threading.Lock()


def load_model() -> None:
    global predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading SAM2 from {CHECKPOINT} on {device} ...", flush=True)
    model = build_sam2(MODEL_CONFIG, CHECKPOINT, device=device)
    model.eval()
    predictor = SAM2ImagePredictor(model)
    print("SAM2 ready", flush=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def fill_mask_holes(mask_bool: np.ndarray) -> np.ndarray:
    """封闭单个 mask 内部的孔洞:从外部 floodfill,不可达的非 mask 区域并入 mask。"""
    img = Image.fromarray((mask_bool.astype(np.uint8)) * 255)
    w, h = img.size
    padded = ImageOps.expand(img, border=1, fill=0)
    ImageDraw.floodfill(padded, (0, 0), 128)
    holes = padded.point(lambda v: 255 if v == 0 else 0).crop((1, 1, w + 1, h + 1))
    return mask_bool | (np.asarray(holes) > 0)


def cutout(req: dict) -> dict:
    run_dir = Path(req["dir"])
    image_path = run_dir / req["image"]
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    output_name = req.get("output") or "cutout.png"
    borders = req.get("borders") or []
    if not borders:
        raise ValueError("borders must be a non-empty array")
    padding_ratio = float(req.get("padding_ratio", 0.02))
    min_padding = float(req.get("min_padding", 1))
    mask_threshold = float(req.get("mask_threshold", 0.5))
    feather_radius = float(req.get("feather_radius", 0))
    multimask = bool(req.get("multimask", False))
    crop_scale = float(req.get("crop_scale", 1.5))
    refine = bool(req.get("refine", True))
    fill_holes = bool(req.get("fill_holes", True))
    # 尺寸分档规则:[{max_side/min_side(像素长边条件), 参数覆盖...}],命中第一条生效
    size_rules = req.get("size_rules") or []
    TUNABLE = ("padding_ratio", "min_padding", "mask_threshold",
               "feather_radius", "crop_scale")
    BOOL_TUNABLE = ("refine", "multimask", "fill_holes")

    def effective_params(border, side_px):
        eff = {"padding_ratio": padding_ratio, "min_padding": min_padding,
               "mask_threshold": mask_threshold, "feather_radius": feather_radius,
               "crop_scale": crop_scale, "refine": refine,
               "multimask": multimask, "fill_holes": fill_holes}
        for rule in size_rules:
            lo, hi = rule.get("min_side"), rule.get("max_side")
            if (lo is None or side_px >= float(lo)) and (hi is None or side_px <= float(hi)):
                for k in TUNABLE:
                    if rule.get(k) is not None:
                        eff[k] = float(rule[k])
                for k in BOOL_TUNABLE:
                    if rule.get(k) is not None:
                        eff[k] = bool(rule[k])
                break
        # border 内联覆盖,优先级最高(为未来逐 icon 定制留的通道)
        inline = border.get("params") or {}
        for k in TUNABLE:
            if inline.get(k) is not None:
                eff[k] = float(inline[k])
        for k in BOOL_TUNABLE:
            if inline.get(k) is not None:
                eff[k] = bool(inline[k])
        return eff

    with Image.open(image_path) as im:
        rgba = ImageOps.exif_transpose(im).convert("RGBA")
    arr = np.asarray(rgba)
    rgb = np.ascontiguousarray(arr[:, :, :3])
    orig_alpha = arr[:, :, 3]
    height, width = rgb.shape[:2]

    # 可选备选源图:与主图逐 border 择优(双源模式)
    alt_rgb = alt_alpha = None
    alt_name = req.get("alt_image")
    if alt_name:
        alt_path = run_dir / alt_name
        if not alt_path.is_file():
            raise FileNotFoundError(f"alt image not found: {alt_path}")
        with Image.open(alt_path) as im:
            alt_im = ImageOps.exif_transpose(im).convert("RGBA")
        if alt_im.size != (width, height):
            # 主图(去字图)经 16 对齐限像素缩放,与原图尺寸常不一致;
            # 归一化坐标与源图分辨率无关,重采样对齐即可
            alt_im = alt_im.resize((width, height), Image.Resampling.LANCZOS)
        alt_arr = np.asarray(alt_im)
        alt_rgb = np.ascontiguousarray(alt_arr[:, :, :3])
        alt_alpha = alt_arr[:, :, 3]

    merged_alpha = np.zeros((height, width), dtype=np.uint8)
    # 择源为 alt 的区域,输出像素也取自 alt 图
    override_mask = (np.zeros((height, width), dtype=bool)
                     if alt_rgb is not None else None)
    source_stats = []
    used = 0
    full_key = None  # 当前整图编码对应的源("primary"/"alt"),避免重复编码

    use_amp = torch.cuda.is_available()
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.inference_mode(), amp:
        for border_index, border in enumerate(borders):
            cx, cy, w, h = (float(v) for v in border["bbox"][:4])
            x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
            x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
            # 按未外扩的像素长边分档,取本 border 的生效参数
            eff = effective_params(border, max(x2 - x1, y2 - y1))
            use_crop = eff["crop_scale"] > 1.0
            predictor.mask_threshold = eff["mask_threshold"]
            pad_x = max((x2 - x1) * eff["padding_ratio"], eff["min_padding"])
            pad_y = max((y2 - y1) * eff["padding_ratio"], eff["min_padding"])
            x1 = clamp(x1 - pad_x, 0, width - 1)
            y1 = clamp(y1 - pad_y, 0, height - 1)
            x2 = clamp(x2 + pad_x, 0, width - 1)
            y2 = clamp(y2 + pad_y, 0, height - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            points, labels = [], []
            for p in border.get("positive_points") or []:
                points.append([float(p[0]) * width, float(p[1]) * height])
                labels.append(1)
            for p in border.get("negative_points") or []:
                points.append([float(p[0]) * width, float(p[1]) * height])
                labels.append(0)

            # 裁剪坐标与提示只由 bbox/points 决定,与源图无关,先算好供所有候选源共用
            if use_crop:
                # 以外扩后的框为中心裁 crop_scale 倍切片,小图形有效分辨率大幅提升
                bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
                half_w = (x2 - x1) * eff["crop_scale"] / 2
                half_h = (y2 - y1) * eff["crop_scale"] / 2
                ox1 = int(clamp(bcx - half_w, 0, width - 1))
                oy1 = int(clamp(bcy - half_h, 0, height - 1))
                ox2 = int(clamp(round(bcx + half_w), 1, width))
                oy2 = int(clamp(round(bcy + half_h), 1, height))
                box_arr = [x1 - ox1, y1 - oy1, x2 - ox1, y2 - oy1]
                # 点换算到切片坐标系,落在切片外的丢弃
                kept = [(px - ox1, py - oy1, lb) for (px, py), lb in zip(points, labels)
                        if ox1 <= px < ox2 and oy1 <= py < oy2]
                pts = [[px, py] for px, py, _ in kept]
                lbs = [lb for _, _, lb in kept]
            else:
                ox1, oy1 = 0, 0
                box_arr = [x1, y1, x2, y2]
                pts, lbs = points, labels

            point_coords = np.asarray(pts, dtype=np.float32) if pts else None
            point_labels = np.asarray(lbs, dtype=np.int32) if pts else None
            box = np.asarray(box_arr, dtype=np.float32)

            def predict_on(src_rgb):
                """在指定源图上跑一次完整预测(含 refine),返回 (mask_bool, 自评分)。"""
                nonlocal full_key
                if use_crop:
                    predictor.set_image(
                        np.ascontiguousarray(src_rgb[oy1:oy2, ox1:ox2]))
                    full_key = None
                masks, scores, low_res = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=eff["multimask"],
                    return_logits=False,
                )
                best = int(np.argmax(scores))
                mask, score = masks[best], float(scores[best])
                if eff["refine"]:
                    # 首轮低分辨率 logits 作为 mask_input,同一组提示再精化一轮
                    masks2, scores2, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box,
                        mask_input=low_res[best : best + 1],
                        multimask_output=False,
                        return_logits=False,
                    )
                    best2 = int(np.argmax(scores2))
                    mask, score = masks2[best2], float(scores2[best2])
                mask = mask.astype(bool)
                if not mask.any():
                    score = -1.0  # 空 mask 直接出局
                return mask, score

            # 候选源:未提供 alt_image 时一律主图;border.source 控制单选或双源择优
            border_source = str(border.get("source") or "primary")
            if alt_rgb is None or border_source == "primary":
                plan = [("primary", rgb)]
            elif border_source == "alt":
                plan = [("alt", alt_rgb)]
            else:  # auto:主图在前,平手取主图
                plan = [("primary", rgb), ("alt", alt_rgb)]

            chosen_src, mask, chosen_score = None, None, -2.0
            scores_by_src = {}
            for src_key, src_rgb in plan:
                if not use_crop and full_key != src_key:
                    predictor.set_image(src_rgb)
                    full_key = src_key
                cand_mask, cand_score = predict_on(src_rgb)
                scores_by_src[src_key] = round(cand_score, 4)
                if cand_score > chosen_score:
                    chosen_src, mask, chosen_score = src_key, cand_mask, cand_score
            if len(plan) > 1:
                source_stats.append({"index": border_index,
                                     "source": chosen_src,
                                     "scores": scores_by_src})

            if eff["fill_holes"]:
                # 每个 icon 的 mask 单独封孔,抠出的图层实心无镂空
                mask = fill_mask_holes(mask)
            if override_mask is not None and chosen_src == "alt":
                override_mask[oy1 : oy1 + mask.shape[0],
                              ox1 : ox1 + mask.shape[1]] |= mask
            # 羽化按 border 的生效参数单独做,再以最大值合并进总 alpha
            mask_u8 = mask.astype(np.uint8) * 255
            if eff["feather_radius"] > 0:
                mask_u8 = np.asarray(
                    Image.fromarray(mask_u8).filter(
                        ImageFilter.GaussianBlur(radius=eff["feather_radius"])
                    )
                )
            mh, mw = mask_u8.shape
            region = merged_alpha[oy1 : oy1 + mh, ox1 : ox1 + mw]
            np.maximum(region, mask_u8, out=region)
            used += 1
    predictor.reset_predictor()

    if not used:
        raise ValueError("all borders were empty or outside the image")

    # 择源为 alt 的区域,RGB 与透明度上限都跟随 alt 图;其余跟随主图。
    # 不给源图透明区域凭空加不透明度。
    out_rgb = rgb
    alpha_limit = orig_alpha
    if override_mask is not None and override_mask.any():
        out_rgb = rgb.copy()
        out_rgb[override_mask] = alt_rgb[override_mask]
        alpha_limit = orig_alpha.copy()
        alpha_limit[override_mask] = alt_alpha[override_mask]
    alpha = np.minimum(merged_alpha, alpha_limit)

    out = Image.fromarray(np.dstack((out_rgb, alpha)).astype(np.uint8))
    out_path = run_dir / output_name
    out.save(out_path, format="PNG")
    opaque = int((alpha > 0).sum())
    return {"output_path": str(out_path), "num_boxes": used,
            "size": [width, height], "opaque_pixels": opaque,
            "crop_scale": eff["crop_scale"] if use_crop else 0, "refine": eff["refine"],
            "sources": source_stats}


def refine_bboxes(req: dict) -> dict:
    """bbox 几何回投:每个 YOLO 框作为 box 提示跑一次分割,用 mask 的紧致外接框替换原框。

    治"检测框小了一截":box 只是软提示,mask 会长到元素的真实边界。
    护栏(任一不满足则保留原框):新旧框 IoU ≥ min_iou、面积膨胀 ≤ max_grow、
    面积收缩 ≥ min_shrink(防止 mask 只抓住元素一部分反而把框改小)。

    请求:{dir, image, borders:[{bbox:[cx,cy,w,h]}], crop_scale=1.5,
          mask_threshold=0.5, min_iou=0.3, max_grow=1.5, min_shrink=0.5}
    响应:{bboxes:[{bbox, refined, iou}]},与 borders 一一对应。
    """
    run_dir = Path(req["dir"])
    image_path = run_dir / req["image"]
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    borders = req.get("borders") or []
    crop_scale = float(req.get("crop_scale", 1.5))
    thr = float(req.get("mask_threshold", 0.5))
    min_iou = float(req.get("min_iou", 0.3))
    max_grow = float(req.get("max_grow", 1.5))
    min_shrink = float(req.get("min_shrink", 0.5))

    with Image.open(image_path) as im:
        arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    rgb = np.ascontiguousarray(arr)
    height, width = rgb.shape[:2]

    def rect_iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    out = []
    use_amp = torch.cuda.is_available()
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    predictor.mask_threshold = thr
    with torch.inference_mode(), amp:
        for border in borders:
            cx, cy, w, h = (float(v) for v in border["bbox"][:4])
            x1 = clamp((cx - w / 2) * width, 0, width - 1)
            y1 = clamp((cy - h / 2) * height, 0, height - 1)
            x2 = clamp((cx + w / 2) * width, 1, width)
            y2 = clamp((cy + h / 2) * height, 1, height)
            item = {"bbox": [cx, cy, w, h], "refined": False, "iou": 1.0}
            if x2 - x1 < 2 or y2 - y1 < 2:
                out.append(item)
                continue
            # 以框为中心裁 crop_scale 倍切片(至少 64px),小元素有效分辨率更高
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            half_w = max((x2 - x1) * crop_scale / 2, 32)
            half_h = max((y2 - y1) * crop_scale / 2, 32)
            ox1 = int(clamp(bcx - half_w, 0, width - 1))
            oy1 = int(clamp(bcy - half_h, 0, height - 1))
            ox2 = int(clamp(round(bcx + half_w), 1, width))
            oy2 = int(clamp(round(bcy + half_h), 1, height))
            predictor.set_image(np.ascontiguousarray(rgb[oy1:oy2, ox1:ox2]))
            masks, scores, _ = predictor.predict(
                box=np.asarray([x1 - ox1, y1 - oy1, x2 - ox1, y2 - oy1],
                               dtype=np.float32),
                multimask_output=False,
                return_logits=False,
            )
            mask = masks[int(np.argmax(scores))].astype(bool)
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                out.append(item)
                continue
            nx1, nx2 = ox1 + xs.min(), ox1 + xs.max() + 1
            ny1, ny2 = oy1 + ys.min(), oy1 + ys.max() + 1
            iou = rect_iou((x1, y1, x2, y2), (nx1, ny1, nx2, ny2))
            old_area = (x2 - x1) * (y2 - y1)
            new_area = (nx2 - nx1) * (ny2 - ny1)
            ratio = new_area / old_area if old_area > 0 else 0
            if iou >= min_iou and min_shrink <= ratio <= max_grow:
                item = {
                    "bbox": [((nx1 + nx2) / 2) / width, ((ny1 + ny2) / 2) / height,
                             (nx2 - nx1) / width, (ny2 - ny1) / height],
                    "refined": True,
                    "iou": round(float(iou), 4),
                }
            out.append(item)
    predictor.reset_predictor()
    return {"bboxes": out}


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
            self._send(200, {"ok": predictor is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/cutout", "/refine_bbox"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length))
            with predict_lock:
                result = (cutout(req) if self.path == "/cutout"
                          else refine_bboxes(req))
            self._send(200, result)
        except Exception:
            tb = traceback.format_exc()
            print(f"[cutout error]\n{tb}", flush=True)  # 落盘到 daemon 日志,便于事后排查
            self._send(500, {"error": tb[-3000:]})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    load_model()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"sam2 daemon listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
