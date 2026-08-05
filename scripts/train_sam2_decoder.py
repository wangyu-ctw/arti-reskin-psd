"""SAM2 mask decoder 微调(在 pod 的 /workspace/sam2-env 里运行)。

数据:prepare_training_data.py 的 sam2 产出(annotations.jsonl + 逐实例图层级 mask)。
方案:冻结 Hiera 图像编码器,只训 prompt encoder + mask decoder;
提示在线合成,分布对齐推理端(sam2_daemon):
  - box:GT bbox 外扩 0.5%~4% + 每边 ±2% 抖动(对应推理端 padding_ratio + 检测框误差)
  - 正点:mask 腐蚀后内部随机 1~3 个(对应"落在图形实体最粗壮处")
  - 负点:70% 采自 box 内、mask 外(圆角/菱形空隙天然命中),30% 采自框外紧邻环带
  - 小元素(长边<100px)按推理端 crop_scale 逻辑裁局部块再喂,其余整图
验证:ansatsu 游戏留出,固定种子合成提示,报逐类 IoU。
损失:dice + BCE + 0.05×IoU 头 MSE。输出 checkpoint 与官方格式兼容
({"model": state_dict}),SAM2_CHECKPOINT 直接指过去即可上线。

用法:
  /workspace/sam2-env/bin/python train_sam2_decoder.py \
      [--steps 8000] [--batch 4] [--lr 2e-5] [--eval-every 1000]
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DATA = Path("/workspace/inputs/train_data_20260804/sam2")
CKPT = "/workspace/sam2/checkpoints/sam2.1_hiera_large.pt"
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
VAL_PREFIX = "ansatsu_"
CROP_MAX_SIDE = 100          # 长边小于此值的实例走 crop 切片(对齐推理)
_image_cache: dict = {}


def load_image(rel: str) -> np.ndarray:
    if rel not in _image_cache:
        _image_cache[rel] = np.asarray(Image.open(DATA / rel).convert("RGB"))
    return _image_cache[rel]


def load_mask(rel: str) -> np.ndarray:
    return np.asarray(Image.open(DATA / rel).convert("L")) > 127


def make_example(rec: dict, rng: random.Random):
    """返回 (crop_rgb, crop_mask, box_xyxy, pos_pts, neg_pts),不合格返回 None。"""
    img = load_image(rec["image"])
    mask = load_mask(rec["mask"])
    ih, iw = mask.shape
    cx, cy, w, h = rec["bbox"]
    x1, y1 = (cx - w / 2) * iw, (cy - h / 2) * ih
    x2, y2 = (cx + w / 2) * iw, (cy + h / 2) * ih
    side = max(x2 - x1, y2 - y1)
    if side < 4:
        return None

    # 小元素裁局部块(推理端 crop_scale 1.3~2.2 的等效分布)
    if side < CROP_MAX_SIDE:
        scale = rng.uniform(1.3, 2.2)
        bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w = max((x2 - x1), 32) * scale / 2
        half_h = max((y2 - y1), 32) * scale / 2
        ox1 = int(max(bcx - half_w, 0)); oy1 = int(max(bcy - half_h, 0))
        ox2 = int(min(bcx + half_w, iw)); oy2 = int(min(bcy + half_h, ih))
        img = img[oy1:oy2, ox1:ox2]
        mask = mask[oy1:oy2, ox1:ox2]
        x1, y1, x2, y2 = x1 - ox1, y1 - oy1, x2 - ox1, y2 - oy1
    if not mask.any() or img.shape[0] < 8 or img.shape[1] < 8:
        return None
    ch, cw = mask.shape

    # box:外扩 + 抖动
    pad = side * rng.uniform(0.005, 0.04)
    jit = side * 0.02
    bx1 = max(0.0, x1 - pad + rng.uniform(-jit, jit))
    by1 = max(0.0, y1 - pad + rng.uniform(-jit, jit))
    bx2 = min(float(cw), x2 + pad + rng.uniform(-jit, jit))
    by2 = min(float(ch), y2 + pad + rng.uniform(-jit, jit))
    if bx2 - bx1 < 2 or by2 - by1 < 2:
        return None

    # 正点:腐蚀后的 mask 内部(避免贴边),腐蚀空了退回原 mask
    eroded = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 127
    pool = eroded if eroded.any() else mask
    ys, xs = np.nonzero(pool)
    n_pos = rng.randint(1, 3)
    idx = [rng.randrange(len(ys)) for _ in range(n_pos)]
    pos = [(float(xs[i]), float(ys[i])) for i in idx]

    # 负点:box 内 mask 外优先,补框外环带
    neg = []
    n_neg = rng.randint(3, 6)
    for _ in range(n_neg * 8):
        if len(neg) >= n_neg:
            break
        if rng.random() < 0.7:
            px = rng.uniform(bx1, bx2 - 1); py = rng.uniform(by1, by2 - 1)
        else:
            ring = side * rng.uniform(0.03, 0.15)
            px = rng.uniform(max(0, bx1 - ring), min(cw - 1, bx2 + ring))
            py = rng.choice([rng.uniform(max(0, by1 - ring), by1),
                             rng.uniform(by2, min(ch - 1, by2 + ring))]) \
                if rng.random() < 0.5 else rng.uniform(max(0, by1 - ring), min(ch - 1, by2 + ring))
        xi, yi = int(px), int(py)
        if 0 <= yi < ch and 0 <= xi < cw and not mask[yi, xi]:
            neg.append((float(px), float(py)))
    return img, mask, (bx1, by1, bx2, by2), pos, neg


def forward_instance(predictor, i, box, pos, neg, train: bool):
    """对 batch 里第 i 张图跑 prompt encoder + mask decoder,返回 (logits, iou_pred)。"""
    model = predictor.model
    device = predictor.device
    coords = [[box[0], box[1]], [box[2], box[3]]] + [list(p) for p in pos] + [list(p) for p in neg]
    labels = [2, 3] + [1] * len(pos) + [0] * len(neg)
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.int32, device=device)
    unnorm = predictor._transforms.transform_coords(
        coords_t, normalize=True, orig_hw=predictor._orig_hw[i])
    concat = (unnorm[None, ...], labels_t[None, ...])

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        sparse, dense = model.sam_prompt_encoder(points=concat, boxes=None, masks=None)
        low_res, iou_pred, _, _ = model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][i].unsqueeze(0),
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[f[i].unsqueeze(0) for f in predictor._features["high_res_feats"]],
        )
        logits = predictor._transforms.postprocess_masks(
            low_res, predictor._orig_hw[i])[0, 0]  # (H, W)
    return logits, iou_pred[0, 0]


def seg_loss(logits, gt_mask, iou_pred):
    gt = torch.from_numpy(gt_mask).to(logits.device, torch.float32)
    prob = torch.sigmoid(logits)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt)
    inter = (prob * gt).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + gt.sum() + 1)
    with torch.no_grad():
        pred_bin = logits > 0
        gt_b = gt > 0.5
        iou = ((pred_bin & gt_b).sum() / ((pred_bin | gt_b).sum() + 1)).float()
    score = torch.nn.functional.mse_loss(iou_pred.float(), iou)
    return bce + dice + 0.05 * score, float(iou)


@torch.no_grad()
def evaluate(predictor, val_recs, batch: int) -> dict:
    ious = defaultdict(list)
    for start in range(0, len(val_recs), batch):
        chunk = val_recs[start:start + batch]
        exs, metas = [], []
        for k, rec in enumerate(chunk):
            ex = make_example(rec, random.Random(9000 + start + k))
            if ex:
                exs.append(ex)
                metas.append(rec["class"])
        if not exs:
            continue
        predictor.set_image_batch([e[0] for e in exs])
        for i, (ex, cls) in enumerate(zip(exs, metas)):
            logits, _ = forward_instance(predictor, i, ex[2], ex[3], ex[4], train=False)
            pred = (logits > 0).cpu().numpy()
            gt = ex[1]
            iou = ((pred & gt).sum()) / ((pred | gt).sum() + 1e-6)
            ious[cls].append(iou)
            ious["_all"].append(iou)
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(ious.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--val-cap", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("/workspace/outputs/sam2_train_20260805"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(DATA / "annotations.jsonl", encoding="utf-8")]
    train_recs = [r for r in recs if not Path(r["image"]).name.startswith(VAL_PREFIX)]
    val_recs = [r for r in recs if Path(r["image"]).name.startswith(VAL_PREFIX)]
    rng_val = random.Random(7)
    rng_val.shuffle(val_recs)
    val_recs = val_recs[: args.val_cap]
    print(f"train {len(train_recs)} / val {len(val_recs)} instances", flush=True)

    model = build_sam2(CFG, CKPT, device="cuda")
    predictor = SAM2ImagePredictor(model)
    for p in model.parameters():
        p.requires_grad = False
    trainable = list(model.sam_mask_decoder.parameters()) + \
        list(model.sam_prompt_encoder.parameters())
    for p in trainable:
        p.requires_grad = True
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    base = evaluate(predictor, val_recs, args.batch)
    print(f"[step 0 baseline] {base}", flush=True)

    rng = random.Random(42)
    t0 = time.time()
    running = []
    for step in range(1, args.steps + 1):
        exs = []
        while len(exs) < args.batch:
            ex = make_example(rng.choice(train_recs), rng)
            if ex:
                exs.append(ex)
        predictor.set_image_batch([e[0] for e in exs])
        opt.zero_grad(set_to_none=True)
        total = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i, ex in enumerate(exs):
                logits, iou_pred = forward_instance(predictor, i, ex[2], ex[3], ex[4], train=True)
                loss, iou = seg_loss(logits, ex[1], iou_pred)
                (loss / args.batch).backward()
                total += float(loss)
                running.append(iou)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 50 == 0:
            speed = step / (time.time() - t0)
            print(f"step {step}/{args.steps} loss {total/args.batch:.4f} "
                  f"train-iou(近200) {np.mean(running[-200:]):.4f} {speed:.2f} it/s",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(predictor, val_recs, args.batch)
            print(f"[step {step} val] {metrics}", flush=True)
            ckpt_path = args.out / f"step-{step}.pt"
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"saved {ckpt_path}", flush=True)
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
