#!/usr/bin/env python3
"""批量自动执行 /pipeline2 管线(纯 stdlib,无第三方依赖)。

对每张图:上传 → 原图YOLO ∥ 六槽分层(双泳道并行)→ panelz → 汇总审核(VL)
→ 级联切取 → 生成PSD → 下载产物(result.psd / preview.png / p2_cascade.json)。

用法:
  python3 scripts/run_pipeline2.py 图1.png 图2.jpg 某目录/ \\
      [--target https://xxx-8888.proxy.runpod.net] [--api-key sk-or-...] \\
      [--out p2_out] [--seed 42] [--resolution 1024] [--effort high]

  --target  缺省读 ui/node_modules/.cache/runpod-target.txt(前端切过代理即有)
  --api-key 缺省读环境变量 OPENROUTER_API_KEY
  参数缺省全部取 ui/src/config/pipeline2Defaults.json,
  VL 提示词取 ui/src/config/pipeline2InventoryPrompt.md(与前端同源)。

批量语义:逐图串行(同图内 ⓪⁺ 与 ① 并行);单图失败不中断整批,
最后打印汇总表,任一失败则退出码 1。
"""
import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
# RunPod proxy 前面的 Cloudflare 会 403 掉 Python-urllib 的默认 UA
UA = {"User-Agent": "pipeline2-batch/1.0"}


# ---------- HTTP ----------

def http_json(target: str, path: str, body: dict = None, timeout: int = 60) -> dict:
    url = target.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    headers = dict(UA)
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_upload(target: str, path: str, file_path: Path, timeout: int = 300) -> dict:
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    head = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{file_path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n").encode()
    body = head + file_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        target.rstrip("/") + path, data=body,
        headers={**UA,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_download(target: str, path: str, dest: Path, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(target.rstrip("/") + path, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dest.write_bytes(resp.read())


# ---------- 任务 ----------

def submit(target: str, task_type: str, run_id: str, params: dict) -> str:
    r = http_json(target, "/tasks",
                  {"type": task_type, "run_id": run_id, "params": params})
    return r["task_id"]


def wait_task(target: str, task_id: str, label: str,
              poll: float = 3.0, timeout: float = 1800.0) -> dict:
    t0 = time.time()
    while True:
        r = http_json(target, f"/tasks/{task_id}")
        st = r.get("status")
        if st == "succeeded":
            return r.get("result") or {}
        if st == "failed":
            err = (r.get("error") or "").strip().splitlines()
            raise RuntimeError(f"{label} 失败: {err[-1] if err else '未知错误'}")
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{label} 超时(>{timeout:.0f}s,task={task_id})")
        time.sleep(poll)


def run_and_wait(target: str, task_type: str, run_id: str, params: dict,
                 label: str, timeout: float = 1800.0) -> dict:
    tid = submit(target, task_type, run_id, params)
    t0 = time.time()
    result = wait_task(target, tid, label, timeout=timeout)
    print(f"    {label}: 完成({time.time() - t0:.0f}s)")
    return result


# ---------- 单图管线 ----------

def process_image(img: Path, target: str, cfg: dict, out_root: Path) -> dict:
    d = cfg["defaults"]
    print(f"[{img.name}] 上传…")
    meta = http_upload(target, "/runs", img)
    run_id = meta["run_id"]
    print(f"    run_id = {run_id}")

    # ⓪⁺ 原图 YOLO 与 ① 六槽分层在服务端走不同泳道,先提交再一起等
    y = d["yolo"]
    six = d["sixSlot"]
    six_params = {"steps": six["steps"], "seed": cfg["seed"] or six["seed"],
                  "true_cfg": six["trueCfg"],
                  "resolution": cfg["resolution"] or six["resolution"]}
    t_detect = submit(target, "p2_detect", run_id,
                      {"yolo_model": y["model"], "imgsz": y["imgsz"],
                       "conf": y["conf"], "iou": y["iou"]})
    t_six = submit(target, "p2_sixslot", run_id, six_params)
    wait_task(target, t_detect, "⓪⁺ 原图YOLO")
    print("    ⓪⁺ 原图YOLO: 完成")
    wait_task(target, t_six, "① 六槽分层")
    print("    ① 六槽分层: 完成")

    pz = d["panelz"]
    run_and_wait(target, "p2_panelz", run_id,
                 {"layers": pz["layers"], "steps": pz["steps"],
                  "seed": cfg["seed"] or pz["seed"], "true_cfg": pz["trueCfg"],
                  "resolution": cfg["resolution"] or pz["resolution"]},
                 "①ᵇ panelz 分层")

    if not cfg["api_key"]:
        # 无 key:GPU 重活(⓪⁺①①ᵇ)已做完,②③④ 拿到 key 后从前端恢复或重跑脚本
        print("    (无 OpenRouter key,停在 ①ᵇ;②③④ 待跑)")
        return {"image": img.name, "runId": run_id, "ok": True,
                "note": "停在①ᵇ(无key)", "assets": None, "lost": None,
                "psdLayers": None, "psdMB": None,
                "diffMean": None, "diffPct": None, "out": ""}

    inv = run_and_wait(target, "p2_inventory", run_id,
                       {"api_key": cfg["api_key"], "model": d["gpt"]["model"],
                        "prompt": cfg["prompt"],
                        "effort": cfg["effort"] or d["gpt"].get("effort", "high"),
                        "speed": cfg["speed"] or d["gpt"].get("speed", "balanced"),
                        "dedup_iou": y["dedupIou"]},
                       "② 汇总审核(VL)")

    cas = run_and_wait(target, "p2_cascade", run_id, {}, "③ 级联切取")
    psd = run_and_wait(target, "p2_psd", run_id, {}, "④ 生成PSD")

    out_dir = out_root / f"{img.stem}_{run_id}"
    for rel, name in [(psd["file"], "result.psd"),
                      (psd["preview"], "preview.png"),
                      ("p2_cascade.json", "p2_cascade.json")]:
        http_download(target, f"/runs/{run_id}/files/{rel}", out_dir / name)
    print(f"    产物已下载 → {out_dir}/")

    summary = {"image": img.name, "runId": run_id, "ok": True,
               "inventory": inv.get("stats") or inv,
               "assets": cas.get("count"), "lost": cas.get("lost"),
               "psdLayers": psd.get("layers"), "psdMB": psd.get("sizeMB"),
               "diffMean": psd.get("diffMean"), "diffPct": psd.get("diffPct"),
               "out": str(out_dir)}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


# ---------- 入口 ----------

def collect_images(inputs: list) -> list:
    imgs = []
    for s in inputs:
        p = Path(s).expanduser()
        if p.is_dir():
            imgs += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS)
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
        else:
            print(f"跳过(不是图片/不存在): {s}", file=sys.stderr)
    return imgs


def default_target() -> str:
    f = REPO / "ui/node_modules/.cache/runpod-target.txt"
    return f.read_text().strip() if f.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="批量执行 /pipeline2 管线", allow_abbrev=False)
    ap.add_argument("images", nargs="+", help="图片文件或目录(可多个)")
    ap.add_argument("--target", default=default_target(),
                    help="服务地址(缺省读 runpod-target.txt)")
    ap.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""),
                    help="OpenRouter Key(缺省读 $OPENROUTER_API_KEY)")
    ap.add_argument("--out", default="p2_out", help="产物输出目录")
    ap.add_argument("--seed", type=int, default=0, help="覆盖生成 seed(0=用默认)")
    ap.add_argument("--resolution", type=int, default=0, help="覆盖分辨率桶(0=用默认)")
    ap.add_argument("--effort", default="", help="VL 推理强度(缺省用配置)")
    ap.add_argument("--speed", default="", help="VL 速度模式(缺省用配置)")
    args = ap.parse_args()

    if not args.target:
        print("错误:未指定 --target 且 runpod-target.txt 不存在", file=sys.stderr)
        return 2
    if not args.api_key:
        print("提示:无 OpenRouter key,只跑到 ①ᵇ panelz(②③④ 之后可续)",
              file=sys.stderr)
    imgs = collect_images(args.images)
    if not imgs:
        print("错误:没有可处理的图片", file=sys.stderr)
        return 2

    try:
        http_json(args.target, "/health")
    except (urllib.error.URLError, OSError) as e:
        print(f"错误:服务不可达 {args.target}: {e}", file=sys.stderr)
        return 2

    cfg = {
        "defaults": json.loads(
            (REPO / "ui/src/config/pipeline2Defaults.json").read_text()),
        "prompt": (REPO / "ui/src/config/pipeline2InventoryPrompt.md")
        .read_text(encoding="utf-8").strip(),
        "api_key": args.api_key,
        "seed": args.seed, "resolution": args.resolution,
        "effort": args.effort, "speed": args.speed,
    }
    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    for i, img in enumerate(imgs, 1):
        print(f"\n=== [{i}/{len(imgs)}] {img} ===")
        try:
            results.append(process_image(img, args.target, cfg, out_root))
        except Exception as e:  # 单图失败不中断整批
            print(f"    ✗ 失败: {e}", file=sys.stderr)
            results.append({"image": img.name, "ok": False, "error": str(e)})

    print(f"\n===== 批量完成({time.time() - t0:.0f}s)=====")
    for r in results:
        if r["ok"] and r.get("note"):
            print(f"  ◐ {r['image']}  run_id={r['runId']}  {r['note']}")
        elif r["ok"]:
            print(f"  ✓ {r['image']}  run_id={r['runId']}  素材{r['assets']} "
                  f"丢失{r['lost']} PSD {r['psdLayers']}层/{r['psdMB']}MB "
                  f"误差{r['diffMean']}/255({r['diffPct']}%)  → {r['out']}")
        else:
            print(f"  ✗ {r['image']}  {r['error']}")
    (out_root / "batch_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
