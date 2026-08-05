"""run 目录管理。

每上传一张原图创建一个高精度时间戳 run_id 目录:
    /workspace/servData/20260723_153045_123456/
        origin.png          上传的原图(统一转成 PNG)
        meta.json           run 元信息
        omnipsd/            后续各任务的输出写回同一目录
        yolo/
        sam2/
"""
import io
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import DATA_ROOT

# meta.json 的读改写必须互斥:并发提交任务时无锁竞写会把文件写坏
_meta_lock = threading.Lock()


def new_run_id() -> str:
    """微秒级时间戳作为 run_id,例如 20260723_153045_123456。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def create_run(original_filename: str, image_bytes: bytes, ext: str) -> dict:
    """创建 run 目录,原图统一存成 origin.png,返回 meta 信息。

    图片格式非法时抛 ValueError。
    """
    # 先校验/转码,失败就不建目录
    if ext == ".png":
        png_bytes = image_bytes
        Image.open(io.BytesIO(image_bytes)).verify()
    else:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if ("A" in img.mode or img.mode == "P") else "RGB")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        png_bytes = buf.getvalue()

    run_id = new_run_id()
    run_dir = DATA_ROOT / run_id
    # 同一微秒内碰撞的兜底(几乎不会发生)
    while run_dir.exists():
        run_id = new_run_id()
        run_dir = DATA_ROOT / run_id
    run_dir.mkdir(parents=True)

    origin_path = run_dir / "origin.png"
    origin_path.write_bytes(png_bytes)

    meta = {
        "run_id": run_id,
        "original_filename": original_filename,
        "origin_path": str(origin_path),
        "created_at": time.time(),
        "tasks": [],  # 后续任务追加记录
    }
    write_meta(run_id, meta)
    return meta


def get_run_dir(run_id: str) -> Path:
    """按 run_id 取目录,不存在或路径非法时抛 ValueError。"""
    # 防目录穿越:run_id 只允许是 DATA_ROOT 的直接子目录名
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        raise ValueError(f"illegal run_id: {run_id!r}")
    run_dir = DATA_ROOT / run_id
    if not run_dir.is_dir():
        raise ValueError(f"run_id not found: {run_id}")
    return run_dir


def read_meta(run_id: str) -> dict:
    text = (get_run_dir(run_id) / "meta.json").read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 历史并发 bug 可能留下"合法 JSON + 尾部残余"的损坏文件,取第一个完整对象自愈
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj


def write_meta(run_id: str, meta: dict) -> None:
    path = DATA_ROOT / run_id / "meta.json"
    # 先写临时文件再原子替换,写一半不会留下损坏的 meta.json
    tmp = path.with_name("meta.json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_task_record(run_id: str, record: dict) -> None:
    """把一次任务的记录追加进 run 的 meta.json(读改写全程持锁)。"""
    with _meta_lock:
        meta = read_meta(run_id)
        meta.setdefault("tasks", []).append(record)
        write_meta(run_id, meta)
