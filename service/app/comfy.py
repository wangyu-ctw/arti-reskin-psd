"""ComfyUI HTTP API 客户端。

ComfyUI 常驻在 127.0.0.1:8188(comfyui.sh 启动),模型只加载一次。
本模块只做薄封装:放输入图 -> 提交 workflow(API 格式 JSON)-> 轮询 history -> 定位输出文件。
GPU 串行由 service 的 worker 队列保证:handler 阻塞到 ComfyUI 出图才返回。
"""
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .config import COMFY_ROOT, COMFY_TIMEOUT, COMFY_URL

COMFY_START_SCRIPT = Path(__file__).resolve().parent.parent / "comfyui.sh"
COMFY_START_TIMEOUT = 240  # 秒,等 ComfyUI 起监听的上限
COMFY_LOG = Path("/workspace/servData/_logs/comfyui.log")


def _http(method: str, path: str, payload: dict = None, timeout: float = 30):
    req = urllib.request.Request(
        f"{COMFY_URL}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def is_up() -> bool:
    try:
        _http("GET", "/system_stats", timeout=5)
        return True
    except Exception:
        return False


def ensure_up() -> None:
    """ComfyUI 不在时自动拉起并等待就绪(pod 重启后第一次任务会走到这里)。"""
    if is_up():
        return
    subprocess.run(["bash", str(COMFY_START_SCRIPT)], check=False,
                   capture_output=True, timeout=30)
    deadline = time.time() + COMFY_START_TIMEOUT
    while time.time() < deadline:
        if is_up():
            return
        time.sleep(3)
    tail = ""
    try:
        tail = COMFY_LOG.read_text(encoding="utf-8", errors="replace")[-2000:]
    except OSError:
        pass
    raise RuntimeError(
        f"ComfyUI failed to start within {COMFY_START_TIMEOUT}s, log tail:\n{tail}"
    )


def place_input_image(src: Path, prefix: str = "") -> str:
    """把输入图拷进 ComfyUI 的 input 目录,返回 LoadImage 用的文件名。"""
    name = f"{prefix}{uuid.uuid4().hex[:8]}{src.suffix}"
    input_dir = COMFY_ROOT / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, input_dir / name)
    return name


def place_input_pil(img, prefix: str = "") -> str:
    """把内存中的 PIL 图片写进 ComfyUI 的 input 目录(如动态生成的 mask)。"""
    name = f"{prefix}{uuid.uuid4().hex[:8]}.png"
    input_dir = COMFY_ROOT / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    img.save(input_dir / name, format="PNG")
    return name


def run_workflow(workflow: dict, timeout: float = COMFY_TIMEOUT) -> dict:
    """提交 workflow 并阻塞到完成,返回 history 条目(含 outputs)。失败抛 RuntimeError。"""
    ensure_up()
    try:
        resp = _http("POST", "/prompt", {"prompt": workflow, "client_id": uuid.uuid4().hex})
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI rejected workflow: {e.read().decode(errors='replace')[:2000]}")
    prompt_id = resp["prompt_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        history = _http("GET", f"/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                messages = [m for m in status.get("messages", [])
                            if m and m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI execution error: "
                                   f"{json.dumps(messages, ensure_ascii=False)[:3000]}")
            if status.get("completed") or entry.get("outputs"):
                entry["prompt_id"] = prompt_id
                return entry
        time.sleep(1)
    raise RuntimeError(f"ComfyUI workflow timed out after {timeout}s (prompt_id={prompt_id})")


def output_image_paths(entry: dict) -> list:
    """从 history 条目里取出所有输出图片的绝对路径(SaveImage 节点的产物)。"""
    paths = []
    for node_output in entry.get("outputs", {}).values():
        for img in node_output.get("images", []):
            if img.get("type") != "output":
                continue
            p = COMFY_ROOT / "output" / img.get("subfolder", "") / img["filename"]
            if p.is_file():
                paths.append(p)
    return paths
