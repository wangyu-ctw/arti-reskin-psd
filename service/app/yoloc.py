"""YOLO 常驻 daemon 的客户端(daemon 见 model_scripts/yolo_daemon.py)。

daemon 不在时自动执行 yolod.sh 拉起并等待模型加载完成。
"""
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import YOLOD_TIMEOUT, YOLOD_URL

YOLOD_START_SCRIPT = Path(__file__).resolve().parent.parent / "yolod.sh"
YOLOD_START_TIMEOUT = 180  # 秒,含模型加载
YOLOD_LOG = Path("/workspace/servData/_logs/yolod.log")


def _http(method: str, path: str, payload: dict = None, timeout: float = 30) -> dict:
    req = urllib.request.Request(
        f"{YOLOD_URL}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def is_up() -> bool:
    try:
        return _http("GET", "/health", timeout=5).get("ok") is True
    except Exception:
        return False


def ensure_up() -> None:
    """daemon 不在时自动拉起并等就绪(pod 重启后第一次任务会走到这里)。"""
    if is_up():
        return
    subprocess.run(["bash", str(YOLOD_START_SCRIPT)], check=False,
                   capture_output=True, timeout=30)
    deadline = time.time() + YOLOD_START_TIMEOUT
    while time.time() < deadline:
        if is_up():
            return
        time.sleep(3)
    tail = ""
    try:
        tail = YOLOD_LOG.read_text(encoding="utf-8", errors="replace")[-2000:]
    except OSError:
        pass
    raise RuntimeError(
        f"yolo daemon failed to start within {YOLOD_START_TIMEOUT}s, log tail:\n{tail}"
    )


def detect(request: dict) -> dict:
    """提交检测请求并阻塞到完成。失败抛 RuntimeError(带 daemon 侧 traceback)。"""
    ensure_up()
    try:
        return _http("POST", "/detect", request, timeout=YOLOD_TIMEOUT)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"yolo detect failed:\n{detail[:3000]}")
