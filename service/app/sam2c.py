"""SAM2 常驻 daemon 的客户端(daemon 见 model_scripts/sam2_daemon.py)。

daemon 不在时自动执行 sam2d.sh 拉起并等待模型加载完成。
GPU 串行由 service 的 worker 队列保证:cutout() 阻塞到出图才返回。
"""
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import SAM2D_TIMEOUT, SAM2D_URL

SAM2D_START_SCRIPT = Path(__file__).resolve().parent.parent / "sam2d.sh"
SAM2D_START_TIMEOUT = 300  # 秒,含模型加载
SAM2D_LOG = Path("/workspace/servData/_logs/sam2d.log")


def _http(method: str, path: str, payload: dict = None, timeout: float = 30) -> dict:
    req = urllib.request.Request(
        f"{SAM2D_URL}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # daemon 把 traceback 放在错误响应体里,透传出来,否则只剩一个干瘪的 500
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
            body = json.loads(body).get("error", body)
        except Exception:
            pass
        raise RuntimeError(f"sam2 daemon HTTP {e.code}: {body[-2000:]}") from None


def is_up() -> bool:
    try:
        return _http("GET", "/health", timeout=5).get("ok") is True
    except Exception:
        return False


def ensure_up() -> None:
    """daemon 不在时自动拉起并等就绪(pod 重启后第一次任务会走到这里)。"""
    if is_up():
        return
    subprocess.run(["bash", str(SAM2D_START_SCRIPT)], check=False,
                   capture_output=True, timeout=30)
    deadline = time.time() + SAM2D_START_TIMEOUT
    while time.time() < deadline:
        if is_up():
            return
        time.sleep(3)
    tail = ""
    try:
        tail = SAM2D_LOG.read_text(encoding="utf-8", errors="replace")[-2000:]
    except OSError:
        pass
    raise RuntimeError(
        f"sam2 daemon failed to start within {SAM2D_START_TIMEOUT}s, log tail:\n{tail}"
    )


def refine_bboxes(request: dict) -> dict:
    """bbox 几何回投:每个框作为 box 提示跑 SAM2,用 mask 外接框替换原框。"""
    ensure_up()
    return _http("POST", "/refine_bbox", request, timeout=SAM2D_TIMEOUT)


def cutout(request: dict) -> dict:
    """提交抠图请求并阻塞到完成。失败抛 RuntimeError(带 daemon 侧 traceback)。"""
    ensure_up()
    try:
        return _http("POST", "/cutout", request, timeout=SAM2D_TIMEOUT)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"sam2 cutout failed:\n{detail[:3000]}")
