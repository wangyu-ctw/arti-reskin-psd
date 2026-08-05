#!/usr/bin/env bash
# 启动脚本: bash /workspace/service/run.sh
# 单进程 uvicorn(--workers 1),保证进程内 FIFO 队列全局唯一,GPU 严格串行。
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SERVICE_DIR"

DATA_ROOT="${SERV_DATA_ROOT:-/workspace/servData}"
mkdir -p "$DATA_ROOT/_logs"

# 依赖装进 /workspace 上的 venv:系统 Python 3.12 受 PEP 668 保护不让直接 pip,
# 且 venv 在网络卷上,pod 重启后依然存在,不用重装
VENV="$SERVICE_DIR/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "[run.sh] creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi
MARKER="$VENV/.deps_installed"
if [ ! -f "$MARKER" ] || [ "requirements.txt" -nt "$MARKER" ]; then
    echo "[run.sh] installing dependencies..."
    "$VENV/bin/pip" install -r requirements.txt
    touch "$MARKER"
fi

# 常驻 ComfyUI(text_back 等模型任务的推理后端,模型只加载一次)
bash "$SERVICE_DIR/comfyui.sh" || echo "[run.sh] warn: ComfyUI start failed, text_back will not work"

# 常驻 SAM2 抠图 daemon
bash "$SERVICE_DIR/sam2d.sh" || echo "[run.sh] warn: sam2 daemon start failed, sam2 will not work"

# 常驻 YOLO 检测 daemon
bash "$SERVICE_DIR/yolod.sh" || echo "[run.sh] warn: yolo daemon start failed, yolo will not work"

# RunPod 模板默认在 8888 起 JupyterLab,和本服务冲突,启动前先停掉
if pgrep -f "[j]upyter-lab" > /dev/null; then
    echo "[run.sh] stopping jupyter-lab (it occupies port 8888)..."
    pkill -f "[j]upyter-lab" || true
    sleep 2
fi

echo "[run.sh] starting service on 0.0.0.0:8888, data root: $DATA_ROOT"
exec "$VENV/bin/python" -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8888 \
    --workers 1 \
    --no-access-log
