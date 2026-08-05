#!/usr/bin/env bash
# 常驻 ComfyUI:只监听本机 127.0.0.1:8188,外部不可达,由 service 代理调用。
# 已在跑则直接退出,可安全重复执行(run.sh 每次启动都会调一次)。
set -euo pipefail

COMFY_ROOT="/workspace/ComfyUI"
COMFY_PYTHON="/workspace/venvs/comfyui/bin/python"
LOG="/workspace/servData/_logs/comfyui.log"

if pgrep -f "[C]omfyUI/main.py" > /dev/null; then
    echo "[comfyui.sh] already running"
    exit 0
fi

mkdir -p "$(dirname "$LOG")"
cd "$COMFY_ROOT"
nohup "$COMFY_PYTHON" "$COMFY_ROOT/main.py" \
    --listen 127.0.0.1 \
    --port 8188 \
    > "$LOG" 2>&1 &
echo "[comfyui.sh] ComfyUI starting (pid $!), log: $LOG"
