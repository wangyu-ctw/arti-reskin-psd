#!/usr/bin/env bash
# 常驻 Qwen-Image-Layered 分层 daemon:只监听 127.0.0.1:8195,由 service 代理调用。
# 仅双卡布局启动(run.sh 判断);fp8 载入量化 ~20G 常驻。
set -euo pipefail

QWEN_PYTHON="/workspace/venvs/comfyui/bin/python"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
DAEMON="/workspace/service/model_scripts/qwen_layered_daemon.py"
LOG="/workspace/servData/_logs/qwenlayerd.log"

if pgrep -f "[q]wen_layered_daemon.py" > /dev/null; then
    echo "[qwenlayerd.sh] already running"
    exit 0
fi

# 双卡布局:run.sh 设 DETECT_GPU 时钉到指定卡
if [ -n "${DETECT_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DETECT_GPU"
fi

mkdir -p "$(dirname "$LOG")"
nohup "$QWEN_PYTHON" "$DAEMON" > "$LOG" 2>&1 &
echo "[qwenlayerd.sh] qwen layered daemon starting (pid $!), log: $LOG"
