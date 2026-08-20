#!/usr/bin/env bash
# 常驻 Qwen-Image-Layered 分层 daemon:只监听 127.0.0.1:8195,由 service 代理调用。
# v2:基座 + six_slot/panelz 双 adapter 热切换;布局 B(双卡)/C(单卡≥80G)启动。
# 目标卡 ≥60G 用 bf16(~40G 常驻);小卡自动降 fp8 载入量化(PEFT over torchao,实验)。
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

# 按目标卡显存选精度(可用 QWEN_LAYERED_QUANT 预设覆盖)
if [ -z "${QWEN_LAYERED_QUANT:-}" ]; then
    GPU_IDX="${DETECT_GPU:-0}"
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_IDX" 2>/dev/null || echo 0)
    if [ "${VRAM_MB:-0}" -ge 60000 ]; then
        export QWEN_LAYERED_QUANT=bf16
    else
        export QWEN_LAYERED_QUANT=fp8
        echo "[qwenlayerd.sh] 目标卡 ${VRAM_MB}MiB <60G,降级 fp8(实验)"
    fi
fi

mkdir -p "$(dirname "$LOG")"
nohup "$QWEN_PYTHON" "$DAEMON" > "$LOG" 2>&1 &
echo "[qwenlayerd.sh] qwen layered daemon starting (pid $!), log: $LOG"
