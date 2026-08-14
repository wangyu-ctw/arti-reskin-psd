#!/usr/bin/env bash
# 启动脚本: bash /workspace/service/run.sh
# 单进程 uvicorn(--workers 1),保证进程内 FIFO 队列全局唯一,GPU 严格串行。
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SERVICE_DIR"

DATA_ROOT="${SERV_DATA_ROOT:-/workspace/servData}"
mkdir -p "$DATA_ROOT/_logs"

# 依赖装进 /workspace 上的 venv(网络卷常驻,pod 重启不丢)。
# 按 Python 小版本分目录:不同 pod 模板的 python3 版本可能不同(3.11/3.12),
# venv 跨版本会坏(No module named xxx),各版本各建一个互不干扰
PYV=$(python3 -c 'import sys; print(f"py{sys.version_info[0]}{sys.version_info[1]}")')
VENV="$SERVICE_DIR/.venv-$PYV"
if [ ! -x "$VENV/bin/python" ]; then
    echo "[run.sh] creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi
MARKER="$VENV/.deps_installed"
if [ ! -f "$MARKER" ] || [ "requirements.txt" -nt "$MARKER" ] \
   || ! "$VENV/bin/python" -c "import uvicorn" 2>/dev/null; then
    echo "[run.sh] installing dependencies..."
    "$VENV/bin/pip" install -r requirements.txt
    touch "$MARKER"
fi

# 按 GPU 数选布局(GPU_PLAN.md):
#   1 卡 = 布局 A:全部落 GPU0,单泳道 FIFO(现状)
#   ≥2 卡 = 布局 B:ComfyUI/FLUX 独占 GPU0;SAM2/YOLO 钉到 GPU1;service 双泳道并行
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
if [ "$N_GPU" -ge 2 ]; then
    echo "[run.sh] $N_GPU GPUs detected -> 双卡布局 B(Comfy@GPU0,SAM2/YOLO@GPU1,双泳道)"
    export DETECT_GPU=1
    export SERVICE_LANE_MODE=dual
else
    echo "[run.sh] single GPU -> 布局 A(单泳道)"
    export SERVICE_LANE_MODE=single
fi

# 常驻 ComfyUI(text_back 等模型任务的推理后端,模型只加载一次;双卡时独占 GPU0)
CUDA_VISIBLE_DEVICES=0 bash "$SERVICE_DIR/comfyui.sh" || echo "[run.sh] warn: ComfyUI start failed, text_back will not work"

# 常驻 SAM2 抠图 daemon(双卡时 DETECT_GPU 钉到 GPU1)
bash "$SERVICE_DIR/sam2d.sh" || echo "[run.sh] warn: sam2 daemon start failed, sam2 will not work"

# 常驻 YOLO 检测 daemon(同上)
bash "$SERVICE_DIR/yolod.sh" || echo "[run.sh] warn: yolo daemon start failed, yolo will not work"

# 常驻 Qwen-Image-Layered 分层 daemon(仅双卡布局:单卡装不下 fp8 常驻)
if [ "$N_GPU" -ge 2 ]; then
    bash "$SERVICE_DIR/qwenlayerd.sh" || echo "[run.sh] warn: qwen layered daemon start failed"
fi

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
