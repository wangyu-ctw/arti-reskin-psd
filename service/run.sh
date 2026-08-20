#!/usr/bin/env bash
# 启动脚本: bash /workspace/service/run.sh
# 单进程 uvicorn(--workers 1),保证进程内 FIFO 队列全局唯一,GPU 严格串行。
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SERVICE_DIR"

DATA_ROOT="${SERV_DATA_ROOT:-/workspace/servData}"
mkdir -p "$DATA_ROOT/_logs"

# 依赖装进容器本地盘的 venv:网络卷(mfs)会随机丢小文件,venv 放卷上
# 反复损坏(bin/python 消失、包缺 __main__)。venv 是一次性产物,每个
# pod 首次启动重建一次(~30s),不值得为持久化冒损坏风险。
# 按 Python 小版本分目录:不同 pod 模板 python3 版本不同(3.11/3.12)
PYV=$(python3 -c 'import sys; print(f"py{sys.version_info[0]}{sys.version_info[1]}")')
VENV="${SERVICE_VENV_DIR:-/root/.venvs}/service-$PYV"
mkdir -p "$(dirname "$VENV")"
if [ ! -x "$VENV/bin/python" ]; then
    echo "[run.sh] creating venv at $VENV ..."
    # --copies:网络卷上软链会莫名消失(mfs),venv 一律用实体文件
    python3 -m venv --copies "$VENV"
fi
MARKER="$VENV/.deps_installed"
if [ ! -f "$MARKER" ] || [ "requirements.txt" -nt "$MARKER" ] \
   || ! "$VENV/bin/python" -c "import uvicorn" 2>/dev/null; then
    echo "[run.sh] installing dependencies..."
    "$VENV/bin/pip" install -r requirements.txt
    touch "$MARKER"
fi

# 按 GPU 数/显存选布局(GPU_PLAN.md):
#   1 卡 <80G = 布局 A:全部落 GPU0,单泳道 FIFO(现状)
#   1 卡 ≥80G = 布局 C:同 A,另起 qwenlayerd(bf16 基座+双 adapter ~40G 与
#               FLUX 换载共存,96G 从容)
#   ≥2 卡 = 布局 B:ComfyUI/FLUX 独占 GPU0;SAM2/YOLO 钉到 GPU1;双泳道并行
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
QWENLD=0
if [ "$N_GPU" -ge 2 ]; then
    echo "[run.sh] $N_GPU GPUs detected -> 双卡布局 B(Comfy@GPU0,SAM2/YOLO@GPU1,双泳道)"
    export DETECT_GPU=1
    export SERVICE_LANE_MODE=dual
    QWENLD=1
elif [ "${VRAM_MB:-0}" -ge 80000 ]; then
    # 双泳道:qwen 生成(分钟级)走主道,检测/审核/抠取走副道,互不排队
    echo "[run.sh] single GPU ${VRAM_MB}MiB -> 布局 C(双泳道 + qwenlayerd)"
    export SERVICE_LANE_MODE=dual
    QWENLD=1
else
    echo "[run.sh] single GPU ${VRAM_MB}MiB -> 布局 A(单泳道)"
    export SERVICE_LANE_MODE=single
fi

# 常驻 ComfyUI(text_back 等模型任务的推理后端,模型只加载一次;双卡时独占 GPU0)
CUDA_VISIBLE_DEVICES=0 bash "$SERVICE_DIR/comfyui.sh" || echo "[run.sh] warn: ComfyUI start failed, text_back will not work"

# 常驻 SAM2 抠图 daemon(双卡时 DETECT_GPU 钉到 GPU1)
bash "$SERVICE_DIR/sam2d.sh" || echo "[run.sh] warn: sam2 daemon start failed, sam2 will not work"

# 常驻 YOLO 检测 daemon(同上)
bash "$SERVICE_DIR/yolod.sh" || echo "[run.sh] warn: yolo daemon start failed, yolo will not work"

# 常驻 Qwen-Image-Layered 分层 daemon(布局 B/C;v2:基座+six_slot/panelz 双 adapter)
if [ "$QWENLD" = "1" ]; then
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
