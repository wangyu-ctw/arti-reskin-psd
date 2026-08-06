#!/usr/bin/env bash
# 常驻 SAM2 抠图 daemon:只监听本机 127.0.0.1:8189,由 service 代理调用。
# 已在跑则直接退出,可安全重复执行(run.sh 每次启动都会调一次)。
set -euo pipefail

SAM2_PYTHON="/workspace/sam2-env/bin/python"
DAEMON="/workspace/service/model_scripts/sam2_daemon.py"
LOG="/workspace/servData/_logs/sam2d.log"
# 2026-08-05 二次上线:icon 专项续训(贴轮廓环带负点,专治底座误判;val icon 0.924)
# 回退:改回 sam2_train_20260805/step-7000.pt(全类均衡版);删掉整行则回官方权重
export SAM2_CHECKPOINT="/workspace/outputs/sam2_icon_20260805/step-1000.pt"

if pgrep -f "[s]am2_daemon.py" > /dev/null; then
    echo "[sam2d.sh] already running"
    exit 0
fi

mkdir -p "$(dirname "$LOG")"
cd /workspace/sam2
nohup "$SAM2_PYTHON" "$DAEMON" > "$LOG" 2>&1 &
echo "[sam2d.sh] sam2 daemon starting (pid $!), log: $LOG"
