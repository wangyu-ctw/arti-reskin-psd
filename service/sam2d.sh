#!/usr/bin/env bash
# 常驻 SAM2 抠图 daemon:只监听本机 127.0.0.1:8189,由 service 代理调用。
# 已在跑则直接退出,可安全重复执行(run.sh 每次启动都会调一次)。
set -euo pipefail

SAM2_PYTHON="/workspace/sam2-env/bin/python"
DAEMON="/workspace/service/model_scripts/sam2_daemon.py"
LOG="/workspace/servData/_logs/sam2d.log"

if pgrep -f "[s]am2_daemon.py" > /dev/null; then
    echo "[sam2d.sh] already running"
    exit 0
fi

mkdir -p "$(dirname "$LOG")"
cd /workspace/sam2
nohup "$SAM2_PYTHON" "$DAEMON" > "$LOG" 2>&1 &
echo "[sam2d.sh] sam2 daemon starting (pid $!), log: $LOG"
