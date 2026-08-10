#!/usr/bin/env bash
# 常驻 YOLO 检测 daemon:只监听本机 127.0.0.1:8190,由 service 代理调用。
# 已在跑则直接退出,可安全重复执行(run.sh 每次启动都会调一次)。
set -euo pipefail

YOLO_PYTHON="/workspace/ui_skin/.venv/bin/python"
DAEMON="/workspace/service/model_scripts/yolo_daemon.py"
LOG="/workspace/servData/_logs/yolod.log"
# 多模型注册表在 yolo_daemon.py 里(game0804_11m / game0804_p2 / game0728_p2),
# 请求用 "model" 字段选择;这里只定默认。旧 YOLO_MODEL 变量仍兼容(注册为 "env")。
export YOLO_DEFAULT_MODEL="game0804_p2"

if pgrep -f "[y]olo_daemon.py" > /dev/null; then
    echo "[yolod.sh] already running"
    exit 0
fi

mkdir -p "$(dirname "$LOG")"
cd /workspace/ui_skin
nohup "$YOLO_PYTHON" "$DAEMON" > "$LOG" 2>&1 &
echo "[yolod.sh] yolo daemon starting (pid $!), log: $LOG"
