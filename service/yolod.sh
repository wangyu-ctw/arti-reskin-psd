#!/usr/bin/env bash
# 常驻 YOLO 检测 daemon:只监听本机 127.0.0.1:8190,由 service 代理调用。
# 已在跑则直接退出,可安全重复执行(run.sh 每次启动都会调一次)。
set -euo pipefail

YOLO_PYTHON="/workspace/ui_skin/.venv/bin/python"
DAEMON="/workspace/service/model_scripts/yolo_daemon.py"
LOG="/workspace/servData/_logs/yolod.log"
# 当前:11m 新数据版(game0804,整体 mAP50 0.752,2026-08-05 对比测试中)
# 回退:yolo_game0728_p2_best.pt(旧 P2)/ yolo_game0804_p2_best.pt(新数据 P2,mAP50 0.701)/ yolo_ui_element_best.pt(旧单类)
export YOLO_MODEL="/workspace/ui_skin/pretrained/yolo/yolo_game0804_best.pt"

if pgrep -f "[y]olo_daemon.py" > /dev/null; then
    echo "[yolod.sh] already running"
    exit 0
fi

mkdir -p "$(dirname "$LOG")"
cd /workspace/ui_skin
nohup "$YOLO_PYTHON" "$DAEMON" > "$LOG" 2>&1 &
echo "[yolod.sh] yolo daemon starting (pid $!), log: $LOG"
