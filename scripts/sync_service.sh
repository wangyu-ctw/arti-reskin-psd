#!/usr/bin/env bash
# 把本地 service/ 同步到 RunPod 的 /workspace/service
#
# 用法:
#   ./scripts/sync_service.sh --tcp-ports=<port>
#   ./scripts/sync_service.sh --tcp-ports=<port> --host=<pod-ip>
#   ./scripts/sync_service.sh --tcp-ports=<port> --dry-run
#
# host 也可以用环境变量 RUNPOD_HOST,或直接改下面的 DEFAULT_HOST。
# 端口就是 RunPod Connect 页面 "SSH over exposed TCP" 里 -p 后面那个数字,
# pod 每次重启会变,所以每次通过 --tcp-ports= 注入。

set -euo pipefail

# ---- 可按需修改的默认值 ----
DEFAULT_HOST="${RUNPOD_HOST:-}"        # pod 公网 IP
SSH_USER="root"
SSH_KEY="$HOME/.ssh/id_ed25519"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/service"
REMOTE_DIR="/workspace/service"

PORT=""
HOST="$DEFAULT_HOST"
DRY_RUN=""

usage() {
    echo "usage: $0 --tcp-ports=<port> [--host=<ip>] [--dry-run]"
    echo "       host 也可用环境变量 RUNPOD_HOST 提供"
    exit 1
}

for arg in "$@"; do
    case "$arg" in
        --tcp-ports=*) PORT="${arg#*=}" ;;
        --host=*)      HOST="${arg#*=}" ;;
        --dry-run)     DRY_RUN="-n" ;;
        -h|--help)     usage ;;
        *) echo "unknown arg: $arg"; usage ;;
    esac
done

[ -n "$PORT" ] || { echo "error: missing --tcp-ports=<port>"; usage; }
[ -n "$HOST" ] || { echo "error: missing host (--host= 或 RUNPOD_HOST)"; usage; }
[ -d "$LOCAL_DIR" ] || { echo "error: local dir not found: $LOCAL_DIR"; exit 1; }

# pod 重启后 ip:port 对应的 host key 会变,固定校验只会添乱,直接跳过
SSH_CMD="ssh -p $PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "[sync] $LOCAL_DIR/ -> $SSH_USER@$HOST:$REMOTE_DIR/ (port $PORT) ${DRY_RUN:+[dry-run]}"
# 不用 -a:/workspace 是网络卷,不允许 chown,-a 附带的属主/属组保留会报错
rsync -rlptvz --delete $DRY_RUN \
    --exclude '.venv' \
    --exclude '.deps_installed' \
    --exclude '__pycache__' \
    --exclude '.DS_Store' \
    -e "$SSH_CMD" \
    "$LOCAL_DIR/" "$SSH_USER@$HOST:$REMOTE_DIR/"

echo "[sync] done. 远端重启服务: $SSH_CMD $SSH_USER@$HOST 'pkill -f uvicorn; nohup bash /workspace/service/run.sh > /workspace/servData/_logs/service.log 2>&1 &'"
