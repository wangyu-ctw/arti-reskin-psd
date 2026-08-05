#!/usr/bin/env bash
# 新 pod 开机体检:30 秒判断这张卡能不能用,坏的立刻退货,别浪费配置时间。
#
# 用法:
#   ./scripts/check_pod.sh --tcp-ports=<port> --host=<pod-ip>
#
# 检查项:GPU 识别 / ECC 错误 / 显存分配 + bf16 矩阵乘微基准 / 网络卷读速。
# 全部通过输出 POD_OK;任何一项不对劲输出 POD_BAD 和原因。
set -euo pipefail

SSH_KEY="$HOME/.ssh/id_ed25519"
PORT=""; HOST=""
for arg in "$@"; do
    case "$arg" in
        --tcp-ports=*) PORT="${arg#*=}" ;;
        --host=*)      HOST="${arg#*=}" ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done
[ -n "$PORT" ] && [ -n "$HOST" ] || { echo "usage: $0 --tcp-ports=<port> --host=<ip>"; exit 1; }

ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p "$PORT" -i "$SSH_KEY" "root@$HOST" 'bash -s' <<'REMOTE'
set -u
BAD=""

# 1) GPU 识别
if ! GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1); then
    echo "POD_BAD: nvidia-smi 失败 -> $GPU"; exit 1
fi
echo "GPU: $GPU"

# 2) ECC / XID 错误(掉卡、显存坏块的常见前兆)
ECC=$(nvidia-smi -q 2>/dev/null | grep -A2 "ECC Errors" | grep -cE "[1-9][0-9]*" || true)
XID=$(dmesg 2>/dev/null | grep -c "Xid" || true)
[ "$XID" -gt 0 ] && { BAD="$BAD dmesg存在Xid错误($XID条);"; }

# 3) 显存分配 + bf16 矩阵乘微基准(用现成 venv,10GB 分配 + 计时)
PY=/workspace/venvs/omnipsd-cu128/bin/python
[ -x "$PY" ] || PY=/workspace/ui_skin/.venv/bin/python
BENCH=$($PY - <<'EOF' 2>&1
import time, torch
assert torch.cuda.is_available(), "cuda不可用"
x = torch.empty(int(10e9 // 2), dtype=torch.bfloat16, device="cuda")  # 10GB 分配
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    a = a @ a * 1e-4
torch.cuda.synchronize()
ms = (time.time() - t) / 20 * 1000
print(f"BENCH_OK {ms:.1f}ms/matmul")
EOF
) || true
echo "$BENCH" | grep -q BENCH_OK || BAD="$BAD GPU计算测试失败($BENCH);"
echo "计算: $BENCH"
# 经验值:RTX PRO 6000 Blackwell 上 8192^2 bf16 matmul 约 15-40ms;>150ms 说明降频/异常
MS=$(echo "$BENCH" | grep -oE "[0-9.]+ms" | tr -d "ms" | cut -d. -f1 || echo 999)
[ "${MS:-999}" -gt 150 ] && BAD="$BAD matmul过慢(${MS}ms,疑似降频);"

# 4) 网络卷读速(1GB 顺序读,正常 500MB/s+;<150MB/s 会明显拖模型加载)
SPEED=$(dd if=/workspace/models/FLUX.1-Fill-dev/flux1-fill-dev.safetensors of=/dev/null bs=1M count=1024 2>&1 | grep -oE "[0-9.]+ [MG]B/s" | tail -1)
echo "网络卷: $SPEED"
echo "$SPEED" | grep -qE "GB/s|[2-9][0-9][0-9] MB/s" || BAD="$BAD 网络卷读速偏低($SPEED);"

if [ -n "$BAD" ]; then
    echo "POD_BAD:$BAD"
    exit 1
fi
echo "POD_OK"
REMOTE
