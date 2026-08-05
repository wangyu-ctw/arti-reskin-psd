#!/usr/bin/env bash
# YOLO 训练脚本(单机版,在数据所在的 GPU 机器上直接执行)
#
# 数据目录要求:
#   <data>/images/*.png|jpg    训练图片(文件名以游戏名前缀开头,如 ansatsu_xxx.png)
#   <data>/labels/*.txt        YOLO 标注,与图片同名
#
# 用法:
#   ./train_yolo.sh --data=/workspace/inputs/yolo_game0728 --name=game0801
#
# 参数:
#   --data=DIR        数据集目录(必填)
#   --name=xxx        训练名(必填),产物在 <out>/<name>/weights/best.pt
#   --out=DIR         输出根目录,默认 /workspace/outputs/yolo_train
#   --val-game=xxx    留作验证集的游戏前缀,默认 ansatsu;
#                     建议历次训练固定同一个,指标才可比
#   --model=PATH      起点模型(.pt 加训 / .yaml 新架构),
#                     默认 /workspace/ui_skin/pretrained/yolo/yolo_game0728_p2_best.pt
#   --pretrained=PATH 仅当 --model 是 yaml 时使用
#   --yolo-bin=PATH   yolo 命令路径,默认 /workspace/ui_skin/.venv/bin/yolo
#   --imgsz=1600 --epochs=200 --patience=50 --batch=8
set -euo pipefail

DATA=""; NAME=""
OUT="/workspace/outputs/yolo_train"
VAL_GAME="ansatsu"
MODEL="/workspace/ui_skin/pretrained/yolo/yolo_game0728_p2_best.pt"
PRETRAINED=""
YOLO_BIN="/workspace/ui_skin/.venv/bin/yolo"
IMGSZ=1600; EPOCHS=200; PATIENCE=50; BATCH=8

usage() {
    echo "usage: $0 --data=<dir> --name=<run_name> [--out=DIR] [--val-game=ansatsu]"
    echo "          [--model=PATH] [--pretrained=PATH] [--yolo-bin=PATH]"
    echo "          [--imgsz=1600] [--epochs=200] [--patience=50] [--batch=8]"
    exit 1
}

for arg in "$@"; do
    case "$arg" in
        --data=*)       DATA="${arg#*=}" ;;
        --name=*)       NAME="${arg#*=}" ;;
        --out=*)        OUT="${arg#*=}" ;;
        --val-game=*)   VAL_GAME="${arg#*=}" ;;
        --model=*)      MODEL="${arg#*=}" ;;
        --pretrained=*) PRETRAINED="${arg#*=}" ;;
        --yolo-bin=*)   YOLO_BIN="${arg#*=}" ;;
        --imgsz=*)      IMGSZ="${arg#*=}" ;;
        --epochs=*)     EPOCHS="${arg#*=}" ;;
        --patience=*)   PATIENCE="${arg#*=}" ;;
        --batch=*)      BATCH="${arg#*=}" ;;
        -h|--help)      usage ;;
        *) echo "unknown arg: $arg"; usage ;;
    esac
done

[ -n "$DATA" ] || { echo "error: missing --data"; usage; }
[ -n "$NAME" ] || { echo "error: missing --name"; usage; }
[ -d "$DATA/images" ] && [ -d "$DATA/labels" ] || { echo "error: $DATA 下缺 images/ 或 labels/"; exit 1; }
[ -x "$YOLO_BIN" ] || { echo "error: yolo 命令不存在: $YOLO_BIN(用 --yolo-bin 指定)"; exit 1; }

DATA="$(cd "$DATA" && pwd)"   # 转绝对路径,清单里要用

echo "[1/2] 生成 train/val 清单(val=${VAL_GAME}*)..."
DATA="$DATA" VAL_GAME="$VAL_GAME" python3 << 'EOF'
import os
from pathlib import Path

data = Path(os.environ["DATA"])
val_game = os.environ["VAL_GAME"]
imgs = sorted((data / "images").glob("*.png")) + sorted((data / "images").glob("*.jpg"))
assert imgs, "images/ 下没有图片"
train = [str(p) for p in imgs if not p.name.startswith(val_game + "_")]
val = [str(p) for p in imgs if p.name.startswith(val_game + "_")]
assert train, "训练集为空"
assert val, f"验证集为空:没有以 {val_game}_ 开头的图片"
(data / "train.txt").write_text("\n".join(train) + "\n")
(data / "val.txt").write_text("\n".join(val) + "\n")
(data / "dataset.yaml").write_text(
    f"path: {data}\ntrain: train.txt\nval: val.txt\nnc: 6\n"
    "names: ['text', 'icon', 'assets', 'button', 'bar', 'panel']\n")
print(f"  train {len(train)} 张 / val {len(val)} 张")
EOF

echo "[2/2] 开始训练(前台运行,Ctrl+C 中断)..."
mkdir -p "$OUT"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"
PRETRAIN_ARG=""
[ -n "$PRETRAINED" ] && PRETRAIN_ARG="pretrained=$PRETRAINED"

"$YOLO_BIN" detect train \
    model="$MODEL" $PRETRAIN_ARG \
    data="$DATA/dataset.yaml" \
    imgsz="$IMGSZ" epochs="$EPOCHS" patience="$PATIENCE" batch="$BATCH" device=0 \
    project="$OUT" name="$NAME" exist_ok=True

echo
echo "===== 完成 ====="
echo "最优权重: $OUT/$NAME/weights/best.pt"
echo "上线切换: cp 到 /workspace/ui_skin/pretrained/yolo/ 并更新 yolod.sh 的 YOLO_MODEL 后重启 daemon"
