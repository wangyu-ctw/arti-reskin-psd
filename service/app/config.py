"""全局配置。路径类配置支持环境变量覆盖,方便本地调试。"""
import os
from pathlib import Path

# 业务数据统一根目录,每个 run_id 一个子目录
DATA_ROOT = Path(os.environ.get("SERV_DATA_ROOT", "/workspace/servData"))

HOST = "0.0.0.0"
PORT = 8888

# 上传原图允许的扩展名
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".psd"}

# ---- ComfyUI(常驻推理后端)----
COMFY_URL = "http://127.0.0.1:8188"
COMFY_ROOT = Path("/workspace/ComfyUI")
COMFY_TIMEOUT = 3600  # 秒,单个 workflow 的执行上限

# ---- SAM2 抠图 daemon(常驻,模型加载一次)----
SAM2D_URL = "http://127.0.0.1:8189"
SAM2D_TIMEOUT = 600  # 秒,单次抠图上限

# ---- YOLO 检测 daemon(常驻,模型加载一次)----
YOLOD_URL = "http://127.0.0.1:8190"
YOLOD_TIMEOUT = 300  # 秒,单次检测上限

# ---- flux_fill 精准修补(FLUX.1-Fill-dev,按 mask 重绘)----
# unet 文件需下载并软链到 ComfyUI/models/diffusion_models/ 下的这个名字
FLUX_FILL_UNET_NAME = "flux1-fill-dev.safetensors"

# ---- text_back 去字模型(FLUX Kontext + OmniPSD LoRA)----
# LoRA 已软链到 ComfyUI/models/loras/ 下的这个名字
TEXT_BACK_LORA_NAME = "omnipsd_text_back.safetensors"
# 去icon补洞 LoRA(FLUX.1-Fill 范式,2026-08-04 训练)
# 只作用于模型侧,不动文本编码器
ICON_BACK_LORA_NAME = "icon_back_fill.safetensors"

# 冷启动子进程方案(备用,走 diffsynth,每次任务重新加载模型)
OMNIPSD_PYTHON = "/workspace/venvs/omnipsd-cu128/bin/python"
OMNIPSD_ROOT = "/workspace/OmniPSD"
TEXT_BACK_SCRIPT = Path(__file__).resolve().parent.parent / "model_scripts" / "run_text_back_keep_size_service.py"
TEXT_BACK_TIMEOUT = 3600  # 秒,单张推理含模型加载的上限
