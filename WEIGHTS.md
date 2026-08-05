# 推理权重清单

> 全部位于 RunPod `/workspace` 网络卷,pod 重建不丢。
> 最后核实:2026-08-04(pod bq3ssmyjmr6hn3)。权重有增删/切换时请同步更新本文件。

## ComfyUI 系(第 2 步去文字 + 第 8 步去icon)

工作流里只写文件名,实际通过 `ComfyUI/models/` 下的软链指向真实文件。

| 用途 | 真实路径 | 大小 | ComfyUI 内软链位置 | 状态 |
|---|---|---|---|---|
| Kontext 主模型(去文字) | `/workspace/models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors` | 23G | `models/diffusion_models/flux1-kontext-dev.safetensors` | 在役 |
| Fill 主模型(去icon补洞) | `/workspace/models/FLUX.1-Fill-dev/flux1-fill-dev.safetensors` | 23G | `models/diffusion_models/flux1-fill-dev.safetensors` | 在役 |
| CLIP-L 文本编码 | `/workspace/models/FLUX.1-dev/text_encoder/model.safetensors` | 235M | `models/text_encoders/clip_l.safetensors` | 在役 |
| T5-XXL 文本编码 | `/workspace/models/FLUX.1-dev/t5xxl_fp16_merged.safetensors`(手动合并分片) | ~9G | `models/text_encoders/t5xxl_fp16.safetensors` | 在役 |
| VAE | `/workspace/models/FLUX.1-dev/ae.safetensors` | 320M | `models/vae/ae.safetensors` | 在役 |
| 去字 LoRA(2026-08-04 续训 step-1505) | `/workspace/outputs/text_back_20260804/step-1505.safetensors` | 1.2G | `models/loras/omnipsd_text_back.safetensors` | 在役(2026-08-04 上线) |
| 去icon补洞 LoRA(Fill 范式,rank 32,3000 步) | `/workspace/outputs/icon_back_fill_20260804/pytorch_lora_weights.safetensors` | 172M | `models/loras/icon_back_fill.safetensors` | 在役(2026-08-04 上线) |

去字 LoRA 训练史:初版产物在 `/workspace/output/text_back/`(step-5500~7055,曾在役的 step-7055 可随时回退);
2026-08-04 以 step-7055 为基础、用 301 对新 PSD 数据续训 5 轮(lr 2e-5,脚本 `/workspace/OmniPSD/scripts/train_text_back_20260804.sh`),
产物在 `/workspace/outputs/text_back_20260804/`(step-500/1000/1500/1505)。
切换方法:`ln -sfn <目标> /workspace/ComfyUI/models/loras/omnipsd_text_back.safetensors` 后重启 ComfyUI(它会缓存已加载的 LoRA,只改软链不生效)。

去icon补洞 LoRA(icon_back):2026-08-04 用 1308 块 512×512 洞团局部图(no_text_icon+hole_mask 同窗切块)
以 FLUX.1-Fill 范式训练(diffusers fork 训练器,训练脚本 `/workspace/train_icon_back_fill.sh`,
prompt 与第 8 步推理一致,guidance 4)。产物含 checkpoint-500~3000 可回退。
第 8 步 flux_fill 任务默认挂载(payload `lora` 传空串可禁用,`lora_strength` 调强度)。
本地组装的 Fill diffusers 目录在 `/workspace/models/FLUX.1-Fill-dev-diffusers/`(训练用底模)。
注意:pod 上的 HF token 已失效,BFL 系 gated 仓不可下载;需要时让用户重新生成 token。

## 独立守护进程

| 用途 | 路径 | 大小 | 状态 |
|---|---|---|---|
| SAM2(第 7/9/10/11 步抠图) | `/workspace/sam2/checkpoints/sam2.1_hiera_large.pt` | 857M | 在役 |
| YOLO 检测(第 3 步,11m 新数据版 game0804,mAP50 0.752) | `/workspace/ui_skin/pretrained/yolo/yolo_game0804_best.pt` | 39M | 在役(2026-08-05 对比测试中) |
| YOLO 旧 P2(game0728_p2) | `/workspace/ui_skin/pretrained/yolo/yolo_game0728_p2_best.pt` | 42M | 回退/对比备份 |
| YOLO 新数据 P2(game0804_p2,mAP50 0.701 / icon P 0.851) | `/workspace/ui_skin/pretrained/yolo/yolo_game0804_p2_best.pt` | 42M | 回退/对比备份 |
| YOLO 旧版(game0804 的训练起点) | `/workspace/ui_skin/pretrained/yolo/yolo_ui_element_best.pt` | 39M | 回退备份 |

YOLO 训练产物区:`/workspace/outputs/yolo_train/<name>/weights/best.pt`(game0804_from_old 即本次)。

## 冷备通道(`text_back_cold`,diffsynth 直载,正常流程不触发)

复用 Kontext 主模型、`/workspace/models/FLUX.1-dev/text_encoder/`、`ae.safetensors`;
唯一独占的是 `/workspace/models/FLUX.1-dev/text_encoder_2/`(T5 分片目录,与合并版内容相同,~9G 重复存储,可清理候选)。

## 无关遗留(非本管线)

- `models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`、`models/loras/qwen-layered/` —— Qwen-layers 实验遗留,可清理候选。

## 配置入口

| 配置 | 位置 |
|---|---|
| ComfyUI 各加载器文件名 | `service/app/tasks.py`(workflow builder)、`service/app/config.py` |
| YOLO 模型路径 | `service/yolod.sh` 的 `YOLO_MODEL` 环境变量 |
| SAM2 检查点路径 | `service/model_scripts/sam2_daemon.py` 的 `SAM2_CHECKPOINT` 环境变量 |
