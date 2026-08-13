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
| T5-XXL 文本编码(fp8 scaled) | `/workspace/models/FLUX.1-dev/t5xxl_fp8_e4m3fn_scaled.safetensors` | 4.9G | `models/text_encoders/t5xxl_fp16.safetensors`(软链名未改) | 在役(2026-08-12 换装,32G 卡腾显存) |
| T5-XXL 文本编码(fp16,回滚备份) | `/workspace/models/FLUX.1-dev/t5xxl_fp16_merged.safetensors`(手动合并分片) | ~9G | — | 备份(`ln -sfn` 即回滚) |
| VAE | `/workspace/models/FLUX.1-dev/ae.safetensors` | 320M | `models/vae/ae.safetensors` | 在役 |
| 去字 LoRA(2026-08-04 续训 step-1505) | `/workspace/outputs/text_back_20260804/step-1505.safetensors` | 1.2G | `models/loras/omnipsd_text_back.safetensors` | 在役(2026-08-04 上线) |
| 去icon补洞 LoRA(Fill 范式,rank 32,3000 步) | `/workspace/outputs/icon_back_fill_20260804/pytorch_lora_weights.safetensors` | 172M | `models/loras/icon_back_fill.safetensors` | 在役·第 9 步(2026-08-04 上线) |
| 中景修补 LoRA(mid_fill,icon_back 续训 3000→6000 步,1875 块中景洞团) | `/workspace/outputs/mid_fill_20260807/pytorch_lora_weights.safetensors` | 172M | `models/loras/mid_fill.safetensors` | 在役·第 14 步(2026-08-10 上线,A/B 幻觉少于 icon_back) |
| panel 修补 LoRA(panel_fill,mid_fill 续训 6000→9000 步,1012 样本/2086 块剥洋葱数据) | `/workspace/outputs/panel_fill_20260807/pytorch_lora_weights.safetensors` | 172M | `models/loras/panel_fill.safetensors` | 在役·panel_extract 默认(2026-08-11 上线);checkpoint-6500~9000 可回退 |

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

## Qwen-Image-Edit(icon_repair 任务:icon 语义修复+高清)

2026-08-06 上线。与 FLUX 系共用一个 ComfyUI,按任务自动换载(48G 装不下并存,切换时有加载耗时)。

| 用途 | 真实路径 | 大小 | ComfyUI 内软链位置 | 状态 |
|---|---|---|---|---|
| 编辑主模型(2511 fp8mixed) | `/workspace/hf_models/comfy_qwen_image/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` | ~20G | `models/diffusion_models/qwen_image_edit.safetensors` | 在役 |
| Qwen2.5-VL 7B 文本编码(fp8) | `.../split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | ~9G | `models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | 在役 |
| Qwen Image VAE | `.../split_files/vae/qwen_image_vae.safetensors` | ~250M | `models/vae/qwen_image_vae.safetensors` | 在役 |

来源:Comfy-Org 公开仓(匿名可下,不受 HF token 失效影响)。
icon_repair 链路:逐 icon 裁块 → Qwen 编辑(修复混乱区+统一放大到 1MP 高清)→ SAM2 抠图 →
最小透明 PNG 存 `<run>/icon/`,`manifest.csv` 记录回贴坐标,`recompose.png` 为拼回预览。
icon_asset 链路(第 8+ 步素材化,2026-08-07 上线):第 7 步分组(name/slug)→ 每组选最大成员
→ 上下文裁块 + icons.png 抠图合成纯色底 → Qwen 双图参照重绘(Plus 节点)→ 自适应边界泛洪去底
(以输出图边框中位色为准,再清贴边杂物)→ `<run>/icon_assets/<slug>.png` + manifest + 拼回预览。

## 独立守护进程

| 用途 | 路径 | 大小 | 状态 |
|---|---|---|---|
| SAM2(抠图,icon 专项续训 step-1000:贴轮廓环带负点治底座误判,val icon 0.924) | `/workspace/outputs/sam2_icon_20260805/step-1000.pt` | 857M | 在役(2026-08-05 二次上线) |
| SAM2 全类均衡版(step-7000,val IoU 0.890/基线 0.748) | `/workspace/outputs/sam2_train_20260805/step-7000.pt` | 857M | 回退备份 |
| SAM2 官方原版 sam2.1_hiera_large | `/workspace/sam2/checkpoints/sam2.1_hiera_large.pt` | 857M | 回退备份(sam2d.sh 删掉 SAM2_CHECKPOINT 行即回退) |
| YOLO game0804_p2(默认,新数据 P2,mAP50 0.701 / icon P 0.851) | `/workspace/ui_skin/pretrained/yolo/yolo_game0804_p2_best.pt` | 42M | 在役·唯一注册(2026-08-12 起) |
| YOLO game0804_11m(mAP50 0.752) | `/workspace/ui_skin/pretrained/yolo/yolo_game0804_best.pt` | 39M | 已撤注册,文件保留 |
| YOLO game0728_p2(旧 P2) | `/workspace/ui_skin/pretrained/yolo/yolo_game0728_p2_best.pt` | 42M | 已撤注册,文件保留 |

YOLO 注册表(2026-08-12 精简):daemon 只挂 game0804_p2 一个权重(前端下拉同步只剩它);
历史权重与 panel_amodal 文件仍在 pretrained/yolo/,要 A/B 时在 yolo_daemon.py 的 MODEL_PATHS 加回即可。
默认由 yolod.sh 的 `YOLO_DEFAULT_MODEL` 指定(当前 game0804_p2)。
另:检测后默认做 SAM2 bbox 几何回投(refine_bbox,治框小一截;text/panel 类不回投)。
| YOLO 旧版(game0804 的训练起点) | `/workspace/ui_skin/pretrained/yolo/yolo_ui_element_best.pt` | 39M | 回退备份 |

YOLO 训练产物区:`/workspace/outputs/yolo_train/<name>/weights/best.pt`(game0804_from_old 即本次)。

panel 专项 YOLO(amodal 全貌 bbox,单类 panel;注册已撤、文件保留——实测效果差于 game0804_p2 的 panel 类,提升需加新游戏数据):
`/workspace/ui_skin/pretrained/yolo/yolo_panel_amodal_20260811_best.pt`(39M)。
数据 `/workspace/inputs/panel_yolo_20260811/`(786 训练/90 验证,ansatsu holdout,每 PSD 三态图 full/mid/stack 共享 amodal 标签)。
v1 默认超参 mAP50 0.552;v2(AdamW 5e-4+余弦退火+fliplr)mAP50 0.570 / mAP50-95 0.372,即当前在役版。
两版都是 best 停在前几个 epoch、后续过拟合——瓶颈是训练集游戏风格多样性(泛化到全新游戏就这个水平),
提升需加游戏而非调参。曾短暂接入第 15 步"新yolo"按钮,实测后撤下。

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
