import type { CSSProperties } from 'react'
import { Button, Card, InputNumber, Popover, Spin } from 'antd'
import { SettingOutlined } from '@ant-design/icons'
import { useDetectionStore } from '../stores/useDetectionStore'

const CHECKERBOARD: CSSProperties = {
  backgroundImage:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

const FIELDS: {
  key: 'layers' | 'steps' | 'seed' | 'trueCfg'
  label: string
  hint: string
  step: number
  min: number
  max?: number
}[] = [
  { key: 'layers', label: '层数', hint: '含 bg;训练口径为 6(bg+5)', step: 1, min: 2, max: 8 },
  { key: 'steps', label: 'Step', hint: '采样步数', step: 1, min: 10, max: 60 },
  { key: 'seed', label: 'SEED', hint: '随机种子,可复现', step: 1, min: 0 },
  { key: 'trueCfg', label: 'CFG', hint: '引导强度(true_cfg)', step: 0.5, min: 1, max: 10 },
]

function SettingsPopover() {
  const params = useDetectionStore((s) => s.qwenLayerParams)
  const status = useDetectionStore((s) => s.qwenLayerStatus)
  const setField = useDetectionStore((s) => s.setField)
  const disabled = status === 'running'
  return (
    <Popover
      trigger="hover"
      placement="bottomRight"
      content={
        <div className="grid w-[320px] grid-cols-2 gap-3">
          {FIELDS.map((f) => (
            <div key={f.key}>
              <div className="text-[13px] font-bold">{f.label}</div>
              <div className="mb-1 text-[11px] leading-snug text-black/40">
                {f.hint}
              </div>
              <InputNumber
                className="w-full"
                value={params[f.key]}
                onChange={(v) =>
                  setField('qwenLayerParams', { ...params, [f.key]: v ?? 0 })
                }
                disabled={disabled}
                step={f.step}
                min={f.min}
                max={f.max}
              />
            </div>
          ))}
        </div>
      }
    >
      <SettingOutlined
        className="cursor-pointer text-[16px] text-black/45 transition-colors hover:text-[#1677ff]"
        aria-label="Qwen 分层参数设置"
      />
    </Popover>
  )
}

/**
 * 第 16 步(新)"Qwen分层":微调后的 Qwen-Image-Layered 一步把 mid_fill
 * 分解为 bg + panel 按 z 分层的 RGBA 图层(替换旧 16/17 两步试验中,
 * 旧实现代码保留)。需要双卡布局(daemon 常驻 GPU1)。
 */
export default function QwenLayerPanel() {
  const {
    runInfo,
    midFillStatus,
    qwenLayerStatus,
    qwenLayerError,
    qwenLayerFiles,
    qwenLayerTick,
    qwenLayerElapsed,
    runQwenLayer,
  } = useDetectionStore()

  const canRun = Boolean(runInfo) && midFillStatus === 'done'
  const running = qwenLayerStatus === 'running'

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              16
            </span>
            Qwen分层
          </span>
          <div className="flex items-center gap-2">
            {qwenLayerStatus === 'done' ? (
              <Button size="small" disabled={!canRun} onClick={() => void runQwenLayer()}>
                重新分层
              </Button>
            ) : null}
            <SettingsPopover />
          </div>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {running ? (
        <div className="flex flex-col items-center gap-3 py-6">
          <Spin />
          <span className="text-[12px] text-black/45">
            Qwen-Image-Layered 分层中…(40 步采样约 2~4 分钟,GPU1 执行,
            与 FLUX 任务互不排队)
          </span>
        </div>
      ) : qwenLayerStatus === 'done' ? (
        <div className="flex flex-col gap-3">
          <span className="text-[12px] font-bold">
            完成:{qwenLayerFiles.length} 层,耗时 {qwenLayerElapsed}s;
            产物在 panel_layers_qwen/(bg 最底,z 越大越上层)
          </span>
          {[...qwenLayerFiles].reverse().map((f) => (
            <div key={f} className="flex flex-col gap-1">
              <span className="text-[12px] font-bold">
                {f.split('/').pop()?.replace('.png', '')}
              </span>
              <div style={CHECKERBOARD} className="rounded border border-neutral-200">
                <img
                  src={`/api/runs/${runInfo?.run_id}/files/${f}?t=${qwenLayerTick}`}
                  alt={f}
                  className="h-auto max-w-full object-contain"
                />
              </div>
            </div>
          ))}
          <span className="text-[11px] leading-relaxed text-black/45">
            生成式一步分层(LoRA 微调,层语义=你们的 panel 口径):被遮挡区域
            amodal 补全、透明通道原生。几何为生成级——需要像素级精确时用旧
            16/17 步(代码保留,App 里注释)。
          </span>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 py-4">
          {qwenLayerStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              分层失败：{qwenLayerError}
            </div>
          ) : null}
          <Button type="primary" disabled={!canRun} onClick={() => void runQwenLayer()}>
            {qwenLayerStatus === 'error' ? '重试' : 'Qwen一步分层'}
          </Button>
          <span className="px-2 text-center text-[12px] text-black/45">
            {!runInfo
              ? '请先上传图片'
              : midFillStatus !== 'done'
                ? '请先完成第 15 步中景修补(需要 mid_fill.png)'
                : '将把 mid_fill 一步分解为 bg + 5 个 panel 层(RGBA)'}
          </span>
        </div>
      )}
    </Card>
  )
}
