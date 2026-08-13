import type { CSSProperties } from 'react'
import { Button, Card, Checkbox, Input, InputNumber, Popover, Spin } from 'antd'
import { SettingOutlined } from '@ant-design/icons'
import { useDetectionStore } from '../stores/useDetectionStore'

type PeelNumKey =
  | 'seed' | 'steps' | 'guidance' | 'grow' | 'blur'
  | 'paddingRatio' | 'minPadding' | 'maskThreshold' | 'featherRadius' | 'cropScale'
type PeelBoolKey = 'refine' | 'multimask' | 'fillHoles'

const FILL_FIELDS: {
  key: PeelNumKey
  label: string
  hint: string
  step: number
  min: number
  max?: number
}[] = [
  { key: 'seed', label: 'SEED', hint: '随机种子,可复现', step: 1, min: 0 },
  { key: 'steps', label: 'Step', hint: '采样步数,越多越精细越慢', step: 1, min: 1, max: 50 },
  { key: 'guidance', label: 'guidance', hint: '提示词强度,越低越贴周边', step: 1, min: 1, max: 100 },
  { key: 'grow', label: 'grow', hint: '洞外扩像素,盖住抠图残边', step: 1, min: 0 },
  { key: 'blur', label: 'blur', hint: '洞边羽化,衔接更自然', step: 0.5, min: 0 },
]

const SAM_FIELDS: typeof FILL_FIELDS = [
  { key: 'paddingRatio', label: 'paddingRatio', hint: '框每边按比例外扩', step: 0.01, min: 0 },
  { key: 'minPadding', label: 'minPadding', hint: '每边最小外扩像素', step: 1, min: 0 },
  { key: 'maskThreshold', label: 'maskThreshold', hint: '掩码阈值,越高越保守', step: 0.05, min: -10, max: 10 },
  { key: 'featherRadius', label: 'feather', hint: '抠图边缘羽化,0 硬边', step: 0.5, min: 0 },
  { key: 'cropScale', label: 'cropScale', hint: '按框倍数裁切片分割;≤1 整图', step: 0.1, min: 0, max: 5 },
]

const SAM_BOOLS: { key: PeelBoolKey; label: string }[] = [
  { key: 'refine', label: '二轮精化(mask_input 再收敛一轮)' },
  { key: 'multimask', label: '多候选取最优' },
  { key: 'fillHoles', label: 'mask 自动封孔(实心无镂空)' },
]

function SettingsPopover() {
  const params = useDetectionStore((s) => s.panelPeelParams)
  const status = useDetectionStore((s) => s.panelPeelStatus)
  const setField = useDetectionStore((s) => s.setField)
  const disabled = status === 'running'
  return (
    <Popover
      trigger="hover"
      placement="bottomRight"
      content={
        <div className="flex w-[380px] flex-col gap-3">
          <div className="text-[13px] font-bold">拆(SAM2 整层抠出)</div>
          <div className="grid grid-cols-3 gap-3">
            {SAM_FIELDS.map((f) => (
              <div key={f.key}>
                <div className="text-[13px] font-bold">{f.label}</div>
                <div className="mb-1 text-[11px] leading-snug text-black/40">
                  {f.hint}
                </div>
                <InputNumber
                  className="w-full"
                  value={params[f.key]}
                  onChange={(v) =>
                    setField('panelPeelParams', { ...params, [f.key]: v ?? 0 })
                  }
                  disabled={disabled}
                  step={f.step}
                  min={f.min}
                  max={f.max}
                />
              </div>
            ))}
          </div>
          <div className="flex flex-col gap-1">
            {SAM_BOOLS.map((f) => (
              <Checkbox
                key={f.key}
                checked={params[f.key]}
                disabled={disabled}
                onChange={(e) =>
                  setField('panelPeelParams', { ...params, [f.key]: e.target.checked })
                }
              >
                <span className="text-[12px]">{f.label}</span>
              </Checkbox>
            ))}
          </div>
          <div className="text-[13px] font-bold">补(flux_fill 还原下层)</div>
          <div className="grid grid-cols-3 gap-3">
            {FILL_FIELDS.map((f) => (
              <div key={f.key}>
                <div className="text-[13px] font-bold">{f.label}</div>
                <div className="mb-1 text-[11px] leading-snug text-black/40">
                  {f.hint}
                </div>
                <InputNumber
                  className="w-full"
                  value={params[f.key]}
                  onChange={(v) =>
                    setField('panelPeelParams', { ...params, [f.key]: v ?? 0 })
                  }
                  disabled={disabled}
                  step={f.step}
                  min={f.min}
                  max={f.max}
                />
              </div>
            ))}
          </div>
          <div>
            <div className="text-[13px] font-bold">补洞提示词</div>
            <div className="mb-1 text-[11px] leading-snug text-black/40">
              每次"补下层"的 flux_fill 用它(LoRA 固定挂 panel_fill,不开放编辑)
            </div>
            <Input.TextArea
              value={params.prompt}
              onChange={(e) =>
                setField('panelPeelParams', { ...params, prompt: e.target.value })
              }
              disabled={disabled}
              rows={4}
            />
          </div>
        </div>
      }
    >
      <SettingOutlined
        className="cursor-pointer text-[16px] text-black/45 transition-colors hover:text-[#1677ff]"
        aria-label="分层提取参数设置"
      />
    </Popover>
  )
}

const CHECKERBOARD: CSSProperties = {
  backgroundImage:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

/**
 * 第 17 步"分层提取":按第 16 步审核的层级做 拆-补-拆-补 剥洋葱——
 * 从最上层起,SAM2 整层抠出 → flux_fill(panel_fill LoRA)把洞补全还原下层
 * → 继续拆下一层。每个 z 层出一张完整 RGBA(panel_layers/z<k>.png)。
 * 每个 panel 的细拆放到后续步骤。
 */
export default function PanelExtractPanel() {
  const {
    runInfo,
    midFillStatus,
    panelAuditStatus,
    panelAuditItems,
    panelPeelStatus,
    panelPeelError,
    panelPeelLevels,
    panelPeelTick,
    panelPeelElapsed,
    runPanelPeel,
  } = useDetectionStore()

  const auditCount = panelAuditItems.length
  const levelCount = new Set(panelAuditItems.map((a) => a.z)).size
  const canRun =
    Boolean(runInfo) && midFillStatus === 'done' &&
    panelAuditStatus === 'done' && auditCount > 0
  const running = panelPeelStatus === 'running'

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              17
            </span>
            分层提取
          </span>
          <div className="flex items-center gap-2">
            {panelPeelStatus === 'done' ? (
              <Button size="small" disabled={!canRun} onClick={() => void runPanelPeel()}>
                重新提取
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
            拆-补-拆 剥洋葱中…(共 {levelCount} 层,每层 SAM2 整层抠出 +
            补全下层,层间串行)
          </span>
        </div>
      ) : panelPeelStatus === 'done' ? (
        <div className="flex flex-col gap-3">
          <span className="text-[12px] font-bold">
            完成:{panelPeelLevels.length} 层,耗时 {panelPeelElapsed}s;
            产物在 panel_layers/(z 越大越上层)
          </span>
          {[...panelPeelLevels]
            .sort((a, b) => b.z - a.z)
            .map((lv) => (
              <div key={lv.z} className="flex flex-col gap-1">
                <span className="text-[12px] font-bold">
                  z{lv.z}({lv.count} 个 panel)
                </span>
                <div style={CHECKERBOARD} className="rounded border border-neutral-200">
                  <img
                    src={`/api/runs/${runInfo?.run_id}/files/${lv.file}?t=${panelPeelTick}`}
                    alt={`层 z${lv.z}`}
                    className="h-auto max-w-full object-contain"
                  />
                </div>
              </div>
            ))}
          <span className="text-[11px] leading-relaxed text-black/45">
            上层拆走后,下层被压住的区域已由 panel_fill 补全再拆——每层都是
            完整图层;补后的中间工作图存为 panel_layers/stage_after_z&lt;k&gt;.png
            可复查。每个 panel 的细拆在后续步骤做。
          </span>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 py-4">
          {panelPeelStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              分层提取失败：{panelPeelError}
            </div>
          ) : null}
          <Button type="primary" disabled={!canRun} onClick={() => void runPanelPeel()}>
            {panelPeelStatus === 'error' ? '重试' : '分层提取'}
          </Button>
          <span className="px-2 text-center text-[12px] text-black/45">
            {!runInfo
              ? '请先上传图片'
              : midFillStatus !== 'done'
                ? '请先完成第 15 步中景修补(需要 mid_fill.png)'
                : panelAuditStatus !== 'done' || auditCount === 0
                  ? '请先完成第 16 步 panel修正(分层按它的清单与层级执行)'
                  : `将按第 16 步的 ${auditCount} 个 panel / ${levelCount} 层做 拆-补-拆 剥洋葱`}
          </span>
        </div>
      )}
    </Card>
  )
}
