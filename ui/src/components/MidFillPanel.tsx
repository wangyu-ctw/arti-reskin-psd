import { Button, Card, Checkbox, Input, InputNumber, Popover, Spin } from 'antd'
import { SettingOutlined } from '@ant-design/icons'
import {
  DEFAULT_MID_FILL_PROMPT,
  useDetectionStore,
} from '../stores/useDetectionStore'

const { TextArea } = Input

function NumberField({
  label,
  hint,
  field,
  step,
  min,
  max,
}: {
  label: string
  hint: string
  field:
    | 'midFillSeed'
    | 'midFillSteps'
    | 'midFillGuidance'
    | 'midFillGrowMask'
    | 'midFillMaskBlur'
    | 'midFillMaxPixels'
  step: number
  min: number
  max?: number
}) {
  const value = useDetectionStore((s) => s[field])
  const setField = useDetectionStore((s) => s.setField)
  return (
    <div>
      <div className="text-[13px] font-bold">{label}</div>
      <div className="mb-1 text-[11px] leading-snug text-black/40">{hint}</div>
      <InputNumber
        className="w-full"
        value={value}
        onChange={(v) => setField(field, v ?? 0)}
        step={step}
        min={min}
        max={max}
      />
    </div>
  )
}

function SettingsPopover() {
  const midFillPrompt = useDetectionStore((s) => s.midFillPrompt)
  const midFillFillHoles = useDetectionStore((s) => s.midFillFillHoles)
  const midFillStatus = useDetectionStore((s) => s.midFillStatus)
  const runInfo = useDetectionStore((s) => s.runInfo)
  const setField = useDetectionStore((s) => s.setField)
  const seekHref = `/seedseek?task=fill_mid${
    runInfo
      ? `&img=${encodeURIComponent(`/api/runs/${runInfo.run_id}/files/icon_back.png`)}`
      : ''
  }`

  return (
    <Popover
      trigger="hover"
      placement="bottomRight"
      content={
        <div className="flex w-[360px] flex-col gap-3">
          <div>
            <div className="text-[13px] font-bold">提示词</div>
            <div className="mb-1 text-[11px] leading-snug text-black/40">
              描述破洞区域应该补成什么内容
            </div>
            <TextArea
              value={midFillPrompt}
              onChange={(e) => setField('midFillPrompt', e.target.value)}
              placeholder={DEFAULT_MID_FILL_PROMPT}
              rows={4}
            />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <NumberField
              label="SEED"
              hint="随机种子，相同参数结果可复现"
              field="midFillSeed"
              step={1}
              min={0}
            />
            <NumberField
              label="Step"
              hint="采样步数，越多越精细但越慢"
              field="midFillSteps"
              step={1}
              min={1}
              max={50}
            />
            <NumberField
              label="guidance"
              hint="引导强度，越高越听提示词，越低越贴周边"
              field="midFillGuidance"
              step={1}
              min={1}
              max={100}
            />
            <NumberField
              label="growMask"
              hint="修补区外扩像素，盖住元素边缘残留"
              field="midFillGrowMask"
              step={1}
              min={0}
            />
            <NumberField
              label="maskBlur"
              hint="修补区边缘羽化，衔接过渡更自然"
              field="midFillMaskBlur"
              step={0.5}
              min={0}
            />
            <NumberField
              label="maxPixels"
              hint="处理像素上限，超出按比例缩小"
              field="midFillMaxPixels"
              step={65536}
              min={65536}
            />
          </div>
          <Checkbox
            checked={midFillFillHoles}
            disabled={midFillStatus === 'running'}
            onChange={(e) => setField('midFillFillHoles', e.target.checked)}
          >
            <span className="text-[12px]">
              自动封闭 mask 内部孔洞（消灭元素残片；镂空处也会一并重绘）
            </span>
          </Checkbox>
          <Button size="small" href={seekHref} target="_blank">
            找seed(新标签页打开)
          </Button>
        </div>
      }
    >
      <SettingOutlined
        className="cursor-pointer text-[16px] text-black/45 transition-colors hover:text-[#1677ff]"
        aria-label="修补参数设置"
      />
    </Popover>
  )
}

export default function MidFillPanel() {
  const {
    midHoleStatus,
    midFillStatus,
    midFillImageUrl,
    midFillError,
    runMidFill,
  } = useDetectionStore()

  const canRun = midHoleStatus === 'done'

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              15
            </span>
            修补
          </span>
          <div className="flex items-center gap-2">
            {midFillStatus === 'done' ? (
              <Button size="small" onClick={() => void runMidFill()}>
                重新生成
              </Button>
            ) : null}
            <SettingsPopover />
          </div>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {midFillStatus === 'running' ? (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          <Spin />
          <span className="text-[12px] text-black/45">
            正在修补…（Fill 补洞 + icon_back LoRA，GPU 队列串行执行）
          </span>
        </div>
      ) : midFillStatus === 'done' ? (
        <img
          src={midFillImageUrl}
          alt="中景层修补结果"
          className="h-auto max-w-full object-contain"
        />
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          {midFillStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              修补失败：{midFillError}
            </div>
          ) : null}
          <Button
            type="primary"
            disabled={!canRun}
            onClick={() => void runMidFill()}
          >
            {midFillStatus === 'error' ? '重试' : '修补'}
          </Button>
          {!canRun ? (
            <span className="text-[12px] text-black/45">
              请先完成第 14 步中景层破洞图
            </span>
          ) : null}
        </div>
      )}
    </Card>
  )
}
