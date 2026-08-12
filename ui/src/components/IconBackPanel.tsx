import { Button, Card, Checkbox, Input, InputNumber, Popover, Spin } from 'antd'
import { SettingOutlined } from '@ant-design/icons'
import {
  DEFAULT_ICON_BACK_PROMPT,
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
    | 'iconBackSeed'
    | 'iconBackSteps'
    | 'iconBackGuidance'
    | 'iconBackGrowMask'
    | 'iconBackMaskBlur'
    | 'iconBackMaxPixels'
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
  const iconBackPrompt = useDetectionStore((s) => s.iconBackPrompt)
  const iconBackFillHoles = useDetectionStore((s) => s.iconBackFillHoles)
  const iconBackStatus = useDetectionStore((s) => s.iconBackStatus)
  const runInfo = useDetectionStore((s) => s.runInfo)
  const setField = useDetectionStore((s) => s.setField)
  const seekHref = `/seedseek?task=fill_icon${
    runInfo
      ? `&img=${encodeURIComponent(`/api/runs/${runInfo.run_id}/files/text_back.png`)}`
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
              value={iconBackPrompt}
              onChange={(e) => setField('iconBackPrompt', e.target.value)}
              placeholder={DEFAULT_ICON_BACK_PROMPT}
              rows={4}
            />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <NumberField
              label="SEED"
              hint="随机种子，相同参数结果可复现"
              field="iconBackSeed"
              step={1}
              min={0}
            />
            <NumberField
              label="Step"
              hint="采样步数，越多越精细但越慢"
              field="iconBackSteps"
              step={1}
              min={1}
              max={50}
            />
            <NumberField
              label="guidance"
              hint="引导强度，越高越听提示词，越低越贴周边"
              field="iconBackGuidance"
              step={1}
              min={1}
              max={100}
            />
            <NumberField
              label="growMask"
              hint="修补区外扩像素，盖住 icon 边缘残留"
              field="iconBackGrowMask"
              step={1}
              min={0}
            />
            <NumberField
              label="maskBlur"
              hint="修补区边缘羽化，衔接过渡更自然"
              field="iconBackMaskBlur"
              step={0.5}
              min={0}
            />
            <NumberField
              label="maxPixels"
              hint="处理像素上限，超出按比例缩小"
              field="iconBackMaxPixels"
              step={65536}
              min={65536}
            />
          </div>
          <Checkbox
            checked={iconBackFillHoles}
            disabled={iconBackStatus === 'running'}
            onChange={(e) => setField('iconBackFillHoles', e.target.checked)}
          >
            <span className="text-[12px]">
              自动封闭 mask 内部孔洞（消灭 icon 残片；镂空处也会一并重绘）
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
        aria-label="去icon参数设置"
      />
    </Popover>
  )
}

export default function IconBackPanel() {
  const {
    iconStatus,
    iconBackStatus,
    iconBackImageUrl,
    iconBackError,
    midStatus,
    runIconBack,
    runMidExtract,
  } = useDetectionStore()

  const canRun = iconStatus === 'done'
  const midRunning = Object.values(midStatus).some((s) => s === 'running')

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              10
            </span>
            生成中景层
          </span>
          <div className="flex items-center gap-2">
            {iconBackStatus === 'done' ? (
              <Button size="small" onClick={() => void runIconBack()}>
                重新生成
              </Button>
            ) : null}
            <Button
              size="small"
              type="primary"
              ghost
              loading={midRunning}
              disabled={iconBackStatus !== 'done'}
              onClick={() => void runMidExtract()}
            >
              提取中景层
            </Button>
            <SettingsPopover />
          </div>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {iconBackStatus === 'running' ? (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          <Spin />
          <span className="text-[12px] text-black/45">
            正在去 icon…（生成破洞图 + Fill 修补，GPU 队列串行执行）
          </span>
        </div>
      ) : iconBackStatus === 'done' ? (
        <img
          src={iconBackImageUrl}
          alt="去 icon 结果"
          className="h-auto max-w-full object-contain"
        />
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          {iconBackStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              去 icon 失败：{iconBackError}
            </div>
          ) : null}
          <Button
            type="primary"
            disabled={!canRun}
            onClick={() => void runIconBack()}
          >
            {iconBackStatus === 'error' ? '重试' : '去icon'}
          </Button>
          {!canRun ? (
            <span className="text-[12px] text-black/45">
              请先完成第 8 步提icon
            </span>
          ) : null}
        </div>
      )}
    </Card>
  )
}
