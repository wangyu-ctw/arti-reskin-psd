import type { CSSProperties } from 'react'
import { Button, Card, Checkbox, InputNumber, Popover, Spin } from 'antd'
import { SettingOutlined } from '@ant-design/icons'
import {
  useDetectionStore,
  type MidExtractParams,
  type MidKey,
} from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'

// 透明 PNG 用棋盘格背景展示,方便看抠图边缘
const CHECKERBOARD: CSSProperties = {
  backgroundImage:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

const NUMBER_FIELDS: {
  key: keyof MidExtractParams
  label: string
  hint: string
  step: number
  min: number
  max?: number
}[] = [
  { key: 'paddingRatio', label: 'paddingRatio', hint: '检测框每边按框尺寸比例外扩', step: 0.01, min: 0 },
  { key: 'minPadding', label: 'minPadding', hint: '每边最小外扩像素', step: 1, min: 0 },
  { key: 'maskThreshold', label: 'maskThreshold', hint: '掩码判定阈值，越高抠得越保守', step: 0.05, min: -10, max: 10 },
  { key: 'featherRadius', label: 'featherRadius', hint: '抠图边缘羽化半径，0 为硬边', step: 0.5, min: 0 },
  { key: 'cropScale', label: 'cropScale', hint: '按框倍数裁切片逐个分割；≤1 整图模式', step: 0.1, min: 0, max: 5 },
]

const BOOL_FIELDS: { key: keyof MidExtractParams; label: string }[] = [
  { key: 'refine', label: '二轮精化（mask_input 再收敛一轮，有填洞倾向）' },
  { key: 'multimask', label: '多候选取最优（目标连衬底整块被抠时开它）' },
  { key: 'fillHoles', label: 'mask 自动封孔（实心无镂空）' },
]

function SettingsPopover({ category }: { category: MidKey }) {
  const params = useDetectionStore((s) => s.midParams[category])
  const status = useDetectionStore((s) => s.midStatus[category])
  const setMidParam = useDetectionStore((s) => s.setMidParam)
  const disabled = status === 'running'

  return (
    <Popover
      trigger="hover"
      placement="bottomRight"
      content={
        <div className="flex w-[340px] flex-col gap-3">
          <div className="grid grid-cols-2 gap-3">
            {NUMBER_FIELDS.map((f) => (
              <div key={f.key}>
                <div className="text-[13px] font-bold">{f.label}</div>
                <div className="mb-1 text-[11px] leading-snug text-black/40">
                  {f.hint}
                </div>
                <InputNumber
                  className="w-full"
                  value={params[f.key] as number}
                  onChange={(v) => setMidParam(category, f.key, (v ?? 0) as never)}
                  disabled={disabled}
                  step={f.step}
                  min={f.min}
                  max={f.max}
                />
              </div>
            ))}
          </div>
          {BOOL_FIELDS.map((f) => (
            <Checkbox
              key={f.key}
              checked={params[f.key] as boolean}
              disabled={disabled}
              onChange={(e) => setMidParam(category, f.key, e.target.checked as never)}
            >
              <span className="text-[12px]">{f.label}</span>
            </Checkbox>
          ))}
        </div>
      }
    >
      <SettingOutlined
        className="cursor-pointer text-[16px] text-black/45 transition-colors hover:text-[#1677ff]"
        aria-label={`提${category}参数设置`}
      />
    </Popover>
  )
}

export default function MidExtractPanel({
  category,
  stepNo,
  title,
}: {
  category: MidKey
  stepNo: number
  title: string
}) {
  const structuredResult = useDetectionStore((s) => s.structuredResult)
  const iconBackStatus = useDetectionStore((s) => s.iconBackStatus)
  const status = useDetectionStore((s) => s.midStatus[category])
  const imageUrl = useDetectionStore((s) => s.midImageUrl[category])
  const error = useDetectionStore((s) => s.midError[category])
  const runMidOne = useDetectionStore((s) => s.runMidOne)

  const count = pickDetections(structuredResult, category).length
  const canRun = iconBackStatus === 'done' && count > 0

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              {stepNo}
            </span>
            {title}
          </span>
          <div className="flex items-center gap-2">
            {status === 'done' ? (
              <Button size="small" disabled={!canRun} onClick={() => void runMidOne(category)}>
                重新提取
              </Button>
            ) : null}
            <SettingsPopover category={category} />
          </div>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {status === 'running' ? (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          <Spin />
          <span className="text-[12px] text-black/45">
            正在提取 {category}…（SAM2 抠图，GPU 队列串行执行）
          </span>
        </div>
      ) : status === 'done' ? (
        <div style={CHECKERBOARD}>
          <img
            src={imageUrl}
            alt={`${category} 提取结果`}
            className="h-auto max-w-full object-contain"
          />
        </div>
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          {status === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              提取失败：{error}
            </div>
          ) : null}
          <Button
            type="primary"
            disabled={!canRun}
            onClick={() => void runMidOne(category)}
          >
            {status === 'error' ? '重试' : '提取'}
          </Button>
          {!canRun ? (
            <span className="px-2 text-center text-[12px] text-black/45">
              {iconBackStatus !== 'done'
                ? '请先完成第 8 步去icon'
                : `检测结果中没有 ${category}`}
            </span>
          ) : (
            <span className="text-[12px] text-black/45">
              将从去icon图提取 {count} 个 {category}（或在第 8 步一键"提取中景层"）
            </span>
          )}
        </div>
      )}
    </Card>
  )
}
