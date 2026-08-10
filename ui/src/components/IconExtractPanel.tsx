import type { CSSProperties } from 'react'
import { Button, Card, Checkbox, InputNumber, Popover, Select, Spin } from 'antd'
import { SettingOutlined } from '@ant-design/icons'
import {
  useDetectionStore,
  type IconTier,
  type IconTierParams,
} from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'

// 透明 PNG 用棋盘格背景展示,方便看抠图边缘
const CHECKERBOARD: CSSProperties = {
  backgroundImage:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

const TIER_FIELDS: {
  key: keyof IconTierParams
  label: string
  step: number
  min: number
  max?: number
}[] = [
  { key: 'paddingRatio', label: 'padRatio', step: 0.01, min: 0 },
  { key: 'minPadding', label: 'minPad', step: 1, min: 0 },
  { key: 'maskThreshold', label: 'maskThr', step: 0.05, min: -10, max: 10 },
  { key: 'featherRadius', label: 'feather', step: 0.5, min: 0 },
  { key: 'cropScale', label: 'cropScale', step: 0.1, min: 0, max: 5 },
]

function TierSection({
  tier,
  title,
  disabled,
}: {
  tier: IconTier
  title: string
  disabled: boolean
}) {
  const params = useDetectionStore((s) => s.iconTierParams[tier])
  const setIconTierParam = useDetectionStore((s) => s.setIconTierParam)
  return (
    <div className="rounded border border-neutral-200 p-2">
      <div className="mb-1 text-[12px] font-bold">{title}</div>
      <div className="grid grid-cols-5 gap-2">
        {TIER_FIELDS.map((f) => (
          <div key={f.key}>
            <div className="mb-0.5 text-[10px] text-black/45">{f.label}</div>
            <InputNumber
              className="w-full"
              size="small"
              value={params[f.key] as number}
              onChange={(v) => setIconTierParam(tier, f.key, (v ?? 0) as never)}
              disabled={disabled}
              step={f.step}
              min={f.min}
              max={f.max}
            />
          </div>
        ))}
      </div>
      <div className="mt-1.5 flex gap-4">
        <Checkbox
          checked={params.refine}
          disabled={disabled}
          onChange={(e) => setIconTierParam(tier, 'refine', e.target.checked)}
        >
          <span className="text-[11px]">二轮精化</span>
        </Checkbox>
        <Checkbox
          checked={params.multimask}
          disabled={disabled}
          onChange={(e) => setIconTierParam(tier, 'multimask', e.target.checked)}
        >
          <span className="text-[11px]">多候选取最优</span>
        </Checkbox>
        <Checkbox
          checked={params.fillHoles}
          disabled={disabled}
          onChange={(e) => setIconTierParam(tier, 'fillHoles', e.target.checked)}
        >
          <span className="text-[11px]">mask 自动封孔</span>
        </Checkbox>
      </div>
    </div>
  )
}

function SettingsPopover() {
  const iconSource = useDetectionStore((s) => s.iconSource)
  const iconSmallMaxSide = useDetectionStore((s) => s.iconSmallMaxSide)
  const iconLargeMinSide = useDetectionStore((s) => s.iconLargeMinSide)
  const iconStatus = useDetectionStore((s) => s.iconStatus)
  const setField = useDetectionStore((s) => s.setField)
  const running = iconStatus === 'running'

  return (
    <Popover
      trigger="hover"
      placement="bottomRight"
      content={
        <div className="flex w-[460px] flex-col gap-3">
          <div className="flex items-center gap-2">
            <span className="text-[12px] font-bold">提取源图</span>
            <Select
              size="small"
              className="flex-1"
              value={iconSource}
              disabled={running}
              onChange={(v) => setField('iconSource', v)}
              options={[
                {
                  value: 'auto',
                  label: '双源择优',
                },
                { value: 'text_back', label: '去字图（第 2 步结果）' },
                { value: 'origin', label: '原图' },
              ]}
            />
          </div>
          <div className="flex items-center gap-2 text-[12px]">
            <span>分档阈值（icon 像素长边）：≤</span>
            <InputNumber
              size="small"
              className="w-16"
              value={iconSmallMaxSide}
              onChange={(v) => setField('iconSmallMaxSide', v ?? 48)}
              disabled={running}
              min={1}
              precision={0}
            />
            <span>为小图标，≥</span>
            <InputNumber
              size="small"
              className="w-16"
              value={iconLargeMinSide}
              onChange={(v) => setField('iconLargeMinSide', v ?? 160)}
              disabled={running}
              min={2}
              precision={0}
            />
            <span>为大图标</span>
          </div>
          <TierSection tier="small" title={`小图标（长边 ≤ ${iconSmallMaxSide}px）`} disabled={running} />
          <TierSection tier="medium" title="中图标（介于两者之间）" disabled={running} />
          <TierSection tier="large" title={`大图标（长边 ≥ ${iconLargeMinSide}px）`} disabled={running} />
          <div className="text-[11px] leading-snug text-black/40">
            padRatio=外扩比例 · minPad=最小外扩px · maskThr=掩码阈值（越高越保守）·
            feather=边缘羽化 · cropScale=切片倍数（≤1 整图）
            <br />
            二轮精化=mask_input 再收敛一轮，有填洞倾向，保留区偏多时先关它 ·
            多候选取最优=icon 连衬底整块被抠时开它，常能选中更细粒度 ·
            mask 自动封孔=每个 icon 的 mask 实心无镂空，想保留镂空处透明就关掉
          </div>
        </div>
      }
    >
      <SettingOutlined
        className="cursor-pointer text-[16px] text-black/45 transition-colors hover:text-[#1677ff]"
        aria-label="提icon参数设置"
      />
    </Popover>
  )
}

export default function IconExtractPanel() {
  const {
    runInfo,
    structuredResult,
    textBackStatus,
    iconStatus,
    iconSource,
    iconImageUrl,
    iconError,
    iconRefineQaStatus,
    iconRefineQaError,
    iconRefineQaInfo,
    runExtractIcons,
    runRefineIcons,
  } = useDetectionStore()

  const analyzedIcons = useDetectionStore((s) => s.analyzedIcons)
  const useAnalyzed = Boolean(analyzedIcons?.length)
  const iconCount = useAnalyzed
    ? (analyzedIcons?.length ?? 0)
    : pickDetections(structuredResult, 'icon').length
  // 纯原图只需已上传;涉及去字图的模式(去字图/双源择优)需第 2 步已出结果
  const sourceReady =
    iconSource === 'origin' ? Boolean(runInfo) : textBackStatus === 'done'
  const sourceLabel =
    iconSource === 'auto' ? '双源择优' : iconSource === 'origin' ? '原图' : '去文字图'
  const canExtract = sourceReady && iconCount > 0


  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              8
            </span>
            提icon
          </span>
          <div className="flex items-center gap-2">
            {iconStatus === 'done' ? (
              <>
                <Button
                  size="small"
                  loading={iconRefineQaStatus === 'running'}
                  onClick={() => void runRefineIcons()}
                >
                  修正
                </Button>
                <Button
                  size="small"
                  disabled={!canExtract}
                  onClick={() => void runExtractIcons()}
                >
                  重新提取
                </Button>
              </>
            ) : null}
            <SettingsPopover />
          </div>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {iconStatus === 'running' ? (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          <Spin />
          <span className="text-[12px] text-black/45">
            正在提取 icon…（SAM2 抠图，GPU 队列串行执行）
          </span>
        </div>
      ) : iconStatus === 'done' ? (
        <div className="flex flex-col gap-2">
          {iconRefineQaStatus === 'running' ? (
            <div className="text-[12px] text-black/45">
              正在质检…（gemini 3.1 pro 检查粘连/保守问题,发现问题会自动重抠）
            </div>
          ) : iconRefineQaStatus === 'error' ? (
            <div className="break-all text-[12px] text-[#cf1322]">
              修正失败：{iconRefineQaError}
            </div>
          ) : iconRefineQaInfo ? (
            <div className="break-all text-[12px] text-black/60">
              {iconRefineQaInfo}
            </div>
          ) : null}
          <div style={CHECKERBOARD}>
            <img
              src={iconImageUrl}
              alt="icon 提取结果"
              className="h-auto max-w-full object-contain"
            />
          </div>
        </div>
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          {iconStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              提取失败：{iconError}
            </div>
          ) : null}
          <Button
            type="primary"
            disabled={!canExtract}
            onClick={() => void runExtractIcons()}
          >
            {iconStatus === 'error' ? '重试' : '提取'}
          </Button>
          {!canExtract ? (
            <span className="px-2 text-center text-[12px] text-black/45">
              {!sourceReady
                ? iconSource === 'origin'
                  ? '请先在第 1 步上传图片'
                  : '请先完成第 2 步去文字'
                : '检测结果中没有 icon（第 3 步检测需返回非空 icon 数组）'}
            </span>
          ) : (
            <span className="text-[12px] text-black/45">
              将以「{sourceLabel}」提取 {iconCount} 个 icon
              {useAnalyzed ? '（小图标带第 7 步正点，其余仅检测框）' : '（未分析，使用原始检测框）'}
            </span>
          )}
        </div>
      )}
    </Card>
  )
}
