import { Button, Card, Divider, Input, InputNumber, Select, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'
import stepDefaults from '../config/stepDefaults'

const { TextArea } = Input

function ConfigSection() {
  const iconAnalysisModel = useDetectionStore((s) => s.iconAnalysisModel)
  const iconAnalysisTemperature = useDetectionStore((s) => s.iconAnalysisTemperature)
  const iconAnalysisSystemPrompt = useDetectionStore((s) => s.iconAnalysisSystemPrompt)
  const iconAnalysisUserPrompt = useDetectionStore((s) => s.iconAnalysisUserPrompt)
  const iconAnalysisStatus = useDetectionStore((s) => s.iconAnalysisStatus)
  const setField = useDetectionStore((s) => s.setField)
  const disabled = iconAnalysisStatus === 'running'

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-[13px] font-bold">模型</div>
          <div className="mb-1 text-[11px] leading-snug text-black/40">
            候选项在 stepDefaults.json 里维护
          </div>
          <Select
            className="w-full"
            value={iconAnalysisModel}
            onChange={(v) => setField('iconAnalysisModel', v)}
            disabled={disabled}
            options={stepDefaults.iconAnalysis.models.map((id) => ({
              value: id,
              label: id.split('/').pop(),
            }))}
          />
        </div>
        <div>
          <div className="text-[13px] font-bold">temperature</div>
          <div className="mb-1 text-[11px] leading-snug text-black/40">
            采样随机性，分析任务建议接近 0
          </div>
          <InputNumber
            className="w-full"
            value={iconAnalysisTemperature}
            onChange={(v) => setField('iconAnalysisTemperature', v ?? 0.1)}
            disabled={disabled}
            step={0.1}
            min={0}
            max={2}
          />
        </div>
      </div>
      <div>
        <div className="text-[13px] font-bold">系统提示词</div>
        <div className="mb-1 text-[11px] leading-snug text-black/40">
          定义分析规则与输出要求
        </div>
        <TextArea
          value={iconAnalysisSystemPrompt}
          onChange={(e) => setField('iconAnalysisSystemPrompt', e.target.value)}
          disabled={disabled}
          rows={6}
        />
      </div>
      <div>
        <div className="text-[13px] font-bold">用户提示词</div>
        <div className="mb-1 text-[11px] leading-snug text-black/40">
          发送时会在后面拼接 icon 检测框 JSON
        </div>
        <TextArea
          value={iconAnalysisUserPrompt}
          onChange={(e) => setField('iconAnalysisUserPrompt', e.target.value)}
          disabled={disabled}
          rows={2}
        />
      </div>
    </div>
  )
}

export default function IconAnalysisPanel() {
  const {
    structuredResult,
    textBackStatus,
    iconAnalysisStatus,
    iconAnalysisError,
    analyzedIcons,
    runAnalyzeIcons,
  } = useDetectionStore()

  // 与第 5 步/分析逻辑同口径:不计 discard 的审计记录
  const iconCount = pickDetections(structuredResult, 'icon').length
  const canAnalyze = textBackStatus === 'done' && iconCount > 0

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              6
            </span>
            分析icon
          </span>
          {iconAnalysisStatus === 'done' ? (
            <Button size="small" disabled={!canAnalyze} onClick={() => void runAnalyzeIcons()}>
              重新分析
            </Button>
          ) : null}
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      <ConfigSection />

      <Divider className="my-4" />

      {iconAnalysisStatus === 'running' ? (
        <div className="flex flex-col items-center gap-3 py-6">
          <Spin />
          <span className="text-[12px] text-black/45">
            正在分析 icon…（Gemini 审核轮廓并生成正负点）
          </span>
        </div>
      ) : iconAnalysisStatus === 'done' && analyzedIcons ? (
        <div className="flex flex-col gap-2">
          {analyzedIcons.map((icon) => (
            <div
              key={icon.index}
              className="rounded border border-neutral-200 px-3 py-2"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="min-w-0 truncate text-[13px]" title={icon.description}>
                  #{icon.index} {icon.description}
                </span>
                <span className="shrink-0 text-[11px] text-black/45">
                  正点 {icon.positive_points.length} · 负点 {icon.negative_points.length}
                </span>
              </div>
            </div>
          ))}
          <div className="mt-1 text-center text-[11px] text-black/45">
            下一步提icon将使用检测框原值 + 以上正负点
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 py-4">
          {iconAnalysisStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              分析失败：{iconAnalysisError}
            </div>
          ) : null}
          <Button
            type="primary"
            disabled={!canAnalyze}
            onClick={() => void runAnalyzeIcons()}
          >
            {iconAnalysisStatus === 'error' ? '重试' : '分析'}
          </Button>
          {!canAnalyze ? (
            <span className="px-2 text-center text-[12px] text-black/45">
              {textBackStatus !== 'done'
                ? '请先完成第 2 步去文字'
                : '检测结果中没有 icon（第 3 步检测需返回非空 icon 数组）'}
            </span>
          ) : (
            <span className="text-[12px] text-black/45">
              将分析 {iconCount} 个 icon（可跳过，直接在下一步用原始检测框提取）
            </span>
          )}
        </div>
      )}
    </Card>
  )
}
