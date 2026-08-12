import { useEffect, useState } from 'react'
import { Button, Card, Input, Select, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'
import { recomposeDecomposedBar } from '../lib/barDecompose'
import stepDefaults from '../config/stepDefaults'

const { TextArea } = Input

const PREVIEW_BG: React.CSSProperties = {
  background:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

/**
 * 第 13+ 步"bar分解",两段式:
 * 1. 归组区(可选,≥2 条 bar 才显示):gemini 判同类并各选一个代表;
 *    不归组则"全部拆解"分解所有 bar。
 * 2. 拆解区:三段式拆解(边框/进度内容/底板,chroma green),并行生成、
 *    出一张显示一张;每条结果可独立"拼回"/"重新分解",互不影响。
 */
export default function BarDecomposePanel() {
  const {
    runInfo,
    structuredResult,
    midStatus,
    barDecomposeModel,
    barDecomposeSystemPrompt,
    barDecomposeSystemPromptV,
    barDecomposeUserPrompt,
    barDecomposeStatus,
    barDecomposeError,
    barDecomposeItems,
    barDecomposeBusy,
    barGroupModel,
    barGroupSystemPrompt,
    barGroupUserPrompt,
    barGroupStatus,
    barGroupError,
    barGroups,
    runBarGrouping,
    runBarDecompose,
    cancelBarDecompose,
    setField,
  } = useDetectionStore()

  const barCount = pickDetections(structuredResult, 'bar').length
  const canRun =
    Boolean(runInfo) && barCount > 0 && midStatus.bar === 'done'
  const running = barDecomposeBusy.length > 0
  const grouping = barGroupStatus === 'running'
  // 拼回预览:index → dataURL('' 表示计算中);换 run 即清
  const [recomposed, setRecomposed] = useState<Record<number, string>>({})
  const runId = runInfo?.run_id
  useEffect(() => {
    setRecomposed({})
  }, [runId])
  const recompose = async (index: number, url: string, orientation: string) => {
    setRecomposed((m) => ({ ...m, [index]: '' }))
    try {
      const out = await recomposeDecomposedBar(
        url, orientation as 'horizontal' | 'vertical',
      )
      setRecomposed((m) => ({ ...m, [index]: out }))
    } catch {
      setRecomposed((m) => {
        const next = { ...m }
        delete next[index]
        return next
      })
    }
  }

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              13+
            </span>
            bar分解
          </span>
          {running ? (
            <Button size="small" danger onClick={() => cancelBarDecompose()}>
              取消
            </Button>
          ) : null}
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      <div className="flex flex-col gap-3">
        {/* ── 归组区(可选):只有多条 bar 才有意义 ── */}
        {barCount > 1 ? (
          <div className="flex flex-col gap-2 rounded border border-neutral-200 bg-neutral-50 p-2">
            <div className="flex items-center justify-between">
              <span className="text-[13px] font-bold">
                同类归组(可选)
              </span>
              <Button
                size="small"
                loading={grouping}
                disabled={!canRun || running}
                onClick={() => void runBarGrouping()}
              >
                {barGroups.length ? '重新归组' : '归组'}
              </Button>
            </div>
            <div>
              <div className="text-[12px] font-bold">归组模型</div>
              <Input
                size="small"
                value={barGroupModel}
                onChange={(e) => setField('barGroupModel', e.target.value)}
                disabled={grouping}
              />
            </div>
            <div>
              <div className="text-[12px] font-bold">归组系统提示词</div>
              <TextArea
                value={barGroupSystemPrompt}
                onChange={(e) => setField('barGroupSystemPrompt', e.target.value)}
                disabled={grouping}
                rows={4}
              />
            </div>
            <div>
              <div className="text-[12px] font-bold">归组用户提示词</div>
              <TextArea
                value={barGroupUserPrompt}
                onChange={(e) => setField('barGroupUserPrompt', e.target.value)}
                disabled={grouping}
                rows={2}
              />
            </div>
            {barGroupStatus === 'error' ? (
              <div className="break-all text-[12px] text-[#cf1322]">
                归组失败：{barGroupError}
              </div>
            ) : null}
            {barGroups.length ? (
              <div className="flex flex-col gap-1">
                <span className="text-[12px] font-bold">
                  归组结果({barGroups.length} 类 / {barCount} 条,每类只拆解代表)
                </span>
                {barGroups.map((g, gi) => (
                  <div key={gi} className="text-[12px] leading-relaxed">
                    <span className="font-bold">{g.name}</span>
                    ：成员 {g.members.map((m) => `#${m}`).join(' ')}，选
                    <span className="font-bold text-[#1677ff]"> #{g.selected} </span>
                    拆解
                    {g.reason ? (
                      <span className="text-black/45">（{g.reason}）</span>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : (
              <span className="text-[11px] text-black/45">
                传入原图 + 第 13 步 bar 提取层,gemini 判同类 bar(同素材、同进度含义,仅列表复用)并各选一个"进度最多但非 100%"的代表。不归组则下方"全部拆解"会分解所有 {barCount} 条 bar。
              </span>
            )}
          </div>
        ) : null}

        {/* ── 拆解区 ── */}
        <div>
          <div className="text-[13px] font-bold">拆解模型</div>
          <Select
            className="w-full"
            value={barDecomposeModel}
            onChange={(v) => setField('barDecomposeModel', v)}
            disabled={running}
            options={stepDefaults.barDecompose.models.map((id: string) => ({
              value: id,
              label: id.split('/').pop(),
            }))}
          />
        </div>
        <div>
          <div className="text-[13px] font-bold">系统提示词(纵向bar → 三段横排)</div>
          <TextArea
            value={barDecomposeSystemPrompt}
            onChange={(e) => setField('barDecomposeSystemPrompt', e.target.value)}
            disabled={running}
            rows={5}
          />
        </div>
        <div>
          <div className="text-[13px] font-bold">系统提示词(横向bar → 三段竖排,不压缩比例)</div>
          <TextArea
            value={barDecomposeSystemPromptV}
            onChange={(e) => setField('barDecomposeSystemPromptV', e.target.value)}
            disabled={running}
            rows={5}
          />
        </div>
        <div>
          <div className="text-[13px] font-bold">用户提示词</div>
          <TextArea
            value={barDecomposeUserPrompt}
            onChange={(e) => setField('barDecomposeUserPrompt', e.target.value)}
            disabled={running}
            rows={5}
          />
        </div>

        <Button
          type="primary"
          disabled={!canRun || running}
          onClick={() => void runBarDecompose()}
        >
          开始全部拆解
          {barCount > 1
            ? barGroups.length
              ? `(${barGroups.length} 个代表)`
              : `(全部 ${barCount} 条)`
            : ''}
        </Button>
        {barDecomposeStatus === 'error' ? (
          <div className="break-all text-center text-[12px] text-[#cf1322]">
            拆解失败：{barDecomposeError}
          </div>
        ) : null}
        {!canRun ? (
          <span className="px-2 text-center text-[12px] text-black/45">
            {!runInfo
              ? '请先上传图片'
              : barCount === 0
                ? '检测结果中没有 bar'
                : '请先完成第 13 步提bar(需要 bar.png 提取层)'}
          </span>
        ) : null}
        {running ? (
          <div className="flex items-center justify-center gap-3">
            <Spin size="small" />
            <span className="text-[12px] text-black/45">
              并行拆解中…进行中 {barDecomposeBusy.length} 条(#
              {barDecomposeBusy.join(' #')}),出一张显示一张
            </span>
          </div>
        ) : null}

        {barDecomposeItems.map((it) => (
          <div key={it.index} className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-[12px] font-bold">
                bar #{it.index}（{it.orientation === 'horizontal' ? '横向' : '纵向'}）
              </span>
              <span className="flex gap-1">
                <Button
                  size="small"
                  disabled={recomposed[it.index] === ''}
                  onClick={() => void recompose(it.index, it.url, it.orientation)}
                >
                  拼回
                </Button>
                <Button
                  size="small"
                  disabled={!canRun || barDecomposeBusy.includes(it.index)}
                  onClick={() => {
                    setRecomposed((m) => {
                      const next = { ...m }
                      delete next[it.index]
                      return next
                    })
                    void runBarDecompose([it.index])
                  }}
                >
                  重新分解
                </Button>
              </span>
            </div>
            <div style={PREVIEW_BG}>
              <img
                src={it.url}
                alt={`bar ${it.index} 分解`}
                className="h-auto max-w-full object-contain"
              />
            </div>
            {it.layers && Object.keys(it.layers).length ? (
              <span className="text-[11px] text-black/45">
                已落盘最小 PNG:{Object.values(it.layers).join('、')}
                {!it.layers.border ? '(无 border 层,未生成)' : ''}
              </span>
            ) : null}
            {recomposed[it.index] !== undefined ? (
              <div className="flex flex-col gap-1">
                <span className="text-[11px] text-black/45">
                  拼回效果(底板→填充→边框 中轴线配准叠合,棋盘格 = 透明)
                </span>
                {recomposed[it.index] === '' ? (
                  <div className="flex justify-center py-2">
                    <Spin size="small" />
                  </div>
                ) : (
                  <div style={PREVIEW_BG}>
                    <img
                      src={recomposed[it.index]}
                      alt={`bar ${it.index} 拼回`}
                      className="h-auto max-w-full object-contain"
                    />
                  </div>
                )}
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </Card>
  )
}
