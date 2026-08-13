import { Button, Card, Input, Select, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'
import stepDefaults from '../config/stepDefaults'

const { TextArea } = Input

const KIND_LABEL: Record<string, string> = {
  panel: '普通panel',
  banner: 'banner',
  title: '标题',
}
const TRANS_LABEL: Record<string, string> = {
  opaque: '不透明',
  uniform: '整体半透明',
  frame_solid: '边框实·中间透明',
  center_solid: '中间实·边缘透明',
}

/**
 * 第 16 步"panel修正":沿用第 3 步结构化检测的 panel,前端粗排层级后
 * 交给 Gemini VL 审核——纠正/删补 bbox、修层级 z、分 banner/title/panel、
 * 判透明类型。结果落盘 panel_audit.json,画框预览按 kind 分色。
 */
export default function PanelAuditPanel() {
  const {
    runInfo,
    structuredResult,
    midFillStatus,
    panelAuditModel,
    panelAuditSystemPrompt,
    panelAuditUserPrompt,
    panelAuditStatus,
    panelAuditError,
    panelAuditItems,
    panelAuditDeleted,
    panelAuditImageUrl,
    runPanelAudit,
    setField,
  } = useDetectionStore()

  const consumedCount = useDetectionStore(
    (s) => s.barConsumedOverlays.filter((c) => c.category === 'panel').length,
  )
  const panelCount =
    pickDetections(structuredResult, 'panel').length - consumedCount
  const canRun =
    Boolean(runInfo) && midFillStatus === 'done' && panelCount > 0
  const running = panelAuditStatus === 'running'

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              16
            </span>
            panel修正
          </span>
          {panelAuditStatus === 'done' ? (
            <Button size="small" disabled={!canRun} onClick={() => void runPanelAudit()}>
              重新审核
            </Button>
          ) : null}
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      <div className="flex flex-col gap-3">
        <div>
          <div className="text-[13px] font-bold">模型</div>
          <Select
            className="w-full"
            value={panelAuditModel}
            onChange={(v) => setField('panelAuditModel', v)}
            disabled={running}
            options={stepDefaults.panelAudit.models.map((id: string) => ({
              value: id,
              label: id.split('/').pop(),
            }))}
          />
        </div>
        <div>
          <div className="text-[13px] font-bold">系统提示词</div>
          <TextArea
            value={panelAuditSystemPrompt}
            onChange={(e) => setField('panelAuditSystemPrompt', e.target.value)}
            disabled={running}
            rows={5}
          />
        </div>
        <div>
          <div className="text-[13px] font-bold">用户提示词</div>
          <div className="mb-1 text-[11px] leading-snug text-black/40">
            发送时会在后面拼接 panel 检测项 JSON(含粗排 z)
          </div>
          <TextArea
            value={panelAuditUserPrompt}
            onChange={(e) => setField('panelAuditUserPrompt', e.target.value)}
            disabled={running}
            rows={2}
          />
        </div>

        {running ? (
          <div className="flex flex-col items-center gap-3 py-6">
            <Spin />
            <span className="text-[12px] text-black/45">
              Gemini 审核中…(纠删补 bbox / 层级 / banner·标题 / 透明类型)
            </span>
          </div>
        ) : panelAuditStatus === 'done' ? (
          <div className="flex flex-col gap-2">
            <span className="text-[12px] font-bold">
              审核结果:{panelAuditItems.length} 个 panel
              (新补 {panelAuditItems.filter((i) => i.source_index < 0).length} 个,
              原检测 {panelCount} 个),已落盘 panel_audit.json
            </span>
            {panelAuditImageUrl ? (
              <img
                src={panelAuditImageUrl}
                alt="panel 审核画框"
                className="h-auto max-w-full rounded border border-neutral-200 object-contain"
              />
            ) : null}
            <div className="flex flex-col gap-1">
              {panelAuditItems.map((it, i) => (
                <div key={i} className="text-[12px] leading-relaxed">
                  <span className="font-bold">z{it.z}</span>
                  {' · '}
                  {it.source_index < 0 ? (
                    <span className="font-bold text-[#d46b08]">新补</span>
                  ) : (
                    `#${it.source_index}`
                  )}
                  {' · '}
                  {KIND_LABEL[it.kind] ?? it.kind}
                  {' · '}
                  {TRANS_LABEL[it.transparency] ?? it.transparency}
                  {it.note ? (
                    <span className="text-black/45">（{it.note}）</span>
                  ) : null}
                </div>
              ))}
            </div>
            {panelAuditDeleted.length ? (
              <div className="flex flex-col gap-1 rounded border border-neutral-200 bg-neutral-50 p-2">
                <span className="text-[12px] font-bold">
                  已删除 {panelAuditDeleted.length} 条(附理由)
                </span>
                {panelAuditDeleted.map((del, i) => (
                  <div key={i} className="text-[12px] leading-relaxed">
                    <span className="font-bold text-[#cf1322]">#{del.source_index}</span>
                    {' '}
                    <span className="text-black/60">{del.reason}</span>
                  </div>
                ))}
              </div>
            ) : null}
            <span className="text-[11px] text-black/45">
              画框颜色:红=普通panel、橙=banner、蓝=标题;标签含 z 序与来源下标。
            </span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 py-4">
            {panelAuditStatus === 'error' ? (
              <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
                审核失败：{panelAuditError}
              </div>
            ) : null}
            <Button type="primary" disabled={!canRun} onClick={() => void runPanelAudit()}>
              {panelAuditStatus === 'error' ? '重试' : '审核panel'}
            </Button>
            <span className="px-2 text-center text-[12px] text-black/45">
              {!runInfo
                ? '请先上传图片'
                : midFillStatus !== 'done'
                  ? '请先完成第 15 步中景修补(需要 mid_fill.png)'
                  : panelCount === 0
                    ? '检测结果中没有 panel'
                    : `将把 ${panelCount} 个 panel(粗排层级后)交给模型审核修正` +
                      (consumedCount ? `;另有 ${consumedCount} 个已补进 bar,不参与` : '')}
            </span>
          </div>
        )}
      </div>
    </Card>
  )
}
