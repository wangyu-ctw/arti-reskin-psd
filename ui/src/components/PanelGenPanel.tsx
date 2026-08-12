import { Button, Card, Input, Select, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'
import stepDefaults from '../config/stepDefaults'

const { TextArea } = Input

// 绿底图看边缘用浅色衬底即可
const PREVIEW_BG: React.CSSProperties = {
  background:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

/**
 * 第 16 步"提panel":把第 14 步修补图 + 红框标注版 + panel bbox 一起交给
 * nano banana,在纯绿底上平铺生成所有 panel(不要背景),对称参考补全缺边。
 * 结果静默上传 pod(panels_green.png)。
 */
export default function PanelGenPanel() {
  const {
    runInfo,
    structuredResult,
    midFillStatus,
    panelGenModel,
    panelGenPrompt,
    panelGenStatus,
    panelGenError,
    panelGenLayers,
    panelGenLayerUrls,
    runPanelGen,
    setField,
  } = useDetectionStore()

  const panelCount = pickDetections(structuredResult, 'panel').length
  const canRun =
    Boolean(runInfo) && midFillStatus === 'done' && panelCount > 0

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              16
            </span>
            提panel
          </span>
          {panelGenStatus === 'done' ? (
            <Button size="small" disabled={!canRun} onClick={() => void runPanelGen()}>
              重新生成
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
          <div className="mb-1 text-[11px] leading-snug text-black/40">
            GPT Image 系,走 OpenRouter,按张计费
          </div>
          <Select
            className="w-full"
            value={panelGenModel}
            onChange={(v) => setField('panelGenModel', v)}
            disabled={panelGenStatus === 'running'}
            options={stepDefaults.panelGen.models.map((id: string) => ({
              value: id,
              label: id.split('/').pop(),
            }))}
          />
        </div>
        <div>
          <div className="text-[13px] font-bold">提示词</div>
          <TextArea
            value={panelGenPrompt}
            onChange={(e) => setField('panelGenPrompt', e.target.value)}
            disabled={panelGenStatus === 'running'}
            rows={7}
          />
        </div>

        {panelGenStatus === 'running' ? (
          <div className="flex flex-col gap-3 py-2">
            <div className="flex items-center justify-center gap-3">
              <Spin size="small" />
              <span className="text-[12px] text-black/45">
                正在逐层生成…已完成 {panelGenLayerUrls.length} 层
              </span>
            </div>
            {panelGenLayerUrls.map((url, k) => (
              <div key={k} style={PREVIEW_BG}>
                <img src={url} alt={`层 ${k + 1}`} className="h-auto max-w-full object-contain" />
              </div>
            ))}
          </div>
        ) : panelGenStatus === 'done' && panelGenLayerUrls.length ? (
          <div className="flex flex-col gap-3">
            {panelGenLayerUrls.map((url, k) => (
              <div key={k} className="flex flex-col gap-1">
                <span className="text-[12px] font-bold">
                  第 {k + 1} 层（{panelGenLayers[k]?.length ?? 0} 个 panel:#
                  {(panelGenLayers[k] ?? []).join(' #')}）
                </span>
                <div style={PREVIEW_BG}>
                  <img
                    src={url}
                    alt={`panel 层 ${k + 1}`}
                    className="h-auto max-w-full object-contain"
                  />
                </div>
              </div>
            ))}
            <div className="text-[11px] text-black/45">
              分层原位生成(无重叠最少层);已静默上传 pod(panels_green_L*.png)。
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 py-4">
            {panelGenStatus === 'error' ? (
              <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
                生成失败：{panelGenError}
              </div>
            ) : null}
            <Button type="primary" disabled={!canRun} onClick={() => void runPanelGen()}>
              {panelGenStatus === 'error' ? '重试' : '生成'}
            </Button>
            <span className="px-2 text-center text-[12px] text-black/45">
              {!runInfo
                ? '请先上传图片'
                : midFillStatus !== 'done'
                  ? '请先完成第 14 步修补(需要 mid_fill.png)'
                  : panelCount === 0
                    ? '检测结果中没有 panel'
                    : `将把 ${panelCount} 个 panel 平铺到纯绿底(保比例、留间距、精修补边)`}
            </span>
          </div>
        )}
      </div>
    </Card>
  )
}
