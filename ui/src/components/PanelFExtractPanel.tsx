import type { CSSProperties } from 'react'
import { Button, Card, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'

const CHECKERBOARD: CSSProperties = {
  backgroundImage:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

/**
 * 第 9 步"提取panel_f":SAM2 把结构检测出的前景夹层面板(panel_f)
 * 从去字图上抠出(参数复用第 8 步 icon 中档)。产物 panel_f.png;
 * 第 10 步去icon挖洞时会把这层的 alpha 一并并入 mask。
 */
export default function PanelFExtractPanel() {
  const {
    runInfo,
    structuredResult,
    textBackStatus,
    panelFStatus,
    panelFImageUrl,
    panelFError,
    runPanelF,
  } = useDetectionStore()

  const count = pickDetections(structuredResult, 'panel_f').length
  const canRun = Boolean(runInfo) && textBackStatus === 'done' && count > 0

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              9
            </span>
            提取panel_f
          </span>
          {panelFStatus === 'done' ? (
            <Button size="small" disabled={!canRun} onClick={() => void runPanelF()}>
              重新提取
            </Button>
          ) : null}
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {panelFStatus === 'running' ? (
        <div className="flex flex-col items-center gap-3 py-6">
          <Spin />
          <span className="text-[12px] text-black/45">
            正在提取 panel_f…(SAM2,icon 中档参数)
          </span>
        </div>
      ) : panelFStatus === 'done' ? (
        <div className="flex flex-col gap-3">
          <div style={CHECKERBOARD} className="rounded border border-neutral-200">
            <img
              src={panelFImageUrl}
              alt="panel_f 提取层"
              className="h-auto max-w-full object-contain"
            />
          </div>
          <div className="text-[11px] leading-relaxed text-black/45">
            前景夹层面板已抠出(panel_f.png)。第 10 步去icon挖洞时会把这层
            alpha 与 icons.png 取并集一起移除;重新提取会使已生成的去icon图过期。
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 py-4">
          {panelFStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              提取失败：{panelFError}
            </div>
          ) : null}
          <Button type="primary" disabled={!canRun} onClick={() => void runPanelF()}>
            {panelFStatus === 'error' ? '重试' : '提取'}
          </Button>
          <span className="px-2 text-center text-[12px] text-black/45">
            {!runInfo
              ? '请先上传图片'
              : textBackStatus !== 'done'
                ? '请先完成第 2 步去文字'
                : count === 0
                  ? '检测结果中没有 panel_f,本步可跳过(第 10 步不带此层)'
                  : `将从去字图提取 ${count} 个 panel_f(前景夹层面板)`}
          </span>
        </div>
      )}
    </Card>
  )
}
