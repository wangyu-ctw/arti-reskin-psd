import { Button, Card, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'

/**
 * 第 16 步"提panel"(确定性,替代旧 16/17 生成式方案):
 * 逐 panel 用 SAM2 在 mid_fill 上抠可见部分 → z 序判定(包含→内者在上,
 * 相交→面积小者在上)→ 被上层压住的区域用 flux_fill(panel_fill LoRA)
 * 原位补全 → 补完后 SAM2 复测量 amodal 边界(带守门)。
 * 产物:run 目录 panel_extract/p<下标>.png + manifest.csv,几何零漂移。
 */
export default function PanelExtractPanel() {
  const {
    runInfo,
    structuredResult,
    midFillStatus,
    panelExtractStatus,
    panelExtractError,
    panelExtractInfo,
    panelExtractItems,
    panelExtractSourceSize,
    panelExtractTick,
    runPanelExtract,
  } = useDetectionStore()

  const panelCount = pickDetections(structuredResult, 'panel').length
  const canRun = Boolean(runInfo) && midFillStatus === 'done' && panelCount > 0

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
          {panelExtractStatus === 'done' ? (
            <Button size="small" disabled={!canRun} onClick={() => void runPanelExtract()}>
              重新提取
            </Button>
          ) : null}
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {panelExtractStatus === 'running' ? (
        <div className="flex flex-col items-center gap-3 py-6">
          <Spin />
          <span className="text-[12px] text-black/45">
            逐 panel 抠图,被压住的再补洞复测量…(每个补洞约 15s)
          </span>
        </div>
      ) : panelExtractStatus === 'done' ? (
        <div className="flex flex-col gap-3">
          <div className="text-[13px]">
            <span className="font-bold">完成:</span> {panelExtractInfo}
          </div>
          {panelExtractItems.length && panelExtractSourceSize ? (
            <div>
              <div className="mb-1 text-[12px] text-black/45">
                叠放预览（素材按原位 z 序回贴,应与去件图的 panel 完全重合）
              </div>
              <div
                className="relative w-full overflow-hidden rounded border border-neutral-200"
                style={{
                  aspectRatio: `${panelExtractSourceSize[0]} / ${panelExtractSourceSize[1]}`,
                  background:
                    'conic-gradient(#eee 25%, #fff 0 50%, #eee 0 75%, #fff 0) 0 0 / 16px 16px',
                }}
              >
                {[...panelExtractItems]
                  .filter((it) => it.file)
                  .sort((a, b) => a.z - b.z)
                  .map((it) => (
                    <img
                      key={it.panel_index}
                      src={`/api/runs/${runInfo?.run_id}/files/panel_extract/${it.file}?t=${panelExtractTick}`}
                      alt={it.file}
                      title={
                        `panel #${it.panel_index} · z${it.z}` +
                        ` · 隐藏 ${Math.round(it.hidden_ratio * 100)}%` +
                        (it.filled ? ' · 已补洞' : '') +
                        (it.remeasured ? ' · 已复测量' : '') +
                        (it.needs_review ? '（建议复核）' : '')
                      }
                      className={`absolute ${it.needs_review ? 'outline-2 outline-dashed outline-red-500' : ''}`}
                      style={{
                        left: `${(it.paste_x / panelExtractSourceSize[0]) * 100}%`,
                        top: `${(it.paste_y / panelExtractSourceSize[1]) * 100}%`,
                        width: `${(it.paste_w / panelExtractSourceSize[0]) * 100}%`,
                        height: `${(it.paste_h / panelExtractSourceSize[1]) * 100}%`,
                        objectFit: 'contain',
                      }}
                    />
                  ))}
              </div>
            </div>
          ) : null}
          <div className="text-[11px] leading-relaxed text-black/45">
            素材在 run 目录 panel_extract/ 下,文件名 p&lt;检测下标&gt;.png;manifest.csv
            记录回贴矩形、z 序、是否补洞/复测量。红色虚线框 = 隐藏占比超 60%,补全主要靠生成,
            建议人工核对。补洞挂 panel_fill LoRA,连通性约束只补与本体相连的遮挡区。
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 py-4">
          {panelExtractStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              提取失败：{panelExtractError}
            </div>
          ) : null}
          <Button type="primary" disabled={!canRun} onClick={() => void runPanelExtract()}>
            {panelExtractStatus === 'error' ? '重试' : '提取panel'}
          </Button>
          <span className="px-2 text-center text-[12px] text-black/45">
            {!runInfo
              ? '请先上传图片'
              : midFillStatus !== 'done'
                ? '请先完成第 15 步中景修补(需要 mid_fill.png)'
                : panelCount === 0
                  ? '检测结果中没有 panel'
                  : `将从 mid_fill 上确定性提取 ${panelCount} 个 panel(SAM2+z序+原位补洞)`}
          </span>
        </div>
      )}
    </Card>
  )
}
