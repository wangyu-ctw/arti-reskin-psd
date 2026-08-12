import { Button, Card, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'

/**
 * 第 17 步"panel素材":对第 16 步的绿底平铺图做色键切割 + 连通域计数审计,
 * 用"长宽比 + 色彩网格"双特征把素材匹配回原 panel 位置。
 * 产物:run 目录 panel_assets/p<下标>.png + manifest.csv(回贴矩形与匹配代价)。
 */
export default function PanelAssetPanel() {
  const {
    runInfo,
    structuredResult,
    panelGenStatus,
    panelAssetStatus,
    panelAssetError,
    panelAssetInfo,
    panelAssetItems,
    panelAssetSourceSize,
    panelAssetTick,
    runPanelAsset,
  } = useDetectionStore()

  const panelCount = pickDetections(structuredResult, 'panel').length
  const canRun =
    Boolean(runInfo) && panelGenStatus === 'done' && panelCount > 0

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              17
            </span>
            panel素材
          </span>
          {panelAssetStatus === 'done' ? (
            <Button size="small" disabled={!canRun} onClick={() => void runPanelAsset()}>
              重新切割
            </Button>
          ) : null}
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {panelAssetStatus === 'running' ? (
        <div className="flex flex-col items-center gap-3 py-6">
          <Spin />
          <span className="text-[12px] text-black/45">
            正在切割匹配…(纯 CPU 计算,通常几秒)
          </span>
        </div>
      ) : panelAssetStatus === 'done' ? (
        <div className="flex flex-col gap-3">
          <div className="text-[13px]">
            <span className="font-bold">完成:</span> {panelAssetInfo}
          </div>
          {panelAssetItems.length && panelAssetSourceSize ? (
            <div>
              <div className="mb-1 text-[12px] text-black/45">
                叠放预览（素材按匹配到的原 panel 位置等比回贴）
              </div>
              <div
                className="relative w-full overflow-hidden rounded border border-neutral-200"
                style={{
                  aspectRatio: `${panelAssetSourceSize[0]} / ${panelAssetSourceSize[1]}`,
                  background:
                    'conic-gradient(#eee 25%, #fff 0 50%, #eee 0 75%, #fff 0) 0 0 / 16px 16px',
                }}
              >
                {[...panelAssetItems]
                  .sort((a, b) => (a.z ?? 0) - (b.z ?? 0))
                  .map((it) => (
                  <img
                    key={it.panel_index}
                    src={`/api/runs/${runInfo?.run_id}/files/panel_assets/${it.file}?t=${panelAssetTick}`}
                    alt={it.file}
                    title={`panel #${it.panel_index} · z${it.z ?? 0} · cost ${it.cost}${it.uncertain ? '（低置信）' : ''}`}
                    className={`absolute ${it.uncertain ? 'outline-2 outline-dashed outline-red-500' : ''}`}
                    style={{
                      left: `${(it.paste_x / panelAssetSourceSize[0]) * 100}%`,
                      top: `${(it.paste_y / panelAssetSourceSize[1]) * 100}%`,
                      width: `${(it.paste_w / panelAssetSourceSize[0]) * 100}%`,
                      height: `${(it.paste_h / panelAssetSourceSize[1]) * 100}%`,
                      objectFit: 'contain',
                    }}
                  />
                ))}
              </div>
            </div>
          ) : null}
          <div className="text-[11px] leading-relaxed text-black/45">
            素材在 run 目录 panel_assets/ 下,文件名 p&lt;检测下标&gt;.png;
            manifest.csv 记录回贴矩形与匹配代价;红色虚线框 = 低置信/复核未过,建议人工核对;叠放按 z 序渲染。出边走色键软阈值(SAM2 模式已停用,后端 mask_mode 可切回)。
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 py-4">
          {panelAssetStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              切割匹配失败：{panelAssetError}
            </div>
          ) : null}
          <Button type="primary" disabled={!canRun} onClick={() => void runPanelAsset()}>
            {panelAssetStatus === 'error' ? '重试' : '切割匹配'}
          </Button>
          <span className="px-2 text-center text-[12px] text-black/45">
            {!runInfo
              ? '请先上传图片'
              : panelGenStatus !== 'done'
                ? '请先完成第 16 步提panel(需要 panels_green.png)'
                : panelCount === 0
                  ? '检测结果中没有 panel'
                  : `将切割绿底平铺图并匹配回 ${panelCount} 个 panel 原位`}
          </span>
        </div>
      )}
    </Card>
  )
}
