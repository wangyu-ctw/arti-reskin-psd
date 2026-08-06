import { useEffect, useMemo, useState } from 'react'
import { Modal } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import ZoomableCanvas from './ZoomableCanvas'

/**
 * 第 7 步分析结果查看:左侧去字图 + bbox 框 + 正负点(悬停行单独查看),
 * 右侧逐 icon 列表,附 bbox 像素宽高与正负点数量。
 */
export default function AnalysisViewerModal({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const analyzedIcons = useDetectionStore((s) => s.analyzedIcons)
  const textBackImageUrl = useDetectionStore((s) => s.textBackImageUrl)

  const [imgSize, setImgSize] = useState<[number, number] | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)

  useEffect(() => {
    if (!open) {
      setHovered(null)
      return
    }
    if (!textBackImageUrl) return
    const img = new Image()
    img.onload = () => setImgSize([img.naturalWidth, img.naturalHeight])
    img.src = textBackImageUrl
  }, [open, textBackImageUrl])

  const icons = analyzedIcons ?? []
  const shown = useMemo(
    () => (hovered != null ? icons.filter((i) => i.index === hovered) : icons),
    [icons, hovered],
  )
  const boxes = shown.map((i) => i.bbox)
  const posPoints = shown.flatMap((i) => i.positive_points ?? [])
  const negPoints = shown.flatMap((i) => i.negative_points ?? [])

  const pxSize = (bbox: number[]) => {
    if (!imgSize || !Array.isArray(bbox) || bbox.length < 4) return '—'
    return `${Math.round(bbox[2] * imgSize[0])} × ${Math.round(bbox[3] * imgSize[1])} px`
  }

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width="85vw"
      style={{ top: 24, maxWidth: 1500 }}
      title={
        <div className="flex items-baseline gap-4">
          <span>分析结果查看</span>
          <span className="text-[13px] font-normal text-black/45">
            {icons.length} 个 icon
            {imgSize ? ` · 去字图 ${imgSize[0]}×${imgSize[1]}px` : ''}
            · 绿=正点 红=负点
          </span>
        </div>
      }
    >
      <div className="flex gap-4" style={{ height: 'calc(100vh - 160px)' }}>
        <div className="w-[360px] overflow-auto">
          {textBackImageUrl ? (
            <ZoomableCanvas
              src={textBackImageUrl}
              alt="分析结果预览"
              boxes={boxes}
              posPoints={posPoints}
              negPoints={negPoints}
            />
          ) : (
            <div className="grid h-full place-items-center text-black/45">
              没有可用的去字图
            </div>
          )}
        </div>

        <div className="flex flex-1 shrink-0 flex-col rounded-lg border border-black/10 bg-[#fbfcfe]">
          <div className="flex-1 overflow-auto p-2">
            {icons.length === 0 ? (
              <div className="grid h-full place-items-center text-black/45">
                暂无分析结果
              </div>
            ) : (
              icons.map((icon) => (
                <div
                  key={icon.index}
                  className="rounded-md px-2 py-[7px] hover:bg-[#eaf1ff]"
                  onMouseEnter={() => setHovered(icon.index)}
                  onMouseLeave={() =>
                    setHovered((prev) => (prev === icon.index ? null : prev))
                  }
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span
                      className="min-w-0 truncate text-[13px]"
                      title={icon.description}
                    >
                      #{icon.index} {icon.description}
                    </span>
                    <span className="shrink-0 font-mono text-[12px] text-black/60">
                      {pxSize(icon.bbox)}
                    </span>
                  </div>
                  <div className="text-[11px] text-black/45">
                    正点 {icon.positive_points?.length ?? 0} · 负点{' '}
                    {icon.negative_points?.length ?? 0}
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="border-t border-black/10 px-3 py-2 text-xs leading-[1.5] text-black/45">
            悬停某一行可单独查看该 icon 的框与正负点。画布支持滚轮缩放、拖动平移、双击复位。
          </div>
        </div>
      </div>
    </Modal>
  )
}
