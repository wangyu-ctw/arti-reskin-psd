import { useEffect, useMemo, useState } from 'react'
import { Modal } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import ZoomableCanvas from './ZoomableCanvas'

const CLASS_NAMES = ['text', 'icon', 'assets', 'button', 'bar', 'panel']

interface ParsedLine {
  raw: string
  lineNumber: number
  valid: boolean
  classId: number
  box: number[] // [cx, cy, w, h]
  confidence?: number
}

/** 与 yolo_result_viewer.html 相同的解析规则:跳过空行/#注释,前 5 个数字必须有效且宽高非负 */
function parseYolo(text: string): ParsedLine[] {
  const items: ParsedLine[] = []
  text.split(/\r?\n/).forEach((rawLine, index) => {
    const line = rawLine.trim()
    if (!line || line.startsWith('#')) return
    const tokens = line.split(/[\s,]+/)
    const values = tokens.slice(0, 5).map(Number)
    const confidence = Number(tokens[5])
    const valid =
      values.length === 5 &&
      values.every(Number.isFinite) &&
      values[3] >= 0 &&
      values[4] >= 0
    items.push({
      raw: rawLine,
      lineNumber: index + 1,
      valid,
      classId: valid ? values[0] : -1,
      box: valid ? values.slice(1) : [],
      confidence: Number.isFinite(confidence) ? confidence : undefined,
    })
  })
  return items
}

export default function YoloViewerModal({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const { yoloResult, previewUrl } = useDetectionStore()

  const items = useMemo(() => parseYolo(yoloResult), [yoloResult])
  const [disabledLines, setDisabledLines] = useState<Set<number>>(new Set())
  const [hoveredLine, setHoveredLine] = useState<number | null>(null)

  // 打开或结果变化时重置勾选/悬停状态
  useEffect(() => {
    setDisabledLines(new Set())
    setHoveredLine(null)
  }, [open, yoloResult])

  const validItems = useMemo(() => items.filter((it) => it.valid), [items])
  const errorLines = useMemo(
    () => items.filter((it) => !it.valid).map((it) => it.lineNumber),
    [items],
  )

  // 列表按左上角坐标排序:y 轴均分 16 带,先比带序,同带内按左边缘 x 升序;非法行放最后
  const sortedItems = useMemo(() => {
    const yBand = (it: ParsedLine) =>
      Math.min(15, Math.max(0, Math.floor((it.box[1] - it.box[3] / 2) * 16)))
    const xLeft = (it: ParsedLine) => it.box[0] - it.box[2] / 2
    return [
      ...[...validItems].sort((a, b) => yBand(a) - yBand(b) || xLeft(a) - xLeft(b)),
      ...items.filter((it) => !it.valid),
    ]
  }, [items, validItems])

  const boxes = useMemo(() => {
    if (hoveredLine != null) {
      const hovered = validItems.find((it) => it.lineNumber === hoveredLine)
      return hovered ? [hovered.box] : []
    }
    return validItems
      .filter((it) => !disabledLines.has(it.lineNumber))
      .map((it) => it.box)
  }, [validItems, disabledLines, hoveredLine])

  const enabledCount = validItems.filter(
    (it) => !disabledLines.has(it.lineNumber),
  ).length

  const toggleLine = (lineNumber: number, checked: boolean) => {
    setDisabledLines((prev) => {
      const next = new Set(prev)
      if (checked) next.delete(lineNumber)
      else next.add(lineNumber)
      return next
    })
  }

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width="80vw"
      style={{ top: 24, maxWidth: 1500 }}
      title={
        <div className="flex items-baseline gap-4">
          <span>YOLO 检测结果查看</span>
          <span
            className={`text-[13px] font-normal ${errorLines.length ? 'text-[#cf1322]' : 'text-black/45'}`}
          >
            {errorLines.length
              ? `${enabledCount} / ${validItems.length} 个框 · 第 ${errorLines.join('、')} 行错误`
              : `${enabledCount} / ${validItems.length} 个检测框`}
          </span>
        </div>
      }
    >
      <div className="flex gap-4" style={{ height: 'calc(100vh - 120px)' }}>
        <div className="min-w-0 w-[375px] overflow-auto">
          {previewUrl ? (
            <ZoomableCanvas
              src={previewUrl}
              alt="YOLO 检测框预览"
              boxes={boxes}
            />
          ) : (
            <div className="grid h-full place-items-center text-black/45">
              没有可用的原图,请先在第 1 步上传图片
            </div>
          )}
        </div>

        <div className="flex flex-1 shrink-0 flex-col rounded-lg border border-black/10 bg-[#fbfcfe]">
          <div className="flex-1 overflow-auto p-2">
            {items.length === 0 ? (
              <div className="grid h-full place-items-center text-center leading-7 text-black/45">
                暂无检测结果
                <br />
                请先在第 3 步生成或粘贴 YOLO 结果
              </div>
            ) : (
              sortedItems.map((item) => (
                <div
                  key={`${item.lineNumber}-${item.raw}`}
                  className={`grid grid-cols-[24px_52px_minmax(0,1fr)] items-start gap-2 rounded-md px-2 py-[7px] font-mono text-[13px] leading-[1.55] ${
                    item.valid
                      ? 'hover:bg-[#eaf1ff]'
                      : 'bg-[#fff3f3] text-[#cf1322]'
                  }`}
                  onMouseEnter={() => item.valid && setHoveredLine(item.lineNumber)}
                  onMouseLeave={() =>
                    setHoveredLine((prev) =>
                      prev === item.lineNumber ? null : prev,
                    )
                  }
                >
                  <input
                    type="checkbox"
                    className="mt-[3px] size-4 cursor-pointer accent-[#1677ff]"
                    checked={item.valid && !disabledLines.has(item.lineNumber)}
                    disabled={!item.valid}
                    onChange={(e) => toggleLine(item.lineNumber, e.target.checked)}
                    aria-label={`第 ${item.lineNumber} 行是否绘制`}
                  />
                  <span className="mt-[2px] truncate rounded bg-black/[0.06] px-1 text-center text-[11px] leading-[18px] text-black/60">
                    {CLASS_NAMES[item.classId] ?? '?'}
                  </span>
                  <span className="whitespace-pre-wrap break-all">{item.raw}</span>
                </div>
              ))
            )}
          </div>
          <div className="border-t border-black/10 px-3 py-2 text-xs leading-[1.5] text-black/45">
            悬停某一行可单独查看对应框;取消勾选可隐藏框。画布支持滚轮缩放、拖动平移、双击复位。
          </div>
        </div>
      </div>
    </Modal>
  )
}
