import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { Modal } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'
import {
  cropByBbox,
  downscaleImage,
  greenFullPrompt,
  loadImage,
  qwenGreenRefine,
  removeGreen,
  QWEN_IMAGE_MODEL,
} from '../lib/qwenIconRefine'

const CHECKERBOARD: CSSProperties = {
  backgroundImage:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

interface RowState {
  index: number
  bbox: number[]
  source: string // 去字图裁块
  sam2?: string // SAM2 结果同框裁块
  qwen?: string // qwen 绿底精修 + 抠绿
  error?: string
}

/** 绿底精修对比实验:去字图裁块 → qwen-image-3 绿底重绘 → 抠绿 → 与 SAM2 并排。 */
export default function GreenCompareModal({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const apiKey = useDetectionStore((s) => s.apiKey)
  const textBackImageUrl = useDetectionStore((s) => s.textBackImageUrl)
  const iconImageUrl = useDetectionStore((s) => s.iconImageUrl)
  const analyzedIcons = useDetectionStore((s) => s.analyzedIcons)
  const structuredResult = useDetectionStore((s) => s.structuredResult)

  const [rows, setRows] = useState<RowState[]>([])
  const [status, setStatus] = useState('')
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!open) {
      abortRef.current?.abort()
      return
    }
    const bboxes: number[][] = analyzedIcons?.length
      ? analyzedIcons.filter((a) => !a.should_delete).map((a) => a.bbox)
      : pickDetections(structuredResult, 'icon').map((d) => d.bbox)
    if (!bboxes.length || !textBackImageUrl) {
      setStatus('没有可用的 icon 检测框或去字图')
      setRows([])
      return
    }
    const abort = new AbortController()
    abortRef.current = abort

    void (async () => {
      setStatus('准备裁块…')
      const srcImg = await loadImage(textBackImageUrl)
      const sam2Img = iconImageUrl ? await loadImage(iconImageUrl) : null
      const initial: RowState[] = bboxes.map((bbox, i) => ({
        index: i,
        bbox,
        source: cropByBbox(srcImg, bbox),
        sam2: sam2Img ? cropByBbox(sam2Img, bbox) : undefined,
      }))
      setRows(initial)

      try {
        // 单次整图调用:除 icon 外全部涂绿,再按已知 bbox 本地裁切,费用 = 1 次生成
        setStatus('整图绿底生成中…(单次调用,约 20~60 秒)')
        const green = await qwenGreenRefine(
          apiKey,
          downscaleImage(srcImg),
          greenFullPrompt(initial.length),
          abort.signal,
        )
        if (abort.signal.aborted) return
        setStatus('裁切抠绿中…')
        const greenImg = await loadImage(green)
        for (const row of initial) {
          const rgba = await removeGreen(cropByBbox(greenImg, row.bbox))
          if (abort.signal.aborted) return
          setRows((prev) =>
            prev.map((r) => (r.index === row.index ? { ...r, qwen: rgba } : r)),
          )
        }
        setStatus(`完成 ${initial.length} 个(1 次生成)`)
      } catch (error) {
        if (abort.signal.aborted) return
        const msg = error instanceof Error ? error.message : '生成失败'
        setStatus(`失败:${msg}`)
        setRows((prev) => prev.map((r) => (r.qwen ? r : { ...r, error: msg })))
      }
    })()

    return () => abort.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width="80vw"
      style={{ top: 24, maxWidth: 1400 }}
      title={
        <div className="flex items-baseline gap-4">
          <span>绿底精修对比({QWEN_IMAGE_MODEL})</span>
          <span className="text-[13px] font-normal text-black/45">{status}</span>
        </div>
      }
    >
      <div className="overflow-auto" style={{ maxHeight: 'calc(100vh - 160px)' }}>
        <div className="mb-2 grid grid-cols-[40px_1fr_1fr_1fr] gap-2 text-[12px] font-bold text-black/60">
          <span>#</span>
          <span>去字图裁块(输入)</span>
          <span>SAM2 提取(同框)</span>
          <span>qwen 绿底精修(抠绿后)</span>
        </div>
        {rows.map((row) => (
          <div
            key={row.index}
            className="mb-2 grid grid-cols-[40px_1fr_1fr_1fr] items-center gap-2 border-b border-black/5 pb-2"
          >
            <span className="text-[12px] text-black/45">{row.index}</span>
            <img src={row.source} alt="输入裁块" className="h-auto max-h-40 max-w-full object-contain" />
            {row.sam2 ? (
              <div style={CHECKERBOARD}>
                <img src={row.sam2} alt="SAM2 结果" className="h-auto max-h-40 max-w-full object-contain" />
              </div>
            ) : (
              <span className="text-[12px] text-black/30">(未提取)</span>
            )}
            {row.qwen ? (
              <div style={CHECKERBOARD}>
                <img src={row.qwen} alt="qwen 精修结果" className="h-auto max-h-40 max-w-full object-contain" />
              </div>
            ) : row.error ? (
              <span className="break-all text-[12px] text-[#cf1322]">{row.error}</span>
            ) : (
              <span className="text-[12px] text-black/30">生成中…</span>
            )}
          </div>
        ))}
      </div>
    </Modal>
  )
}
