import { useCallback, useEffect, useMemo, useRef } from 'react'
import { Card, Select } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { BBOX_TYPES, pickDetections } from '../lib/detection'

export default function BboxViewer() {
  const { bboxType, previewUrl, structuredResult, setField } = useDetectionStore()

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const imageUrlRef = useRef('')
  const scaleRef = useRef(1)
  const offsetXRef = useRef(0)
  const offsetYRef = useRef(0)
  const pointerIdRef = useRef<number | null>(null)
  const pointerXRef = useRef(0)
  const pointerYRef = useRef(0)

  const detections = useMemo(
    () => pickDetections(structuredResult, bboxType),
    [structuredResult, bboxType],
  )

  const paint = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const image = imageRef.current
    if (!image || imageUrlRef.current !== previewUrl) return

    const scale = scaleRef.current
    const offsetX = offsetXRef.current
    const offsetY = offsetYRef.current

    ctx.drawImage(
      image,
      offsetX,
      offsetY,
      canvas.width * scale,
      canvas.height * scale,
    )

    const canvasScale =
      canvas.width / Math.max(1, canvas.getBoundingClientRect().width)

    const drawPoints = (points: number[][] | undefined, color: string) => {
      if (!Array.isArray(points)) return
      ctx.fillStyle = color
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 1.5 * canvasScale
      for (const point of points) {
        if (!Array.isArray(point) || point.length < 2) continue
        const px = Number(point[0])
        const py = Number(point[1])
        if (![px, py].every(Number.isFinite)) continue
        ctx.beginPath()
        ctx.arc(
          offsetX + px * canvas.width * scale,
          offsetY + py * canvas.height * scale,
          5 * canvasScale,
          0,
          Math.PI * 2,
        )
        ctx.fill()
        ctx.stroke()
      }
    }

    for (const item of detections) {
      const [cx, cy, w, h] = item.bbox.map(Number)
      if (![cx, cy, w, h].every(Number.isFinite)) continue
      ctx.strokeStyle = '#ff3b30'
      ctx.lineWidth = 2 * canvasScale
      ctx.strokeRect(
        offsetX + (cx - w / 2) * canvas.width * scale,
        offsetY + (cy - h / 2) * canvas.height * scale,
        w * canvas.width * scale,
        h * canvas.height * scale,
      )
      drawPoints(item.positive_points, '#16a34a')
      drawPoints(item.negative_points, '#dc2626')
    }
  }, [detections, previewUrl])

  const resetTransform = useCallback(() => {
    scaleRef.current = 1
    offsetXRef.current = 0
    offsetYRef.current = 0
  }, [])

  // 加载 / 切换图片
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    if (!previewUrl) {
      imageRef.current = null
      imageUrlRef.current = ''
      resetTransform()
      ctx?.clearRect(0, 0, canvas.width, canvas.height)
      return
    }

    if (imageRef.current && imageUrlRef.current === previewUrl) {
      paint()
      return
    }

    let cancelled = false
    const sourceUrl = previewUrl
    const image = new Image()
    image.onload = () => {
      if (cancelled) return
      const maxDimension = 1600
      const imageScale = Math.min(
        1,
        maxDimension / Math.max(image.naturalWidth, image.naturalHeight),
      )
      canvas.width = Math.max(1, Math.round(image.naturalWidth * imageScale))
      canvas.height = Math.max(1, Math.round(image.naturalHeight * imageScale))
      imageRef.current = image
      imageUrlRef.current = sourceUrl
      resetTransform()
      paint()
    }
    image.src = sourceUrl
    return () => {
      cancelled = true
    }
  }, [previewUrl, paint, resetTransform])

  // 检测结果变化时重绘
  useEffect(() => {
    paint()
  }, [paint])

  // 滚轮缩放（需要 passive:false，因此手动绑定）
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const onWheel = (event: WheelEvent) => {
      if (!imageRef.current) return
      event.preventDefault()
      const rect = canvas.getBoundingClientRect()
      const pointerX =
        (event.clientX - rect.left) * (canvas.width / Math.max(1, rect.width))
      const pointerY =
        (event.clientY - rect.top) * (canvas.height / Math.max(1, rect.height))
      const nextScale = Math.min(
        12,
        Math.max(0.1, scaleRef.current * Math.exp(-event.deltaY * 0.0015)),
      )
      const ratio = nextScale / scaleRef.current
      offsetXRef.current = pointerX - (pointerX - offsetXRef.current) * ratio
      offsetYRef.current = pointerY - (pointerY - offsetYRef.current) * ratio
      scaleRef.current = nextScale
      paint()
    }

    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [paint])

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0 || !imageRef.current) return
    pointerIdRef.current = event.pointerId
    pointerXRef.current = event.clientX
    pointerYRef.current = event.clientY
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.pointerId !== pointerIdRef.current) return
    const canvas = event.currentTarget
    const rect = canvas.getBoundingClientRect()
    offsetXRef.current +=
      (event.clientX - pointerXRef.current) *
      (canvas.width / Math.max(1, rect.width))
    offsetYRef.current +=
      (event.clientY - pointerYRef.current) *
      (canvas.height / Math.max(1, rect.height))
    pointerXRef.current = event.clientX
    pointerYRef.current = event.clientY
    paint()
  }

  const stopDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.pointerId !== pointerIdRef.current) return
    pointerIdRef.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  const handleDoubleClick = () => {
    resetTransform()
    paint()
  }

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              5
            </span>
            查看边框
          </span>
          <Select
            className="w-25"
            value={bboxType}
            onChange={(v) => setField('bboxType', v)}
            options={BBOX_TYPES.map((t) => ({ value: t, label: t }))}
          />
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      <div className="mx-auto w-full max-w-[405px] overflow-hidden border border-neutral-200 bg-neutral-100 shadow-lg">
        <canvas
          ref={canvasRef}
          width={900}
          height={900}
          aria-label="可拖动和缩放的检测边框预览"
          className="block h-auto w-full cursor-grab touch-none"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={stopDrag}
          onPointerCancel={stopDrag}
          onDoubleClick={handleDoubleClick}
        />
      </div>
      <p className="mt-3 text-center text-[10px] text-black/45">
        滚轮缩放 · 拖动平移 · 双击复位
      </p>
    </Card>
  )
}
