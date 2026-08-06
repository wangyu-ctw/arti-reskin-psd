import { useCallback, useEffect, useRef } from 'react'

/**
 * 可缩放/平移的图片画布(复用第 5 步查看边框的交互):
 * 滚轮缩放 · 拖动平移 · 双击复位。棋盘格衬底,适合展示透明 PNG。
 * 可选 boxes:归一化 YOLO 格式 [cx, cy, w, h] 的框数组,红框叠加显示。
 * 可选 posPoints/negPoints:归一化 [x, y] 点数组,绿/红圆点叠加(同第 5 步配色)。
 */
export default function ZoomableCanvas({
  src,
  alt,
  boxes,
  posPoints,
  negPoints,
}: {
  src: string
  alt?: string
  boxes?: number[][]
  posPoints?: number[][]
  negPoints?: number[][]
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const srcRef = useRef('')
  const scaleRef = useRef(1)
  const offsetXRef = useRef(0)
  const offsetYRef = useRef(0)
  const pointerIdRef = useRef<number | null>(null)
  const pointerXRef = useRef(0)
  const pointerYRef = useRef(0)

  const paint = useCallback(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const image = imageRef.current
    if (!image || srcRef.current !== src) return
    const scale = scaleRef.current
    const offsetX = offsetXRef.current
    const offsetY = offsetYRef.current
    ctx.drawImage(image, offsetX, offsetY, canvas.width * scale, canvas.height * scale)

    const canvasScale =
      canvas.width / Math.max(1, canvas.getBoundingClientRect().width)
    if (boxes?.length) {
      ctx.strokeStyle = '#ff3b30'
      ctx.lineWidth = 2 * canvasScale
      for (const box of boxes) {
        const [cx, cy, w, h] = box.map(Number)
        if (![cx, cy, w, h].every(Number.isFinite)) continue
        ctx.strokeRect(
          offsetX + (cx - w / 2) * canvas.width * scale,
          offsetY + (cy - h / 2) * canvas.height * scale,
          w * canvas.width * scale,
          h * canvas.height * scale,
        )
      }
    }
    const drawPoints = (points: number[][] | undefined, color: string) => {
      if (!points?.length) return
      ctx.fillStyle = color
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 1.5 * canvasScale
      for (const point of points) {
        const px = Number(point?.[0])
        const py = Number(point?.[1])
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
    drawPoints(posPoints, '#16a34a')
    drawPoints(negPoints, '#dc2626')
  }, [src, boxes, posPoints, negPoints])

  // 框集合变化时重绘
  useEffect(() => {
    paint()
  }, [paint])

  const resetTransform = useCallback(() => {
    scaleRef.current = 1
    offsetXRef.current = 0
    offsetYRef.current = 0
  }, [])

  // 加载/切换图片
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    if (!src) {
      imageRef.current = null
      srcRef.current = ''
      resetTransform()
      canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
      return
    }
    let cancelled = false
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
      srcRef.current = src
      resetTransform()
      paint()
    }
    image.src = src
    return () => {
      cancelled = true
    }
  }, [src, paint, resetTransform])

  // 滚轮缩放(需要 passive:false,手动绑定)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const onWheel = (event: WheelEvent) => {
      if (!imageRef.current) return
      event.preventDefault()
      const rect = canvas.getBoundingClientRect()
      const px = (event.clientX - rect.left) * (canvas.width / Math.max(1, rect.width))
      const py = (event.clientY - rect.top) * (canvas.height / Math.max(1, rect.height))
      const next = Math.min(
        12,
        Math.max(0.1, scaleRef.current * Math.exp(-event.deltaY * 0.0015)),
      )
      const ratio = next / scaleRef.current
      offsetXRef.current = px - (px - offsetXRef.current) * ratio
      offsetYRef.current = py - (py - offsetYRef.current) * ratio
      scaleRef.current = next
      paint()
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [paint])

  return (
    <div
      className="w-full overflow-hidden border border-neutral-200 shadow-lg"
      style={{
        backgroundImage:
          'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
        backgroundSize: '16px 16px',
      }}
    >
      <canvas
        ref={canvasRef}
        width={900}
        height={900}
        aria-label={alt ?? '可拖动和缩放的图片预览'}
        className="block h-auto w-full cursor-grab touch-none"
        onPointerDown={(event) => {
          if (event.button !== 0 || !imageRef.current) return
          pointerIdRef.current = event.pointerId
          pointerXRef.current = event.clientX
          pointerYRef.current = event.clientY
          event.currentTarget.setPointerCapture(event.pointerId)
        }}
        onPointerMove={(event) => {
          if (event.pointerId !== pointerIdRef.current) return
          const canvas = event.currentTarget
          const rect = canvas.getBoundingClientRect()
          offsetXRef.current +=
            (event.clientX - pointerXRef.current) * (canvas.width / Math.max(1, rect.width))
          offsetYRef.current +=
            (event.clientY - pointerYRef.current) * (canvas.height / Math.max(1, rect.height))
          pointerXRef.current = event.clientX
          pointerYRef.current = event.clientY
          paint()
        }}
        onPointerUp={(event) => {
          if (event.pointerId !== pointerIdRef.current) return
          pointerIdRef.current = null
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId)
          }
        }}
        onPointerCancel={(event) => {
          if (event.pointerId !== pointerIdRef.current) return
          pointerIdRef.current = null
        }}
        onDoubleClick={() => {
          resetTransform()
          paint()
        }}
      />
    </div>
  )
}
