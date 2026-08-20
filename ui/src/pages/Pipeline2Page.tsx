import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import {
  Button, Card, Checkbox, Input, InputNumber, Popover, Select, Spin, Tag, Upload, message,
} from 'antd'
import { DeleteOutlined, SettingOutlined, UploadOutlined } from '@ant-design/icons'
import { usePipeline2Store, type P2Element } from '../stores/usePipeline2Store'
import { REASONING_OPTIONS, SPEED_OPTIONS } from '../lib/detection'
import { ImageCompareSlider } from '../components/CompareSliderPanel'

const CHECKERBOARD: CSSProperties = {
  backgroundImage:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

const CLS_COLORS: Record<string, string> = {
  text: '#52c41a', icon: '#faad14', button: '#ff4d4f', bar: '#eb2f96',
  assets: '#40a9ff', panel: '#9254de', panel_f: '#722ed1', unknown: '#8c8c8c',
}

function StepCard({
  step, title, titleInfo, action, disabled, wide, settings, children,
}: {
  step: string
  title: string
  /** 标题栏里的结果摘要/查看按钮 */
  titleInfo?: React.ReactNode
  action: () => void
  disabled: boolean
  wide?: boolean
  /** 配置 popover 内容(说明文字 + 可调参数) */
  settings?: React.ReactNode
  children?: React.ReactNode
}) {
  const status = usePipeline2Store((s) => s.status[step] ?? 'idle')
  const error = usePipeline2Store((s) => s.errors[step] ?? '')
  return (
    <div className={`h-full shrink-0 ${wide ? 'w-auto min-w-110' : 'w-110'}`}>
      <Card
        title={
          <div className="flex items-center justify-between gap-3">
            <span className="text-[15px] font-bold">{title}</span>
            <div className="flex items-center gap-2">
              {titleInfo}
              {status === 'done' ? <Tag color="green">完成</Tag> : null}
              {settings ? (
                <Popover trigger="hover" placement="bottomRight" content={settings}>
                  <SettingOutlined className="cursor-pointer text-[15px] text-black/45 hover:text-[#1677ff]" />
                </Popover>
              ) : null}
              <Button
                type="primary"
                size="small"
                loading={status === 'running'}
                disabled={disabled}
                onClick={action}
              >
                {status === 'done' ? '重跑' : '执行'}
              </Button>
            </div>
          </div>
        }
        className="flex h-full w-full flex-col shadow-sm"
        styles={{ body: { overflow: 'auto', flex: 1 } }}
      >
        {status === 'error' ? (
          <div className="mb-2 max-w-160 break-all text-[12px] text-[#cf1322]">{error}</div>
        ) : null}
        {children}
      </Card>
    </div>
  )
}

function NumField({ label, value, onChange, step = 1, min, max }: {
  label: string
  value: number
  onChange: (v: number) => void
  step?: number
  min?: number
  max?: number
}) {
  return (
    <div>
      <div className="mb-0.5 text-[11px] text-black/50">{label}</div>
      <InputNumber
        className="w-full"
        size="small"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(v) => onChange((v ?? 0) as number)}
      />
    </div>
  )
}

function LayerRow({ layers, runId }: {
  layers: { name: string; file: string; coverage: number; keep: boolean }[]
  runId: string
}) {
  if (!layers.length) return null
  return (
    <div className="flex flex-nowrap gap-2">
      {layers.map((l) => (
        <div key={l.name} className="w-56 shrink-0">
          <div className="mb-0.5 flex items-center gap-1 text-[11px]">
            <span className="font-bold">{l.name}</span>
            <span className="text-black/40">{(l.coverage * 100).toFixed(1)}%</span>
            {!l.keep ? <Tag className="ml-auto">空,弃</Tag> : null}
          </div>
          <div style={CHECKERBOARD} className="rounded border border-neutral-200">
            <img
              src={`/api/runs/${runId}/files/${l.file}`}
              alt={l.name}
              loading="lazy"
              className={`h-auto w-full ${l.keep ? '' : 'opacity-30'}`}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

export function ElementTable({ elements, hoverLink }: {
  elements: P2Element[]
  hoverLink?: boolean
}) {
  const setHovered = usePipeline2Store((s) => s.setHoveredElement)
  if (!elements.length) return null
  const bySlot = new Map<string, P2Element[]>()
  for (const e of elements) {
    const arr = bySlot.get(e.sourceLayer) ?? []
    arr.push(e)
    bySlot.set(e.sourceLayer, arr)
  }
  return (
    <div className="flex flex-col gap-1 text-[12px]">
      {[...bySlot.entries()].map(([slot, els]) => (
        <div key={slot} className="flex flex-wrap items-center gap-1">
          <span className="w-16 font-bold">{slot}</span>
          <span className="text-black/45">{els.length} 个:</span>
          {els.map((e) => {
            const valid = Boolean(hoverLink && e.extract && !e.mergedInto)
            return (
              <Tag
                key={e.id}
                className={valid ? 'cursor-pointer' : undefined}
                onMouseEnter={valid ? () => setHovered(e.id) : undefined}
                onMouseLeave={valid ? () => setHovered(null) : undefined}
                color={e.skip ? 'default' : e.mergedInto ? 'orange' : e.extract ? 'green' : 'blue'}
                title={`${e.cls}${e.type && e.type !== e.cls ? `→${e.type}` : ''}` +
                  `${e.cover && e.cover !== 'none' ? ` cover:${e.cover}` : ''}` +
                  `${e.mergedInto ? ` 并入${e.mergedInto}` : ''}` +
                  `${e.extract ? ` [${e.extract.method}]` : ''}`}
              >
                {e.id}
              </Tag>
            )
          })}
        </div>
      ))}
      <div className="mt-1 text-[11px] text-black/40">
        蓝=待抠 · 绿=已抠 · 橙=过拆并回 · 灰=cover 跳过
        {hoverLink ? ';悬停绿色素材 → 拼回图单层预览' : ''}
      </div>
    </div>
  )
}

/** ⑤ 的"层素材"行:panelz 各 z 层 + panel_f 整层,hover 在 ⑥ 单层预览 */
export function LayerAssetTags() {
  const panelzLayers = usePipeline2Store((s) => s.panelzLayers)
  const slotLayers = usePipeline2Store((s) => s.slotLayers)
  const setHovered = usePipeline2Store((s) => s.setHoveredElement)
  const items = [
    ...panelzLayers
      .filter((l) => l.keep && l.name !== 'bg')
      .map((l) => ({ label: `panel/${l.name}`, file: l.file })),
    ...slotLayers
      .filter((l) => l.keep && l.name === 'panel_f')
      .map((l) => ({ label: 'panel_f 层', file: l.file })),
  ]
  if (!items.length) return null
  return (
    <div className="mt-1 flex flex-wrap items-center gap-1 text-[12px]">
      <span className="w-16 font-bold">层素材</span>
      {items.map((it) => (
        <Tag
          key={it.file}
          color="purple"
          className="cursor-pointer"
          onMouseEnter={() => setHovered(`layer:${it.file}`)}
          onMouseLeave={() => setHovered(null)}
        >
          {it.label}
        </Tag>
      ))}
    </div>
  )
}

/** ⓪ 卡片内的批量执行:选多图 → 每图独立 run 跑全链,进度逐行展示 */
function BatchPanel() {
  const batch = usePipeline2Store((st) => st.batch)
  const runBatch = usePipeline2Store((st) => st.runBatch)
  const apiKey = usePipeline2Store((st) => st.apiKey)
  return (
    <div className="mt-3 flex flex-col gap-1.5 border-t border-black/10 pt-2">
      <div className="flex items-center justify-between">
        <span className="text-[12px] font-bold text-black/65">
          批量执行(每图全链 ⓪⁺→④,逐图串行)
        </span>
        <Upload
          multiple
          accept="image/*"
          showUploadList={false}
          beforeUpload={(file, fileList) => {
            if (file === fileList[0]) {
              if (!apiKey.trim()) {
                message.error('缺少 OpenRouter API Key(② 审核需要),在 ⓪ 配置里填')
              } else {
                void runBatch([...fileList])
              }
            }
            return Upload.LIST_IGNORE
          }}
        >
          <Button size="small" type="primary" ghost loading={batch.running}>
            批量
          </Button>
        </Upload>
      </div>
      {batch.items.map((it, i) => (
        <div key={`${i}-${it.name}`} className="flex items-center gap-2 text-[11px]">
          <span className={
            it.status === 'error' ? 'text-[#cf1322]'
              : it.status === 'done' ? 'text-[#389e0d]' : 'text-[#1677ff]'
          }>
            {it.status === 'error' ? '✗' : it.status === 'done' ? '✓' : '…'}
          </span>
          <span className="max-w-36 truncate" title={it.name}>{it.name}</span>
          <span className="select-all font-mono text-black/60">
            {it.runId || '—'}
          </span>
          <span
            className="max-w-56 truncate text-black/45"
            title={it.status === 'error' ? it.error : it.step}
          >
            {it.status === 'error' ? it.error : it.step}
          </span>
          {it.status === 'done' ? (
            <span className="text-black/45">
              素材{it.assets} · PSD{it.psdLayers}层 · 误差{it.diffMean}
            </span>
          ) : null}
        </div>
      ))}
    </div>
  )
}

// ---- 图片缓存 ----

const imgCache = new Map<string, Promise<HTMLImageElement>>()
function loadImg(url: string): Promise<HTMLImageElement> {
  let p = imgCache.get(url)
  if (!p) {
    p = new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = () => {
        imgCache.delete(url)
        reject(new Error(`加载失败 ${url}`))
      }
      img.src = url
    })
    imgCache.set(url, p)
  }
  return p
}

function bboxPx(bbox: number[], w: number, h: number): [number, number, number, number] {
  const [cx, cy, bw, bh] = bbox
  return [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]
}

interface BoxItem {
  id: string
  cls: string
  bbox: number[]
}

/** ⓪⁺ 查看器:仿旧管线第 5 步——单类别下拉,滚轮缩放/拖动平移/双击复位 */
function PanZoomBoxViewer({ items, imageFile = 'origin.png', allClasses }: {
  items: BoxItem[]
  imageFile?: string
  /** true=不做类别筛选,全部框按类别着色同屏展示 */
  allClasses?: boolean
}) {
  const runId = usePipeline2Store((s) => s.runId)
  const allCls = useMemo(() => [...new Set(items.map((e) => e.cls))], [items])
  const [cls, setCls] = useState<string>('')
  const active = cls || allCls[0] || ''
  const shown = useMemo(
    () => (allClasses ? items : items.filter((e) => e.cls === active)),
    [items, active, allClasses])

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const scaleRef = useRef(1)
  const offXRef = useRef(0)
  const offYRef = useRef(0)
  const ptrRef = useRef<{ id: number; x: number; y: number } | null>(null)

  const paint = () => {
    const canvas = canvasRef.current
    const img = imageRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const sc = scaleRef.current
    const ox = offXRef.current
    const oy = offYRef.current
    ctx.drawImage(img, ox, oy, canvas.width * sc, canvas.height * sc)
    const cs = canvas.width / Math.max(1, canvas.getBoundingClientRect().width)
    ctx.lineWidth = 2 * cs
    for (const e of shown) {
      ctx.strokeStyle = CLS_COLORS[allClasses ? e.cls : active] ?? '#ff3b30'
      const [cx, cy, w, h] = e.bbox
      ctx.strokeRect(
        ox + (cx - w / 2) * canvas.width * sc,
        oy + (cy - h / 2) * canvas.height * sc,
        w * canvas.width * sc,
        h * canvas.height * sc,
      )
    }
  }
  const paintRef = useRef(paint)
  paintRef.current = paint

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !runId) return
    let cancelled = false
    void loadImg(`/api/runs/${runId}/files/${imageFile}`).then((img) => {
      if (cancelled) return
      const maxDim = 1600
      const k = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight))
      canvas.width = Math.max(1, Math.round(img.naturalWidth * k))
      canvas.height = Math.max(1, Math.round(img.naturalHeight * k))
      imageRef.current = img
      scaleRef.current = 1
      offXRef.current = 0
      offYRef.current = 0
      paintRef.current()
    })
    return () => {
      cancelled = true
    }
  }, [runId, imageFile])

  useEffect(() => {
    paintRef.current()
  }, [shown])

  // 滚轮缩放需要 passive:false
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const onWheel = (ev: WheelEvent) => {
      if (!imageRef.current) return
      ev.preventDefault()
      const rect = canvas.getBoundingClientRect()
      const px = (ev.clientX - rect.left) * (canvas.width / Math.max(1, rect.width))
      const py = (ev.clientY - rect.top) * (canvas.height / Math.max(1, rect.height))
      const next = Math.min(12, Math.max(0.1, scaleRef.current * Math.exp(-ev.deltaY * 0.0015)))
      const r = next / scaleRef.current
      offXRef.current = px - (px - offXRef.current) * r
      offYRef.current = py - (py - offYRef.current) * r
      scaleRef.current = next
      paintRef.current()
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [])

  return (
    <div className="flex flex-col gap-2">
      {!allClasses ? (
        <Select
          className="w-40"
          size="small"
          value={active}
          onChange={setCls}
          options={allCls.map((c) => ({
            value: c,
            label: `${c}(${items.filter((e) => e.cls === c).length})`,
          }))}
        />
      ) : (
        <div className="flex flex-wrap gap-1 text-[11px]">
          {allCls.map((c) => (
            <span key={c} style={{ color: CLS_COLORS[c] ?? '#000' }}>
              {c}({items.filter((e) => e.cls === c).length})
            </span>
          ))}
        </div>
      )}
      <div style={CHECKERBOARD} className="w-100 overflow-hidden rounded border border-neutral-200">
        <canvas
          ref={canvasRef}
          className="block h-auto w-full cursor-grab touch-none"
          onPointerDown={(ev) => {
            if (ev.button !== 0 || !imageRef.current) return
            ptrRef.current = { id: ev.pointerId, x: ev.clientX, y: ev.clientY }
            ev.currentTarget.setPointerCapture(ev.pointerId)
          }}
          onPointerMove={(ev) => {
            const p = ptrRef.current
            if (!p || ev.pointerId !== p.id) return
            const canvas = ev.currentTarget
            const rect = canvas.getBoundingClientRect()
            offXRef.current += (ev.clientX - p.x) * (canvas.width / Math.max(1, rect.width))
            offYRef.current += (ev.clientY - p.y) * (canvas.height / Math.max(1, rect.height))
            p.x = ev.clientX
            p.y = ev.clientY
            paintRef.current()
          }}
          onPointerUp={(ev) => {
            if (ptrRef.current?.id === ev.pointerId) ptrRef.current = null
          }}
          onPointerCancel={(ev) => {
            if (ptrRef.current?.id === ev.pointerId) ptrRef.current = null
          }}
          onDoubleClick={() => {
            scaleRef.current = 1
            offXRef.current = 0
            offYRef.current = 0
            paintRef.current()
          }}
        />
      </div>
      <div className="text-center text-[10px] text-black/45">
        滚轮缩放 · 拖动平移 · 双击复位
      </div>
    </div>
  )
}

/** 检测结果画布:原图 + 可筛选类别边框(仿旧管线查看边框) */
export function DetectCanvas({ items }: { items?: BoxItem[] }) {
  const runId = usePipeline2Store((s) => s.runId)
  const imageSize = usePipeline2Store((s) => s.imageSize)
  const storeElements = usePipeline2Store((s) => s.elements)
  const elements = items ?? storeElements
  const allCls = useMemo(
    () => [...new Set(elements.map((e) => e.cls))], [elements])
  const [visible, setVisible] = useState<string[] | null>(null)
  const shown = visible ?? allCls
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !imageSize) return
    let cancelled = false
    void (async () => {
      const { w: W, h: H } = imageSize
      canvas.width = W
      canvas.height = H
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const img = await loadImg(`/api/runs/${runId}/files/origin.png`)
      if (cancelled) return
      ctx.drawImage(img, 0, 0, W, H)
      const lw = Math.max(2, Math.round(W / 500))
      ctx.font = `bold ${Math.max(11, Math.round(W / 70))}px sans-serif`
      ctx.textBaseline = 'top'
      for (const e of elements) {
        if (!shown.includes(e.cls)) continue
        const [x, y, w, h] = bboxPx(e.bbox, W, H)
        const color = CLS_COLORS[e.cls] ?? '#ff00ff'
        ctx.lineWidth = lw
        ctx.strokeStyle = color
        ctx.strokeRect(x, y, w, h)
        const label = e.id
        const tw = ctx.measureText(label).width
        ctx.fillStyle = color
        ctx.fillRect(x, Math.max(0, y - 15), tw + 6, 15)
        ctx.fillStyle = '#ffffff'
        ctx.fillText(label, x + 3, Math.max(0, y - 14))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [runId, imageSize, elements, shown])

  if (!elements.length) return null
  return (
    <div className="flex flex-col gap-2">
      <Checkbox.Group
        options={allCls.map((c) => ({
          label: <span style={{ color: CLS_COLORS[c] ?? '#000' }}>{c}</span>,
          value: c,
        }))}
        value={shown}
        onChange={(v) => setVisible(v as string[])}
      />
      <div className="w-130">
        <canvas ref={ref} className="h-auto w-full rounded border border-neutral-200" />
      </div>
    </div>
  )
}

function iouOf(a: number[], b: number[]): number {
  const ax0 = a[0] - a[2] / 2, ay0 = a[1] - a[3] / 2, ax1 = a[0] + a[2] / 2, ay1 = a[1] + a[3] / 2
  const bx0 = b[0] - b[2] / 2, by0 = b[1] - b[3] / 2, bx1 = b[0] + b[2] / 2, by1 = b[1] + b[3] / 2
  const ix = Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0))
  const iy = Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0))
  const inter = ix * iy
  const uni = a[2] * a[3] + b[2] * b[3] - inter
  return uni > 0 ? inter / uni : 0
}

/** ①⁺ 层检测查看:下拉切换 原图YOLO/各层绿底检测/机械去重合集,
 * 框画在对应底图上(层选项画在该层图上,便于核对 bbox 是否截断内容) */
function LayerYoloInspector() {
  const s = usePipeline2Store()
  const [source, setSource] = useState('origin')

  const layers = useMemo(
    () => [...new Set(s.layerYolo.map((e) => e.sourceLayer))], [s.layerYolo])
  const dedupIou = s.yoloParams.dedupIou

  const deduped = useMemo(() => {
    // 层连通域(像素贴合)优先于原图 YOLO
    const union = [
      ...s.originYolo.map((b) => ({ ...b, sourceLayer: 'origin', prio: 1, native: 1 })),
      ...s.layerYolo.map((e) => ({ ...e, prio: 2, native: 1 })),
    ]
    union.sort((a, b) =>
      b.prio - a.prio || b.native - a.native || b.conf - a.conf)
    const kept: typeof union = []
    for (const e of union) {
      if (kept.every((k) => iouOf(e.bbox, k.bbox) < dedupIou)) kept.push(e)
    }
    return kept
  }, [s.originYolo, s.layerYolo, dedupIou])

  const { items, imageFile } = useMemo(() => {
    if (source === 'origin') {
      return {
        items: s.originYolo.map((b, i) => ({ id: `${b.cls}_${i}`, cls: b.cls, bbox: b.bbox })),
        imageFile: 'origin.png',
      }
    }
    if (source === 'dedup') {
      return {
        items: deduped.map((b, i) => ({
          id: `${b.sourceLayer}·${b.cls}_${i}`, cls: b.cls, bbox: b.bbox })),
        imageFile: 'origin.png',
      }
    }
    const all = [...s.slotLayers, ...s.panelzLayers]
    const layer = all.find((l) => l.name === source)
    return {
      items: s.layerYolo
        .filter((e) => e.sourceLayer === source)
        .map((b, i) => ({ id: `${b.cls}_${i}`, cls: b.cls, bbox: b.bbox })),
      imageFile: layer?.file ?? 'origin.png',
    }
  }, [source, s.originYolo, s.layerYolo, s.slotLayers, s.panelzLayers, deduped])

  return (
    <div className="h-full w-110 shrink-0">
      <Card
        title={
          <div className="flex items-center justify-between gap-3">
            <span className="text-[15px] font-bold">①⁺ 层检测查看</span>
            <Select
              className="w-44"
              size="small"
              value={source}
              onChange={setSource}
              options={[
                { value: 'origin', label: `原图 YOLO(${s.originYolo.length})` },
                ...layers.map((l) => ({
                  value: l,
                  label: `${l} 层(${s.layerYolo.filter((e) => e.sourceLayer === l).length})`,
                })),
                { value: 'dedup', label: `机械去重(${deduped.length})` },
              ]}
            />
          </div>
        }
        className="flex h-full w-full flex-col shadow-sm"
        styles={{ body: { overflow: 'auto', flex: 1 } }}
      >
        {items.length ? (
          <PanZoomBoxViewer key={source} items={items} imageFile={imageFile} allClasses />
        ) : (
          <div className="text-[12px] text-black/40">
            该来源暂无检测数据(层检测在 ① 生成后自动完成)
          </div>
        )}
      </Card>
    </div>
  )
}

/** 级联每层结果:该层切出的素材按 bbox 原位重组(画布=原图尺寸) */
function CascadeStepCanvas({ assetIds }: { assetIds: string[] }) {
  const runId = usePipeline2Store((st) => st.runId)
  const imageSize = usePipeline2Store((st) => st.imageSize)
  const manifest = usePipeline2Store((st) => st.cascadeManifest)
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !imageSize || !manifest) return
    let cancelled = false
    void (async () => {
      // 用账本画布帧(素材像素的原生帧),与 temp 快照同帧、绘制零重采样;
      // 归一化 bbox 帧无关,老账本无 canvas 字段时退回原图尺寸
      const { w: W, h: H } = manifest.canvas ?? imageSize
      canvas.width = W
      canvas.height = H
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, W, H)
      for (const id of assetIds) {
        const rec = manifest.assets.find((a) => a.id === id)
        if (!rec) continue
        try {
          // ?v=账本版本:级联重跑后文件名不变,不换版会读到 imgCache/HTTP 缓存里的旧图
          const img = await loadImg(
            `/api/runs/${runId}/files/${rec.file}?v=${manifest.generatedAt ?? 0}`)
          if (cancelled) return
          const [dx, dy, dw, dh] = bboxPx(rec.bbox, W, H)
          ctx.drawImage(img, dx, dy, dw, dh)
        } catch {
          /* 单个素材失败不阻塞 */
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [runId, imageSize, manifest, assetIds])

  return (
    <div style={CHECKERBOARD} className="rounded border border-neutral-200">
      <canvas ref={ref} className="h-auto w-full" />
    </div>
  )
}

/** ⑥ 拼回画布(前端实时绘制,不落盘) */
export function RecomposeCanvas() {
  const runId = usePipeline2Store((s) => s.runId)
  const imageSize = usePipeline2Store((s) => s.imageSize)
  const slotLayers = usePipeline2Store((s) => s.slotLayers)
  const panelzLayers = usePipeline2Store((s) => s.panelzLayers)
  const elements = usePipeline2Store((s) => s.elements)
  const hoveredId = usePipeline2Store((s) => s.hoveredElementId)
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !imageSize) return
    let cancelled = false
    const file = (f: string) => `/api/runs/${runId}/files/${f}`
    void (async () => {
      const { w: W, h: H } = imageSize
      canvas.width = W
      canvas.height = H
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, W, H)
      if (hoveredId?.startsWith('layer:')) {
        const img = await loadImg(file(hoveredId.slice(6)))
        if (cancelled) return
        ctx.drawImage(img, 0, 0, W, H)
        return
      }
      if (!hoveredId) {
        const bg = slotLayers.find((l) => l.name === 'bg')
        if (bg) {
          const img = await loadImg(file(bg.file))
          if (cancelled) return
          ctx.drawImage(img, 0, 0, W, H)
        }
        for (const z of panelzLayers) {
          if (z.name === 'bg' || !z.keep) continue
          const img = await loadImg(file(z.file))
          if (cancelled) return
          ctx.drawImage(img, 0, 0, W, H)
        }
      }
      const targets = elements.filter((e) => {
        if (hoveredId) return e.id === hoveredId
        // z 层的 panel 元素已包含在底图 z 层里,整图模式不重复绘制
        return !e.mergedInto && !e.sourceLayer.startsWith('z')
      })
      for (const e of targets) {
        const ex = e.extract
        const bbox = ex?.bbox ?? e.bbox
        const [dx, dy, dw, dh] = bboxPx(bbox, W, H)
        try {
          if (ex && ex.method === 'crop') {
            const img = await loadImg(file(ex.file))
            if (cancelled) return
            ctx.drawImage(img, dx, dy, dw, dh)
          } else {
            const src = ex ? ex.file : e.sourceFile
            const img = await loadImg(file(src))
            if (cancelled) return
            const [sx, sy, sw, sh] = bboxPx(bbox, img.naturalWidth, img.naturalHeight)
            ctx.drawImage(img, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        } catch {
          /* 单个素材加载失败不阻塞整图 */
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [runId, imageSize, slotLayers, panelzLayers, elements, hoveredId])

  return (
    <div style={CHECKERBOARD} className="rounded border border-neutral-200">
      <canvas ref={ref} className="h-auto w-full" />
    </div>
  )
}

/** 新管线(/pipeline2):layered 分层 + 逐层 YOLO + GPT 审核 + 抠取 + 拼回。
 * 编排全在服务端 Python(p2_* 任务),本页只遥控与看图;
 * 各步配置在 card title 的 popover 里,默认值来自 pipeline2Defaults.json。 */
export default function Pipeline2Page() {
  const s = usePipeline2Store()
  const restoreRef = useRef('')

  useEffect(() => {
    void s.fetchRunpodTarget()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])


  return (
    <main className="h-screen overflow-x-auto bg-neutral-100">
      <div className="flex h-full min-w-max gap-4 px-4 py-3">
        <StepCard
          step="upload"
          title="⓪ 原图"
          action={() => {}}
          disabled
          settings={
            <div className="flex w-90 flex-col gap-3">
              <div>
                <div className="mb-1 text-[12px] font-bold">RunPod 地址</div>
                <div className="flex gap-2">
                  <Input
                    size="small"
                    placeholder="https://<pod-id>-8888.proxy.runpod.net"
                    value={s.runpodTarget}
                    onChange={(e) => s.setRunpodTarget(e.target.value)}
                  />
                  <Button
                    size="small"
                    onClick={() =>
                      void s.applyRunpodTarget().catch((e) =>
                        message.error(e instanceof Error ? e.message : '设置失败'))
                    }
                  >
                    应用
                  </Button>
                </div>
              </div>
              <div>
                <div className="mb-1 text-[12px] font-bold">OpenRouter API Key</div>
                <Input.Password
                  size="small"
                  placeholder="默认读旧管线配置(只读)"
                  value={s.apiKey}
                  onChange={(e) => s.setApiKey(e.target.value)}
                />
              </div>
              <div>
                <div className="mb-1 text-[12px] font-bold">恢复历史 run</div>
                <div className="flex gap-2">
                  <Input
                    size="small"
                    placeholder="run_id"
                    onChange={(e) => (restoreRef.current = e.target.value.trim())}
                  />
                  <Button size="small" onClick={() => void s.restoreRun(restoreRef.current)}>
                    恢复
                  </Button>
                </div>
              </div>
            </div>
          }
        >
          {s.originUrl ? (
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between text-[12px] text-black/55">
                <span>
                  run: {s.runId}
                  {s.imageSize ? ` · ${s.imageSize.w}×${s.imageSize.h}` : ''}
                </span>
                <Button
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => s.clearRun()}
                >
                  删除重传
                </Button>
              </div>
              <img
                src={s.originUrl}
                alt="原图"
                className="h-auto w-full rounded border border-neutral-200"
              />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center gap-3 py-6">
              <Upload
                showUploadList={false}
                beforeUpload={(file) => {
                  void s.uploadOriginal(file).catch((e) =>
                    message.error(e instanceof Error ? e.message : '上传失败'))
                  return false
                }}
              >
                <Button
                  type="primary"
                  icon={<UploadOutlined />}
                  loading={s.status.upload === 'running'}
                >
                  上传原图
                </Button>
              </Upload>
              <span className="text-[12px] text-black/45">
                或在右上角配置里按 run_id 恢复
              </span>
            </div>
          )}
          <BatchPanel />
        </StepCard>

        <StepCard
          step="detect"
          title="⓪⁺ 原图 YOLO"
          titleInfo={
            s.originYolo.length ? (
              <span className="text-[12px] font-normal text-black/55">
                {s.originYolo.length} 框
              </span>
            ) : null
          }
          action={() => void s.runDetect()}
          disabled={!s.runId}
          settings={
            <div className="w-80">
              <div className="mb-2 text-[11px] leading-snug text-black/45">
                原图整图 YOLO(含 SAM2 框回投),不走 VL;可与 ① 并行。
                结果供 ② 汇总审核使用,点标题"查看"弹窗核对。
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="col-span-2">
                  <div className="mb-0.5 text-[11px] text-black/50">model</div>
                  <Input size="small" value={s.yoloParams.model}
                    onChange={(e) => s.patchYolo({ model: e.target.value })} />
                </div>
                <NumField label="imgsz" value={s.yoloParams.imgsz} step={64} min={320}
                  onChange={(v) => s.patchYolo({ imgsz: v })} />
                <NumField label="conf" value={s.yoloParams.conf} step={0.05} min={0.01} max={1}
                  onChange={(v) => s.patchYolo({ conf: v })} />
              </div>
            </div>
          }
        >
          {s.originYolo.length ? (
            <div className="flex flex-col gap-2">
              <PanZoomBoxViewer
                items={s.originYolo.map((b, i) => ({
                  id: `${b.cls}_${i}`, cls: b.cls, bbox: b.bbox,
                }))}
              />
            </div>
          ) : (
            <div className="text-[12px] text-black/40">未检测</div>
          )}
        </StepCard>

        <StepCard
          step="sixSlot"
          title="① 六槽分层(自动接 panelz)"
          action={() => void s.runSixSlot()}
          disabled={!s.runId}
          wide
          settings={
            <div className="w-80">
              <div className="mb-2 text-[11px] leading-snug text-black/45">
                原图 → 七层(空层自动弃用);完成后自动执行 panelz
                (bg+panel 合成 → z 分层)。上排参数为六槽,下排为 panelz。
              </div>
              <div className="grid grid-cols-2 gap-2">
                <NumField label="steps" value={s.sixSlotParams.steps} min={10} max={60}
                  onChange={(v) => s.patchSixSlot({ steps: v })} />
                <NumField label="seed" value={s.sixSlotParams.seed} min={0}
                  onChange={(v) => s.patchSixSlot({ seed: v })} />
                <NumField label="true_cfg" value={s.sixSlotParams.trueCfg} step={0.5} min={1} max={10}
                  onChange={(v) => s.patchSixSlot({ trueCfg: v })} />
                <NumField label="resolution(640/1024)" value={s.sixSlotParams.resolution} step={64} min={640} max={1280}
                  onChange={(v) => s.patchSixSlot({ resolution: v })} />
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2 border-t border-black/10 pt-2">
                <NumField label="panelz layers(含bg)" value={s.panelzParams.layers} min={2} max={8}
                  onChange={(v) => s.patchPanelz({ layers: v })} />
                <NumField label="panelz steps" value={s.panelzParams.steps} min={10} max={60}
                  onChange={(v) => s.patchPanelz({ steps: v })} />
                <NumField label="panelz seed" value={s.panelzParams.seed} min={0}
                  onChange={(v) => s.patchPanelz({ seed: v })} />
                <NumField label="panelz cfg" value={s.panelzParams.trueCfg} step={0.5} min={1} max={10}
                  onChange={(v) => s.patchPanelz({ trueCfg: v })} />
              </div>
            </div>
          }
        >
          <div className="flex flex-col gap-2">
            {s.slotLayers.length ? (
              <div className="flex gap-2">
                <Button
                  size="small"
                  loading={s.status.sixYolo === 'running'}
                  onClick={() => void s.runLayerYolo('six')}
                >
                  重做六槽检测
                </Button>
                <Button
                  size="small"
                  loading={s.status.panelzYolo === 'running'}
                  disabled={!s.panelzLayers.length}
                  onClick={() => void s.runLayerYolo('panelz')}
                >
                  重做panelz检测
                </Button>
                <span className="self-center text-[11px] text-black/40">
                  层检测(CV 连通域){s.layerYolo.length} 框
                </span>
              </div>
            ) : null}
            <LayerRow layers={s.slotLayers} runId={s.runId} />
            {(s.status.panelz ?? 'idle') === 'running' ? (
              <div className="text-[12px] text-black/45">panelz 分层中…</div>
            ) : null}
            {(s.status.panelz ?? 'idle') === 'error' ? (
              <div className="break-all text-[12px] text-[#cf1322]">
                panelz:{s.errors.panelz}
              </div>
            ) : null}
            {s.panelzLayers.length ? (
              <>
                <div className="text-[12px] font-bold">panelz(bg+panel → z 分层)</div>
                <LayerRow layers={s.panelzLayers} runId={s.runId} />
              </>
            ) : null}
          </div>
        </StepCard>

        <LayerYoloInspector />

        <StepCard
          step="inventory"
          title="② 汇总审核(去重+VL)"
          titleInfo={
            s.inventoryStats ? (
              <span className="text-[12px] font-normal text-black/55">
                {s.inventoryStats.final} 项
              </span>
            ) : null
          }
          action={() => void s.runInventory()}
          disabled={
            !s.originYolo.length && !s.layerYolo.length
          }
          settings={
            <div className="w-130">
              <div className="mb-2 text-[11px] leading-snug text-black/45">
                原图 YOLO + 各层绿底 YOLO 合集 → 机械去重(原图 &gt; 本职层 &gt; 其它,
                IoU 阈值同 ③)→ VL 复审:类别纠正 / 再去重(dup)/ 剔误检(discard)/
                补漏(missing)。产物是级联切取的裁切权威。
              </div>
              <div className="mb-2">
                <div className="mb-0.5 text-[11px] text-black/50">model</div>
                <Input size="small" value={s.detectParams.model}
                  onChange={(e) => s.patchDetect({ model: e.target.value })} />
              </div>
              <div className="mb-2 grid grid-cols-2 gap-2">
                <div>
                  <div className="mb-0.5 text-[11px] text-black/50">推理强度</div>
                  <Select
                    size="small"
                    className="w-full"
                    value={s.detectParams.effort ?? 'high'}
                    onChange={(v) => s.patchDetect({ effort: v })}
                    options={REASONING_OPTIONS}
                  />
                </div>
                <div>
                  <div className="mb-0.5 text-[11px] text-black/50">速度模式</div>
                  <Select
                    size="small"
                    className="w-full"
                    value={s.detectParams.speed ?? 'balanced'}
                    onChange={(v) => s.patchDetect({ speed: v })}
                    options={SPEED_OPTIONS}
                  />
                </div>
              </div>
              <div className="mb-0.5 text-[11px] text-black/50">复审提示词</div>
              <Input.TextArea
                rows={12}
                value={s.detectParams.prompt}
                onChange={(e) => s.patchDetect({ prompt: e.target.value })}
              />
            </div>
          }
        >
          {s.inventoryStats ? (
            <div className="flex flex-col gap-1 text-[12px] text-black/60">
              <div>
                合集 {s.inventoryStats.union} → 机械去重 {s.inventoryStats.afterDedup}
                → VL 去重 −{s.inventoryStats.vlDup} · 剔除 −{s.inventoryStats.vlDiscard}
                · 补漏 +{s.inventoryStats.vlMissing} → 最终 {s.inventoryStats.final}
              </div>
              <PanZoomBoxViewer
                items={s.originInventory.map((b, i) => ({
                  id: `${b.cls}_${i}`, cls: b.cls, bbox: b.bbox,
                }))}
              />
            </div>
          ) : (
            <div className="text-[12px] text-black/40">
              依赖 ⓪⁺ 或 ① 的层检测结果(层检测在 ① 生成后自动完成)
            </div>
          )}
        </StepCard>

        <StepCard
          step="cascade"
          title="③ 级联切取"
          action={() => void s.runCascade()}
          disabled={!s.slotLayers.length || !s.originInventory.length}
          wide
          settings={
            <div className="w-130 text-[11px] leading-snug text-black/45">
              自顶向下逐层(text→icon→panel_f→assets→controls→zN..z0):
              元素层以 ⓪⁺ 审核清单为裁切权威(孤立连通域直裁 / 粘连 SAM2,
              text 层整批 SAM2);面板层以 layered 为权威连通域全裁,
              先救援登记簿里粘连在 panel 上的元素(类型纠正)。
              上层残留 temp 逐层下沉:落在元素框上→独立成材+叠压关系(不合并),
              落在 panel 内/轮廓相近→并入 panel。终局:temp=碎屑图(PSD 置顶);
              miss 余额差分判定走 bg 路(SAM2+批量 fill 净化 bg)或原图路(SAM2);
              全量重影审计。产物 p2_cascade.json。依赖:①六槽 + ⓪⁺ 检测
              (② panelz 可选,跑了 z 层就参与级联)。
            </div>
          }
        >
          {s.cascadeSummary ? (
            <div className="flex flex-col gap-2 text-[12px]">
              <div className="text-black/60">
                素材 {s.cascadeSummary.count} 件 · 叠压关系 {s.cascadeSummary.relations} 条 ·
                彻底丢失 {s.cascadeSummary.lost} · 重影可疑 {s.cascadeSummary.ghosts} ·{' '}
                <a
                  href={`/api/runs/${s.runId}/files/${s.cascadeSummary.file}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  p2_cascade.json
                </a>
              </div>
              {s.cascadeManifest ? (
                <div className="flex flex-nowrap items-start gap-2">
                  {s.cascadeManifest.steps.map((st) => {
                    const layerAssets = s.cascadeManifest!.assets.filter(
                      (a) => st.assets.includes(a.id))
                    return (
                      <div key={st.layer} className="w-52 shrink-0 rounded border border-black/10 p-1.5">
                        <div className="mb-1 text-[12px] font-bold">
                          {st.layer}
                          <span className="ml-1 font-normal text-black/40">{st.kind}</span>
                        </div>
                        <div className="mb-1 text-[11px] leading-snug text-black/55">
                          裁切 {st.cut} · 救援 {st.rescued.length} · 认领 {st.claimed?.length ?? 0} · 叠压 {st.overlays} · 吸收 {st.absorbed}
                        </div>
                        <div className="mb-0.5 text-[10px] text-black/40">
                          本层切取 {layerAssets.length} 件(原位重组)
                        </div>
                        <CascadeStepCanvas assetIds={st.assets} />
                        <div className="mb-0.5 mt-1 text-[10px] text-black/40">
                          temp 残留 {(st.tempCoverage * 100).toFixed(1)}%
                        </div>
                        <div style={CHECKERBOARD} className="rounded border border-neutral-200">
                          <img
                            src={`/api/runs/${s.runId}/files/${st.tempFile}?v=${s.cascadeManifest!.generatedAt ?? 0}`}
                            alt={`temp@${st.layer}`}
                            loading="lazy"
                            className="h-auto w-full"
                          />
                        </div>
                      </div>
                    )
                  })}
                  <div className="w-44 shrink-0 rounded border border-black/10 p-1.5">
                    <div className="mb-1 text-[12px] font-bold">终局</div>
                    <div className="mb-0.5 text-[10px] text-black/40">碎屑图(PSD 置顶)</div>
                    <div style={CHECKERBOARD} className="mb-1 rounded border border-neutral-200">
                      <img
                        src={`/api/runs/${s.runId}/files/${s.cascadeSummary.debris}?v=${s.cascadeManifest.generatedAt ?? 0}`}
                        alt="碎屑"
                        loading="lazy"
                        className="h-auto w-full"
                      />
                    </div>
                    <div className="mb-0.5 text-[10px] text-black/40">最终 bg(找回后已 fill)</div>
                    <img
                      src={`/api/runs/${s.runId}/files/${s.cascadeSummary.background}?v=${s.cascadeManifest.generatedAt ?? 0}`}
                      alt="bg"
                      loading="lazy"
                      className="mb-1 h-auto w-full rounded border border-neutral-200"
                    />
                    <div className="text-[10px] leading-snug text-black/55">
                      找回:bg {s.cascadeManifest.report.recoveredBg} · 原图 {s.cascadeManifest.report.recoveredOrigin}
                    </div>
                    {s.cascadeManifest.report.lost.length ? (
                      <div className="mt-1 flex flex-wrap gap-0.5">
                        {s.cascadeManifest.report.lost.map((l) => (
                          <Tag key={l.id} color="red" className="!m-0 !text-[10px]">{l.id}</Tag>
                        ))}
                      </div>
                    ) : null}
                    {s.cascadeManifest.report.ghostSuspects.length ? (
                      <div className="mt-1 flex flex-wrap gap-0.5">
                        {s.cascadeManifest.report.ghostSuspects.map((g) => (
                          <Tag key={g.id} color="magenta" className="!m-0 !text-[10px]">
                            影:{g.id}
                          </Tag>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}
        </StepCard>

        <StepCard
          step="psd"
          title="④ 生成 PSD"
          titleInfo={
            s.psdSummary ? (
              <span className="text-[12px] font-normal text-black/55">
                {s.psdSummary.layers} 层 · {s.psdSummary.sizeMB}MB
              </span>
            ) : null
          }
          action={() => void s.runPsd()}
          disabled={!s.cascadeSummary}
          settings={
            <div className="w-100 text-[11px] leading-snug text-black/45">
              服务端用 p2_cascade.json + 交付素材生成分组分层 PSD,落网盘
              run 目录 p2_psd/result.psd。图层序:bg → z 层 → controls →
              assets → panel_f → icon → text,碎屑图置顶;素材按来源层进组,
              叠压素材位于被叠元素上方。同时拍平 preview 与原图算误差。
              重跑级联后需重新生成。依赖:③ 级联切取
            </div>
          }
        >
          {s.status.psd === 'running' ? (
            <div className="flex items-center gap-2 text-[12px] text-black/55">
              <Spin size="small" /> 正在拼装图层并写出 PSD…
            </div>
          ) : s.psdSummary ? (
            <div className="flex flex-col gap-2 text-[12px]">
              <div className="w-90">
                <ImageCompareSlider
                  leftSrc={`/api/runs/${s.runId}/files/origin.png`}
                  rightSrc={`/api/runs/${s.runId}/files/${s.psdSummary.preview}?v=${s.psdSummary.generatedAt}`}
                />
                <p className="m-0 mt-1 text-center text-[10px] text-black/45">
                  拖动分界线对比 · 左原图 / 右 PSD 拍平预览
                </p>
              </div>
              <div className="text-black/60">
                图层 {s.psdSummary.layers}(分组 {s.psdSummary.groups}
                {s.psdSummary.skipped ? `,跳过空素材 ${s.psdSummary.skipped}` : ''})·
                文件 {s.psdSummary.sizeMB} MB · 耗时 {s.psdSummary.elapsed}s ·
                与原图平均误差{' '}
                <span className={s.psdSummary.diffMean > 8 ? 'text-[#cf1322]' : 'text-[#389e0d]'}>
                  {s.psdSummary.diffMean}/255
                </span>{' '}
                · 差异像素{' '}
                <span className={s.psdSummary.diffPct > 5 ? 'text-[#cf1322]' : 'text-[#389e0d]'}>
                  {s.psdSummary.diffPct}%
                </span>
                <Button
                  size="small"
                  type="primary"
                  className="ml-3"
                  href={`/api/runs/${s.runId}/files/${s.psdSummary.file}?v=${s.psdSummary.generatedAt}`}
                  download="result.psd"
                >
                  下载 PSD
                </Button>
              </div>
            </div>
          ) : null}
        </StepCard>
      </div>
    </main>
  )
}
