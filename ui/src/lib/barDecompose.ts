// 第 12+ 步"bar分解":逐 bar 裁块(从 bar.png 提取层,绿底合成),
// 按 YOLO 长宽比判横竖,调用 GPT Image 生成五段式拆解图(chroma green)。
import type { DetectionItem } from './detection'

export type BarOrientation = 'horizontal' | 'vertical'

/** 判定覆盖物(icon/assets/panel)是否压在某个 bar 上:
 * 中心落在 bar 框内,或相交面积占覆盖物自身面积 ≥50%;
 * 但相交面积盖住 bar 自身 ≥50% 的是"承载底板"(bar 画在它上面),
 * 不是覆盖物,剔除——覆盖物只会遮住 bar 的一小段。
 * anyOverlap=true(panel_f 用):相交面积占自身 ≥10% 即命中(阈值远低于
 * 常规的 50%),也不走承载底板剔除——panel_f 定义上就是浮在中景上的夹层。 */
export function isOverlayOnBar(
  barBbox: number[],
  det: { bbox: number[] },
  anyOverlap = false,
): boolean {
  const [bcx, bcy, bw, bh] = barBbox
  const [ocx, ocy, ow, oh] = det.bbox
  const ix = Math.min(bcx + bw / 2, ocx + ow / 2) - Math.max(bcx - bw / 2, ocx - ow / 2)
  const iy = Math.min(bcy + bh / 2, ocy + oh / 2) - Math.max(bcy - bh / 2, ocy - oh / 2)
  if (ix <= 0 || iy <= 0) return false
  if (anyOverlap) return (ix * iy) / Math.max(1e-9, ow * oh) >= 0.1
  if ((ix * iy) / Math.max(1e-9, bw * bh) >= 0.5) return false
  const centerInside =
    Math.abs(ocx - bcx) <= bw / 2 && Math.abs(ocy - bcy) <= bh / 2
  return centerInside || (ix * iy) / Math.max(1e-9, ow * oh) >= 0.5
}

/** 两个归一化 cxcywh 框的 IoU(探测结果与结构检测去重用) */
export function bboxIoU(a: number[], b: number[]): number {
  const ix =
    Math.min(a[0] + a[2] / 2, b[0] + b[2] / 2) -
    Math.max(a[0] - a[2] / 2, b[0] - b[2] / 2)
  const iy =
    Math.min(a[1] + a[3] / 2, b[1] + b[3] / 2) -
    Math.max(a[1] - a[3] / 2, b[1] - b[3] / 2)
  if (ix <= 0 || iy <= 0) return 0
  const inter = ix * iy
  return inter / Math.max(1e-9, a[2] * a[3] + b[2] * b[3] - inter)
}

/** 解析 bar 裁块二次 YOLO 探测的 save_txt 行,映射回全图归一化坐标。
 * 只收 icon/assets/button 三类、方形度 ≤maxAspect(排除被误检的标牌等
 * 宽扁部件)、置信度达标的框——这些是结构检测在整图上漏掉的小覆盖物。 */
export function probeLinesToBBoxes(
  lines: string[],
  rect: { x0: number; y0: number; x1: number; y1: number },
  W: number,
  H: number,
  opts = { classes: [1, 2, 3], minConf: 0.3, maxAspect: 1.6 },
): number[][] {
  const cw = rect.x1 - rect.x0
  const ch = rect.y1 - rect.y0
  const out: number[][] = []
  for (const line of lines) {
    const p = line.trim().split(/\s+/).map(Number)
    if (p.length < 5 || p.some((v) => Number.isNaN(v))) continue
    const [cls, cx, cy, w, h] = p
    const conf = p[5] ?? 1
    if (!opts.classes.includes(cls) || conf < opts.minConf) continue
    const pw = w * cw
    const ph = h * ch
    if (Math.max(pw, ph) / Math.max(1, Math.min(pw, ph)) > opts.maxAspect) continue
    out.push([(rect.x0 + cx * cw) / W, (rect.y0 + cy * ch) / H, pw / W, ph / H])
  }
  return out
}

/** 多个归一化 cxcywh 框的并集框 */
export function unionBBoxes(bboxes: number[][]): number[] {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (const [cx, cy, w, h] of bboxes) {
    x0 = Math.min(x0, cx - w / 2)
    y0 = Math.min(y0, cy - h / 2)
    x1 = Math.max(x1, cx + w / 2)
    y1 = Math.max(y1, cy + h / 2)
  }
  return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
}

/** 按 bbox 像素长宽比判定横竖(方形按横处理,与用户提示词口径一致) */
export function barOrientation(
  bbox: number[],
  imgW: number,
  imgH: number,
): BarOrientation {
  return bbox[2] * imgW >= bbox[3] * imgH ? 'horizontal' : 'vertical'
}

/** 从提取层(透明 RGBA)裁出单个 bar,合成到纯绿底上 */
/** 判断一条截面(宽/高 1px)里是否有非绿墨迹 */
function sliceHasInk(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number,
  W: number, H: number,
): boolean {
  if (x < 0 || y < 0 || x + w > W || y + h > H || w <= 0 || h <= 0) return false
  const d = ctx.getImageData(x, y, w, h).data
  for (let i = 0; i < d.length; i += 4) {
    if (Math.abs(d[i]) > 8 || Math.abs(d[i + 1] - 255) > 8 || Math.abs(d[i + 2]) > 8)
      return true
  }
  return false
}

/** 裁块矩形(像素坐标),供裁剪与"探测框映射回全图"共用 */
export function cropRect(
  bbox: number[],
  W: number,
  H: number,
  marginRatio = 0.06,
): { x0: number; y0: number; x1: number; y1: number } {
  const [cx, cy, w, h] = bbox
  const m = Math.max(4, Math.round(Math.max(w * W, h * H) * marginRatio))
  return {
    x0: Math.max(0, Math.round((cx - w / 2) * W) - m),
    y0: Math.max(0, Math.round((cy - h / 2) * H) - m),
    x1: Math.min(W, Math.round((cx + w / 2) * W) + m),
    y1: Math.min(H, Math.round((cy + h / 2) * H) + m),
  }
}

type PxRect = { ex0: number; ey0: number; ex1: number; ey1: number }

/** 几何鼓包检测(alpha 通道版):沿轨道方向扫描**实心**墨迹(α≥200,
 * 半透明辉光不算)的截面高度,找出显著超出管径基线、方形度接近 1 的
 * "鼓包"(里程碑空槽/指示物底座——YOLO 检不出的部件)。
 * 宽扁的端头标牌(aspect>1.6)不会命中。 */
function detectBumps(
  A: Uint8ClampedArray,
  CW: number,
  CH: number,
  orientation: BarOrientation,
): PxRect[] {
  const solid = (x: number, y: number) => A[(y * CW + x) * 4 + 3] >= 200
  // 沿主轴逐条截面:记录实心墨迹跨度(次轴方向)
  const N = orientation === 'horizontal' ? CW : CH
  const M = orientation === 'horizontal' ? CH : CW
  const span: ({ a: number; b: number } | null)[] = []
  for (let i = 0; i < N; i++) {
    let a = -1
    let b = -1
    for (let j = 0; j < M; j++) {
      const ink = orientation === 'horizontal' ? solid(i, j) : solid(j, i)
      if (ink) {
        if (a < 0) a = j
        b = j
      }
    }
    span.push(a < 0 ? null : { a, b })
  }
  const heights = span.filter(Boolean).map((s) => s!.b - s!.a + 1)
  if (heights.length < 20) return []
  // 管径基线 = 实心截面高度的 35 分位数(管子占多数列,分位稳定落在管径)
  const sorted = [...heights].sort((x, y) => x - y)
  const baseline = sorted[Math.floor(sorted.length * 0.35)]
  // 鼓包 = 截面高度显著超基线的连续段(允许 2 条截面的断缝)
  const isBump = (i: number) => {
    const s = span[i]
    return s !== null && s.b - s.a + 1 > Math.max(baseline * 1.6, baseline + 6)
  }
  const rects: { ex0: number; ey0: number; ex1: number; ey1: number }[] = []
  let i = 0
  while (i < N) {
    if (!isBump(i)) {
      i++
      continue
    }
    let j = i
    let gap = 0
    let lo = Infinity
    let hi = -Infinity
    while (j < N && gap <= 2) {
      if (isBump(j)) {
        gap = 0
        lo = Math.min(lo, span[j]!.a)
        hi = Math.max(hi, span[j]!.b)
      } else {
        gap++
      }
      j++
    }
    const runW = j - gap - i
    const runH = hi - lo + 1
    const aspect = Math.max(runW, runH) / Math.max(1, Math.min(runW, runH))
    // 方形度接近 1 才是槽位/底座;宽扁标牌、长条装饰不动
    if (aspect <= 1.6 && runW >= baseline) {
      const p = 4
      rects.push(
        orientation === 'horizontal'
          ? { ex0: Math.max(0, i - p), ey0: Math.max(0, lo - p),
              ex1: Math.min(CW, j - gap + p), ey1: Math.min(CH, hi + 1 + p) }
          : { ex0: Math.max(0, lo - p), ey0: Math.max(0, i - p),
              ex1: Math.min(CW, hi + 1 + p), ey1: Math.min(CH, j - gap + p) },
      )
    }
    i = j
  }
  return rects
}

export async function cropBarOnGreen(
  layerDataUrl: string,
  bbox: number[],
  marginRatio = 0.06,
  eraseBoxes?: number[][],
  bridgeOrientation?: BarOrientation,
): Promise<{ url: string; erased: number }> {
  const img = new Image()
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve()
    img.onerror = () => reject(new Error('bar 提取层加载失败'))
    img.src = layerDataUrl
  })
  const W = img.naturalWidth
  const H = img.naturalHeight
  const { x0, y0, x1, y1 } = cropRect(bbox, W, H, marginRatio)
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, x1 - x0)
  canvas.height = Math.max(1, y1 - y0)
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 不可用')
  const CW = canvas.width
  const CH = canvas.height
  // 先画在透明底上:alpha 通道留给鼓包检测(实心/辉光可分),绿底最后垫
  ctx.drawImage(img, x0, y0, CW, CH, 0, 0, CW, CH)
  const rects: PxRect[] = []
  if (eraseBoxes?.length || bridgeOrientation) {
    const id = ctx.getImageData(0, 0, CW, CH)
    const A = id.data
    const zeroRect = (r: PxRect) => {
      for (let yy = r.ey0; yy < r.ey1; yy++)
        for (let xx = r.ex0; xx < r.ex1; xx++) A[(yy * CW + xx) * 4 + 3] = 0
    }
    // 阶段一:已知覆盖物按检测框抹除(外扩 4px;先全抹再桥,互不污染采样)
    for (const [ecx, ecy, ew, eh] of eraseBoxes ?? []) {
      const r = {
        ex0: Math.max(0, Math.round((ecx - ew / 2) * W) - 4 - x0),
        ey0: Math.max(0, Math.round((ecy - eh / 2) * H) - 4 - y0),
        ex1: Math.min(CW, Math.round((ecx + ew / 2) * W) + 4 - x0),
        ey1: Math.min(CH, Math.round((ecy + eh / 2) * H) + 4 - y0),
      }
      if (r.ex1 - r.ex0 <= 0 || r.ey1 - r.ey0 <= 0) continue
      zeroRect(r)
      rects.push(r)
    }
    // 阶段一·五:几何鼓包兜底(空槽框 YOLO 检不出,按实心截面找方形鼓包)
    if (bridgeOrientation) {
      for (const r of detectBumps(A, CW, CH, bridgeOrientation)) {
        zeroRect(r)
        rects.push(r)
      }
    }
    // 辉光清扫:抹除区外扩 14px 带内的半透明像素(α<200)一并清掉,
    // 不留能暗示覆盖物轮廓的光晕;实心轨道穿过带内不受影响
    const G = 14
    for (const r of rects) {
      for (let yy = Math.max(0, r.ey0 - G); yy < Math.min(CH, r.ey1 + G); yy++)
        for (let xx = Math.max(0, r.ex0 - G); xx < Math.min(CW, r.ex1 + G); xx++) {
          const k = (yy * CW + xx) * 4 + 3
          if (A[k] > 0 && A[k] < 200) A[k] = 0
        }
    }
    ctx.putImageData(id, 0, 0)
  }
  // 垫绿底(destination-over:已清透明处露绿)
  ctx.globalCompositeOperation = 'destination-over'
  ctx.fillStyle = '#00FF00'
  ctx.fillRect(0, 0, CW, CH)
  ctx.globalCompositeOperation = 'source-over'
  // 阶段二:按轨道方向排序逐个桥接,前一个桥好的区域可作下一个的截面源
  rects.sort((a, b) =>
    bridgeOrientation === 'vertical' ? a.ey0 - b.ey0 : a.ex0 - b.ex0,
  )
  for (const { ex0, ey0, ex1, ey1 } of rects) {
    const gw = ex1 - ex0
    const gh = ey1 - ey0
    if (!bridgeOrientation) break
    if (bridgeOrientation === 'horizontal') {
      const lx = ex0 - 2
      const rx = ex1 + 1
      const lOk = sliceHasInk(ctx, lx, ey0, 1, gh, CW, CH)
      const rOk = sliceHasInk(ctx, rx, ey0, 1, gh, CW, CH)
      const mid = ex0 + Math.round(gw / 2)
      if (lOk && rOk) {
        ctx.drawImage(canvas, lx, ey0, 1, gh, ex0, ey0, mid - ex0, gh)
        ctx.drawImage(canvas, rx, ey0, 1, gh, mid, ey0, ex1 - mid, gh)
      } else if (lOk) {
        ctx.drawImage(canvas, lx, ey0, 1, gh, ex0, ey0, gw, gh)
      } else if (rOk) {
        ctx.drawImage(canvas, rx, ey0, 1, gh, ex0, ey0, gw, gh)
      }
    } else {
      const ty = ey0 - 2
      const by = ey1 + 1
      const tOk = sliceHasInk(ctx, ex0, ty, gw, 1, CW, CH)
      const bOk = sliceHasInk(ctx, ex0, by, gw, 1, CW, CH)
      const mid = ey0 + Math.round(gh / 2)
      if (tOk && bOk) {
        ctx.drawImage(canvas, ex0, ty, gw, 1, ex0, ey0, gw, mid - ey0)
        ctx.drawImage(canvas, ex0, by, gw, 1, ex0, mid, gw, ey1 - mid)
      } else if (tOk) {
        ctx.drawImage(canvas, ex0, ty, gw, 1, ex0, ey0, gw, gh)
      } else if (bOk) {
        ctx.drawImage(canvas, ex0, by, gw, 1, ex0, ey0, gw, gh)
      }
    }
  }
  return { url: canvas.toDataURL('image/png'), erased: rects.length }
}

function extractImage(result: unknown): string {
  const r = result as {
    choices?: {
      message?: {
        images?: { image_url?: { url?: string } }[]
        content?: unknown
      }
    }[]
  }
  const msg = r?.choices?.[0]?.message
  const fromImages = msg?.images?.[0]?.image_url?.url
  if (fromImages) return fromImages
  if (Array.isArray(msg?.content)) {
    for (const part of msg.content as { image_url?: { url?: string } }[]) {
      if (part?.image_url?.url) return part.image_url.url
    }
  }
  throw new Error('模型没有返回图片')
}

export async function generateBarDecompose(opts: {
  apiKey: string
  model: string
  systemPrompt: string
  userPrompt: string
  barDataUrl: string
  /** 覆盖物已机械移除+桥直的轨道图;提供时它就是唯一输入图 */
  trackDataUrl?: string
  orientation: BarOrientation
  signal?: AbortSignal
}): Promise<string> {
  // 模型适配:Gemini/Qwen 图像模型对 system 角色的遵循不可靠,
  // 把系统规则并入用户消息(单条 user);GPT 保持 system+user 双消息
  const isQwen = opts.model.startsWith('qwen/')
  const mergeSystem = isQwen || opts.model.includes('gemini')
  const userText =
    (mergeSystem
      ? 'Follow ALL of the rules below strictly when generating the image.\n\n' +
        opts.systemPrompt +
        '\n\n'
      : '') +
    `${opts.userPrompt}\nThe attached input bar is ${opts.orientation.toUpperCase()}.`
  // 只发一张图:轨道图(覆盖物已机械移除+桥直)优先,没有抹除时用原裁块
  const images = [
    { type: 'image_url', image_url: { url: opts.trackDataUrl ?? opts.barDataUrl } },
  ]
  // Qwen 系走专用 images 端点(chat/completions 不载图像生成模型)
  const response = isQwen
    ? await fetch('https://openrouter.ai/api/v1/images', {
        method: 'POST',
        signal: opts.signal,
        headers: {
          Authorization: `Bearer ${opts.apiKey}`,
          'Content-Type': 'application/json',
          'X-OpenRouter-Title': 'Bar Decompose',
        },
        body: JSON.stringify({
          model: opts.model,
          prompt: userText,
          n: 1,
          output_format: 'png',
          input_references: images,
        }),
      })
    : await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        signal: opts.signal,
        headers: {
          Authorization: `Bearer ${opts.apiKey}`,
          'Content-Type': 'application/json',
          'X-OpenRouter-Title': 'Bar Decompose',
        },
        body: JSON.stringify({
          model: opts.model,
          modalities: ['image', 'text'],
          messages: [
            ...(mergeSystem
              ? []
              : [{ role: 'system', content: opts.systemPrompt }]),
            { role: 'user', content: [{ type: 'text', text: userText }, ...images] },
          ],
        }),
      })
  if (!response.ok) {
    const raw = await response.text()
    let detail = raw
    try {
      const body = JSON.parse(raw)
      detail = body?.error?.message || body?.message || raw
    } catch {
      /* 保留原始错误文本 */
    }
    throw new Error(`HTTP ${response.status}：${detail}`)
  }
  const result = await response.json()
  if ((result as { error?: { message?: string } })?.error) {
    throw new Error(
      (result as { error: { message?: string } }).error.message ||
        'OpenRouter 返回了失败状态',
    )
  }
  if (isQwen) {
    const item = (result as {
      data?: { b64_json?: string; media_type?: string; url?: string }[]
    })?.data?.[0]
    if (item?.b64_json)
      return `data:${item.media_type || 'image/png'};base64,${item.b64_json}`
    if (item?.url) return item.url
    throw new Error('images 端点没有返回图片')
  }
  return extractImage(result)
}

function transposeImageData(src: ImageData): ImageData {
  const { width: W, height: H, data } = src
  const out = new ImageData(H, W)
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const si = (y * W + x) * 4
      const di = (x * H + y) * 4
      out.data[di] = data[si]
      out.data[di + 1] = data[si + 1]
      out.data[di + 2] = data[si + 2]
      out.data[di + 3] = data[si + 3]
    }
  return out
}

/** 把三段式分解图切成三个已去绿键的图层(工作坐标系:三段横排、bar 纵向)。
 * recomposeDecomposedBar(拼回)与 extractDecomposedLayers(逐层落盘)共用。 */
async function splitDecomposedSections(
  imageUrl: string,
  orientation: BarOrientation,
): Promise<{ layers: ImageData[]; T: number; H: number; rows: boolean }> {
  const img = new Image()
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve()
    img.onerror = () => reject(new Error('分解图加载失败'))
    img.src = imageUrl
  })
  const src = document.createElement('canvas')
  src.width = img.naturalWidth
  src.height = img.naturalHeight
  const sctx = src.getContext('2d')
  if (!sctx) throw new Error('canvas 不可用')
  sctx.drawImage(img, 0, 0)
  let id = sctx.getImageData(0, 0, src.width, src.height)
  // 横向 bar 的分解图是三段竖排:转置后统一按"横排三段"处理
  const rows = orientation === 'horizontal'
  if (rows) id = transposeImageData(id)
  const { width: W, height: H, data } = id

  // 每列近黑覆盖率(分隔线检测)
  const cov = new Float32Array(W)
  for (let x = 0; x < W; x++) {
    let n = 0
    for (let y = 0; y < H; y++) {
      const i = (y * W + x) * 4
      if (Math.max(data[i], data[i + 1], data[i + 2]) <= 32) n++
    }
    cov[x] = n / H
  }
  const bounds: [number, number][] = []
  for (const k of [1, 2]) {
    const expected = Math.round((W * k) / 3)
    const rad = Math.max(4, Math.round((W / 3) * 0.12))
    let peak = expected
    for (let x = Math.max(0, expected - rad); x <= Math.min(W - 1, expected + rad); x++)
      if (cov[x] > cov[peak]) peak = x
    if (cov[peak] >= 0.5) {
      const expand = Math.max(0.35, cov[peak] * 0.65)
      let s = peak
      let e = peak + 1
      while (s > 0 && cov[s - 1] >= expand) s--
      while (e < W && cov[e] >= expand) e++
      bounds.push([s, e])
    } else {
      bounds.push([expected, expected]) // 没检出分隔线,按理论等分切
    }
  }
  const ranges: [number, number][] = [
    [0, bounds[0][0]],
    [bounds[0][1], bounds[1][0]],
    [bounds[1][1], W],
  ]

  // 逐段切出统一宽度画布(居中)并去绿键
  const T = Math.floor(W / 3)
  const edge = Math.max(2, Math.round((W / 3) * 0.04))
  const layers = ranges.map(([l, r]) => {
    const out = new ImageData(T, H)
    const w = r - l
    const dstOff = w <= T ? Math.floor((T - w) / 2) : 0
    const srcOff = w > T ? Math.floor((w - T) / 2) : 0
    const copyW = Math.min(w, T)
    for (let y = 0; y < H; y++)
      for (let i = 0; i < copyW; i++) {
        // 段边缘的贯穿黑线残留(旧五段图的分隔线)直接跳过不拷
        if (
          (i < edge || i >= copyW - edge) &&
          cov[l + srcOff + i] >= 0.8
        )
          continue
        const si = (y * W + l + srcOff + i) * 4
        const di = (y * T + dstOff + i) * 4
        const rr = data[si]
        const gg = data[si + 1]
        const bb = data[si + 2]
        const dist = Math.sqrt(rr * rr + (gg - 255) * (gg - 255) + bb * bb)
        let a = Math.min(1, Math.max(0, (dist - 35) / 75))
        const excess = gg - Math.max(rr, bb)
        if (excess > 0) a = Math.min(a, 1 - Math.min(1, excess / 255))
        if (a <= 0) continue
        // 反解 observed = a*fg + (1-a)*key,压绿边
        out.data[di] = Math.min(255, rr / a)
        out.data[di + 1] = Math.min(255, Math.max(0, (gg - (1 - a) * 255) / a))
        out.data[di + 2] = Math.min(255, bb / a)
        out.data[di + 3] = Math.round(a * 255)
      }
    return out
  })
  return { layers, T, H, rows }
}

/** 把三段式分解图拼回完整 bar:切段去绿键 → 中轴线配准 → 依序叠合。 */
export async function recomposeDecomposedBar(
  imageUrl: string,
  orientation: BarOrientation,
): Promise<string> {
  const { layers, T, H, rows } = await splitDecomposedSections(
    imageUrl, orientation,
  )
  // 中轴线配准:工作坐标系里(竖排图已转置)三段的 bar 一律纵向,
  // 每个物体的中轴线 = alpha 包围盒的 x 中心;以底板为基准(缺席时
  // 依次退到边框/填充),边框与填充平移对轴,进度方向(y)不动
  const xSpan = layers.map((ld) => {
    let x0 = T
    let x1 = -1
    for (let y = 0; y < H; y++)
      for (let x = 0; x < T; x++)
        if (ld.data[(y * T + x) * 4 + 3] > 8) {
          if (x < x0) x0 = x
          if (x > x1) x1 = x
        }
    return x1 < 0 ? null : { x0, x1 }
  })
  const ref = xSpan[2] ?? xSpan[0] ?? xSpan[1]
  const refCenter = ref ? (ref.x0 + ref.x1) / 2 : T / 2
  // 叠合:base_plate(2) → progress_fill(1) → border(0)
  const cvs = document.createElement('canvas')
  cvs.width = T
  cvs.height = H
  const ctx = cvs.getContext('2d')
  if (!ctx) throw new Error('canvas 不可用')
  for (const k of [2, 1, 0]) {
    const span = xSpan[k]
    const dx = span
      ? Math.round(refCenter - (span.x0 + span.x1) / 2)
      : 0
    const tmp = document.createElement('canvas')
    tmp.width = T
    tmp.height = H
    tmp.getContext('2d')!.putImageData(layers[k], 0, 0)
    ctx.drawImage(tmp, dx, 0)
  }
  if (!rows) return cvs.toDataURL('image/png')
  // 转置回横向
  const back = document.createElement('canvas')
  back.width = H
  back.height = T
  back.getContext('2d')!.putImageData(
    transposeImageData(ctx.getImageData(0, 0, T, H)), 0, 0,
  )
  return back.toDataURL('image/png')
}

export const DECOMPOSE_LAYER_NAMES = ['border', 'progress_fill', 'base_plate'] as const
export type DecomposeLayerName = (typeof DECOMPOSE_LAYER_NAMES)[number]

/** 把三段式分解图逐层导出为最小尺寸透明 PNG(按 alpha 包围盒紧裁,
 * 原方向)。空层(如没有 border)直接跳过不出图。 */
export async function extractDecomposedLayers(
  imageUrl: string,
  orientation: BarOrientation,
): Promise<
  { name: DecomposeLayerName; dataUrl: string; width: number; height: number }[]
> {
  const { layers, rows } = await splitDecomposedSections(imageUrl, orientation)
  const out: {
    name: DecomposeLayerName
    dataUrl: string
    width: number
    height: number
  }[] = []
  for (let k = 0; k < DECOMPOSE_LAYER_NAMES.length; k++) {
    const ld = rows ? transposeImageData(layers[k]) : layers[k]
    const w = ld.width
    const h = ld.height
    let x0 = w, y0 = h, x1 = -1, y1 = -1
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        if (ld.data[(y * w + x) * 4 + 3] > 8) {
          if (x < x0) x0 = x
          if (y < y0) y0 = y
          if (x > x1) x1 = x
          if (y > y1) y1 = y
        }
    if (x1 < 0) continue // 空层(如无 border):不生成 PNG
    const cw = x1 - x0 + 1
    const ch = y1 - y0 + 1
    const full = document.createElement('canvas')
    full.width = w
    full.height = h
    full.getContext('2d')!.putImageData(ld, 0, 0)
    const tight = document.createElement('canvas')
    tight.width = cw
    tight.height = ch
    tight.getContext('2d')!.drawImage(full, x0, y0, cw, ch, 0, 0, cw, ch)
    out.push({
      name: DECOMPOSE_LAYER_NAMES[k],
      dataUrl: tight.toDataURL('image/png'),
      width: cw,
      height: ch,
    })
  }
  return out
}

/** 一类同类 bar:外形一样、进度含义一致,仅因列表复用渲染了多个 */
export interface BarGroup {
  /** 这类 bar 是什么(中文描述) */
  name: string
  /** 组内成员在 bar 检测列表中的下标 */
  members: number[]
  /** 被选中用于分解的代表下标(进度最多但非 100% 优先) */
  selected: number
  /** 选择理由(一句话) */
  reason: string
}

// 模型只返回分组下标,几何数据不过模型的手(几何权威原则)
const BAR_GROUP_SCHEMA = {
  type: 'object',
  properties: {
    groups: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          members: {
            type: 'array',
            items: { type: 'integer', minimum: 0 },
            minItems: 1,
          },
          selected: { type: 'integer', minimum: 0 },
          reason: { type: 'string' },
        },
        required: ['name', 'members', 'selected', 'reason'],
        additionalProperties: false,
      },
    },
  },
  required: ['groups'],
  additionalProperties: false,
} as const

/** VL 分组:传原图 + 第 12 步 bar 提取层,把同类 bar 归组并各选一个代表。
 * 校验兜底:selected 必须在 members 内(否则取组首),漏归组的 bar
 * 自动补成单例组。 */
export async function groupBars(opts: {
  apiKey: string
  model: string
  temperature: number
  systemPrompt: string
  userPrompt: string
  originDataUrl: string
  barLayerDataUrl: string
  bars: { index: number; bbox: number[] }[]
  signal?: AbortSignal
}): Promise<BarGroup[]> {
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    signal: opts.signal,
    headers: {
      Authorization: `Bearer ${opts.apiKey}`,
      'Content-Type': 'application/json',
      'X-OpenRouter-Title': 'Bar Grouping',
    },
    body: JSON.stringify({
      model: opts.model,
      temperature: opts.temperature,
      messages: [
        { role: 'system', content: opts.systemPrompt },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: `${opts.userPrompt}\n${JSON.stringify(opts.bars)}`,
            },
            { type: 'image_url', image_url: { url: opts.originDataUrl } },
            { type: 'image_url', image_url: { url: opts.barLayerDataUrl } },
          ],
        },
      ],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'bar_groups',
          strict: true,
          schema: BAR_GROUP_SCHEMA,
        },
      },
      provider: { require_parameters: true },
    }),
  })
  if (!response.ok) {
    const raw = await response.text()
    let detail = raw
    try {
      const body = JSON.parse(raw)
      detail = body?.error?.message || body?.message || raw
    } catch {
      /* 保留原始错误文本 */
    }
    throw new Error(`HTTP ${response.status}：${detail}`)
  }
  const result = await response.json()
  if ((result as { error?: { message?: string } })?.error) {
    throw new Error(
      (result as { error: { message?: string } }).error.message ||
        'OpenRouter 返回了失败状态',
    )
  }
  const content = (result as {
    choices?: { message?: { content?: string } }[]
  })?.choices?.[0]?.message?.content
  if (!content) throw new Error('分组模型没有返回内容')
  let parsed: { groups?: BarGroup[] }
  try {
    parsed = JSON.parse(content)
  } catch {
    throw new Error('分组模型返回的内容不是有效 JSON')
  }
  const valid = new Set(opts.bars.map((b) => b.index))
  const seen = new Set<number>()
  const groups: BarGroup[] = []
  for (const g of parsed.groups ?? []) {
    const members = (g.members ?? []).filter(
      (i) => valid.has(i) && !seen.has(i),
    )
    if (!members.length) continue
    members.forEach((i) => seen.add(i))
    groups.push({
      name: g.name || `bar 组 ${groups.length + 1}`,
      members,
      selected: members.includes(g.selected) ? g.selected : members[0],
      reason: g.reason || '',
    })
  }
  // 漏归组的 bar 补成单例组
  for (const b of opts.bars) {
    if (!seen.has(b.index)) {
      groups.push({
        name: `未归组 bar #${b.index}`,
        members: [b.index],
        selected: b.index,
        reason: '模型未归组,自动补为单例',
      })
    }
  }
  return groups
}

/** 检测项转 bar 任务清单 */
export function listBars(
  bars: DetectionItem[],
  imgW: number,
  imgH: number,
): { index: number; bbox: number[]; orientation: BarOrientation }[] {
  return bars.map((b, index) => ({
    index,
    bbox: b.bbox,
    orientation: barOrientation(b.bbox, imgW, imgH),
  }))
}
