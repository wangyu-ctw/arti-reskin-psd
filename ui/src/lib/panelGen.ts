// 第 16 步"提panel":把第 14 步修补图(mid_fill)交给 nano banana(gemini 图像编辑),
// 在纯绿底上原位生成所有 panel(不要背景),顺便精修(对称参考补全缺边)。
import type { DetectionItem } from './detection'

export const PANEL_GEN_MODEL = 'openai/gpt-5.4-image-2'

/** 在图上画红色线框标注(panel bbox),返回标注版 data URL */
export async function annotateWithBoxes(
  srcDataUrl: string,
  boxes: number[][],
): Promise<string> {
  const img = new Image()
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve()
    img.onerror = () => reject(new Error('标注底图加载失败'))
    img.src = srcDataUrl
  })
  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 不可用')
  ctx.drawImage(img, 0, 0)
  ctx.strokeStyle = '#ff0000'
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 400))
  for (const [cx, cy, w, h] of boxes) {
    ctx.strokeRect(
      (cx - w / 2) * canvas.width,
      (cy - h / 2) * canvas.height,
      w * canvas.width,
      h * canvas.height,
    )
  }
  return canvas.toDataURL('image/png')
}

export interface PanelSlot {
  index: number
  /** 模板画布上的槽位矩形 [x, y, w, h](像素) */
  rect: [number, number, number, number]
}

/**
 * 生成"对号入座"槽位模板:绿底画布上按每个 panel 的原始像素尺寸
 * 画出编号空槽(书架式排列)。框线和编号用深绿色——和绿底同色系,
 * 切割时色相键自动清除,不会污染素材。
 */
export function buildSlotTemplate(
  panels: DetectionItem[],
  imageSize: [number, number],
): { dataUrl: string; slots: PanelSlot[]; size: [number, number] } {
  const [iw, ih] = imageSize
  const pad = 48
  const sizes = panels.map((p) => [
    Math.max(12, Math.round(p.bbox[2] * iw)),
    Math.max(12, Math.round(p.bbox[3] * ih)),
  ])
  const W = Math.max(
    1024,
    Math.min(2048, Math.max(...sizes.map((sz) => sz[0])) + pad * 2),
  )
  let x = pad
  let y = pad
  let rowH = 0
  const slots: PanelSlot[] = []
  sizes.forEach(([w, h], i) => {
    if (x + w + pad > W) {
      x = pad
      y += rowH + pad
      rowH = 0
    }
    slots.push({ index: i, rect: [x, y, w, h] })
    x += w + pad
    rowH = Math.max(rowH, h)
  })
  const H = y + rowH + pad

  const canvas = document.createElement('canvas')
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 不可用')
  ctx.fillStyle = 'rgb(0,255,0)'
  ctx.fillRect(0, 0, W, H)
  ctx.strokeStyle = 'rgb(0,140,0)'
  ctx.fillStyle = 'rgb(0,140,0)'
  ctx.lineWidth = 3
  ctx.font = 'bold 20px sans-serif'
  for (const s of slots) {
    const [sx, sy, sw, sh] = s.rect
    ctx.strokeRect(sx, sy, sw, sh)
    ctx.fillText(String(s.index), sx + 4, sy + 22)
  }
  return { dataUrl: canvas.toDataURL('image/png'), slots, size: [W, H] }
}

/**
 * 把 panel 分成最少的"无重叠层"(贪心图着色):
 * 面积大的排前(大 panel 一般在下层),每个 panel 放进第一个与其
 * 无重叠的层;重叠判定带 3px 容差(贴边不算重叠)。
 * 返回每层的 panel 下标数组(下标 = 输入数组顺序)。
 */
export function computePanelLayers(
  panels: DetectionItem[],
  imageSize: [number, number],
  tolPx = 3,
): number[][] {
  const [iw, ih] = imageSize
  const rects = panels.map((p) => {
    const [cx, cy, w, h] = p.bbox
    return [
      (cx - w / 2) * iw + tolPx,
      (cy - h / 2) * ih + tolPx,
      (cx + w / 2) * iw - tolPx,
      (cy + h / 2) * ih - tolPx,
    ]
  })
  const overlap = (a: number[], b: number[]) =>
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
  const order = panels
    .map((p, i) => ({ i, area: p.bbox[2] * p.bbox[3] }))
    .sort((a, b) => b.area - a.area)
    .map((o) => o.i)
  const layers: number[][] = []
  for (const i of order) {
    let placed = false
    for (const layer of layers) {
      if (!layer.some((j) => overlap(rects[i], rects[j]))) {
        layer.push(i)
        placed = true
        break
      }
    }
    if (!placed) layers.push([i])
  }
  layers.forEach((l) => l.sort((a, b) => a - b))
  return layers
}

export interface CanvasFit {
  /** 预设画布尺寸(GPT Image 只输出这些比例) */
  size: [number, number]
  /** 源图 → 画布的等比缩放系数 */
  scale: number
  /** 源图在画布上的偏移(居中) */
  offset: [number, number]
}

/** GPT Image 实测无论输入比例如何都倾向输出正方形——画布固定 1024×1024,
 * 源图等比缩放居中垫绿,输入输出同为方形,链路中不存在任何拉伸环节 */
export function fitToPresetCanvas(imgW: number, imgH: number): CanvasFit {
  const cw = 1024
  const ch = 1024
  const scale = Math.min(cw / imgW, ch / imgH)
  return {
    size: [cw, ch],
    scale,
    offset: [(cw - imgW * scale) / 2, (ch - imgH * scale) / 2],
  }
}

/** 源图等比缩放居中垫到预设画布上(四周纯绿) */
export async function padToCanvas(
  srcDataUrl: string,
  fit: CanvasFit,
): Promise<string> {
  const img = new Image()
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve()
    img.onerror = () => reject(new Error('画布垫图加载失败'))
    img.src = srcDataUrl
  })
  const canvas = document.createElement('canvas')
  canvas.width = fit.size[0]
  canvas.height = fit.size[1]
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 不可用')
  ctx.fillStyle = 'rgb(0,255,0)'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(
    img,
    fit.offset[0],
    fit.offset[1],
    img.naturalWidth * fit.scale,
    img.naturalHeight * fit.scale,
  )
  return canvas.toDataURL('image/png')
}

/** 源图归一化 bbox → 画布归一化 bbox */
export function bboxToCanvas(
  bbox: number[],
  imgW: number,
  imgH: number,
  fit: CanvasFit,
): number[] {
  const [cw, ch] = fit.size
  return [
    (bbox[0] * imgW * fit.scale + fit.offset[0]) / cw,
    (bbox[1] * imgH * fit.scale + fit.offset[1]) / ch,
    (bbox[2] * imgW * fit.scale) / cw,
    (bbox[3] * imgH * fit.scale) / ch,
  ]
}

/** 双色标注:红框 = 本层保留目标,蓝框 = 需剥离的其它 panel */
export async function annotateLayer(
  srcDataUrl: string,
  keepBoxes: number[][],
  stripBoxes: number[][],
): Promise<string> {
  const img = new Image()
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve()
    img.onerror = () => reject(new Error('标注底图加载失败'))
    img.src = srcDataUrl
  })
  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 不可用')
  ctx.drawImage(img, 0, 0)
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 400))
  const draw = (boxes: number[][], color: string) => {
    ctx.strokeStyle = color
    for (const [cx, cy, w, h] of boxes) {
      ctx.strokeRect(
        (cx - w / 2) * canvas.width,
        (cy - h / 2) * canvas.height,
        w * canvas.width,
        h * canvas.height,
      )
    }
  }
  draw(stripBoxes, '#0040ff')
  draw(keepBoxes, '#ff0000')
  return canvas.toDataURL('image/png')
}

/** 单层"原位保留+剥离"生成 */
export async function generatePanelLayer(opts: {
  apiKey: string
  model: string
  prompt: string
  midFillDataUrl: string
  annotatedDataUrl: string
  keep: { index: number; bbox: number[] }[]
  stripCount: number
  layerNo: number
  layerTotal: number
}): Promise<string> {
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${opts.apiKey}`,
      'Content-Type': 'application/json',
      'X-OpenRouter-Title': 'Panel Layer Gen',
    },
    body: JSON.stringify({
      model: opts.model,
      modalities: ['image', 'text'],
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text:
                `${opts.prompt}\n本次为第 ${opts.layerNo}/${opts.layerTotal} 层。` +
                `红框(保留)共 ${opts.keep.length} 个,蓝框(剥离)共 ${opts.stripCount} 个。\n` +
                `保留 panel 列表(归一化 [cx,cy,w,h]):\n${JSON.stringify(opts.keep)}`,
            },
            { type: 'image_url', image_url: { url: opts.midFillDataUrl } },
            { type: 'image_url', image_url: { url: opts.annotatedDataUrl } },
          ],
        },
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
  return extractImage(result)
}

/** 从 OpenRouter 图像输出响应里抠出第一张图的 data URL */
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
  // 兜底:部分模型把图放在 content 数组里
  if (Array.isArray(msg?.content)) {
    for (const part of msg.content as { type?: string; image_url?: { url?: string } }[]) {
      if (part?.image_url?.url) return part.image_url.url
    }
  }
  throw new Error('模型没有返回图片')
}

export async function generatePanels(opts: {
  apiKey: string
  model: string
  prompt: string
  midFillDataUrl: string
  annotatedDataUrl: string
  panels: DetectionItem[]
  /** 源图像素尺寸,用于给每个 panel 报宽高比(ar) */
  imageSize: [number, number]
}): Promise<string> {
  const [iw, ih] = opts.imageSize
  const boxList = opts.panels.map((p, index) => ({
    index,
    bbox: p.bbox,
    // 宽高比 = 像素宽 ÷ 像素高,模型按此作画,严禁偏离
    ar: Math.round(((p.bbox[2] * iw) / Math.max(1, p.bbox[3] * ih)) * 100) / 100,
    // 原始像素尺寸:相对大小与分辨率下限的依据
    px: [Math.round(p.bbox[2] * iw), Math.round(p.bbox[3] * ih)],
  }))
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${opts.apiKey}`,
      'Content-Type': 'application/json',
      'X-OpenRouter-Title': 'Panel Gen',
    },
    body: JSON.stringify({
      model: opts.model || PANEL_GEN_MODEL,
      modalities: ['image', 'text'],
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: `${opts.prompt}\n红框共 ${boxList.length} 个,输出图上必须恰好平铺 ${boxList.length} 个 panel,不多不少;每个 panel 的宽高比必须等于列表中的 ar 值。\npanel 检测框列表(归一化 [cx,cy,w,h],ar=宽÷高,px=原始像素尺寸):\n${JSON.stringify(boxList)}`,
            },
            { type: 'image_url', image_url: { url: opts.midFillDataUrl } },
            { type: 'image_url', image_url: { url: opts.annotatedDataUrl } },
          ],
        },
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
  return extractImage(result)
}
