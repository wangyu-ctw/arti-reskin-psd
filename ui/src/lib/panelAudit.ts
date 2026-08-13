// 第 16 步"panel修正":沿用结构化检测的 panel,粗排层级后交给 Gemini VL
// 审核——纠正/删补 bbox、修层级、分 banner/title/panel、判透明类型。
import { extractResponseText } from './detection'

export type PanelKind = 'panel' | 'banner' | 'title'
export type PanelTransparency =
  | 'opaque'
  | 'uniform'
  | 'frame_solid'
  | 'center_solid'

/** 被删除的检测项及理由 */
export interface PanelAuditDeletion {
  source_index: number
  reason: string
}

export interface PanelAuditItem {
  /** 原检测下标;-1 = Gemini 新补的 */
  source_index: number
  bbox: number[]
  /** 层级:越大越靠上;同层互不重叠可同值 */
  z: number
  kind: PanelKind
  transparency: PanelTransparency
  /** 一句话说明(修了什么/为什么补) */
  note: string
}

/** 粗略层级排序:包含→内者在上;相交→面积小者在上;z 从 0(最底)起。
 * 实现:按面积降序为底序(大的在下),再用包含关系做一遍冒泡修正。 */
export function roughPanelZ(bboxes: number[][]): number[] {
  const n = bboxes.length
  const contains = (a: number[], b: number[]) => {
    // a 完整包含 b(留 2% 容差)
    const tol = 0.02
    return (
      a[0] - a[2] / 2 - tol <= b[0] - b[2] / 2 &&
      a[0] + a[2] / 2 + tol >= b[0] + b[2] / 2 &&
      a[1] - a[3] / 2 - tol <= b[1] - b[3] / 2 &&
      a[1] + a[3] / 2 + tol >= b[1] + b[3] / 2
    )
  }
  const order = bboxes
    .map((b, i) => ({ i, area: b[2] * b[3] }))
    .sort((x, y) => y.area - x.area)
    .map((o) => o.i)
  // 包含修正:容器必须排在内容物之前(更靠底)
  for (let pass = 0; pass < n; pass++) {
    let moved = false
    for (let a = 0; a < order.length; a++)
      for (let b = 0; b < a; b++)
        if (contains(bboxes[order[a]], bboxes[order[b]])) {
          const [outer] = order.splice(a, 1)
          order.splice(b, 0, outer)
          moved = true
        }
    if (!moved) break
  }
  const z = new Array<number>(n).fill(0)
  order.forEach((idx, pos) => {
    z[idx] = pos
  })
  return z
}

const PANEL_AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    panels: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          source_index: { type: 'integer', minimum: -1 },
          bbox: {
            type: 'array',
            items: { type: 'number', minimum: 0, maximum: 1 },
            minItems: 4,
            maxItems: 4,
          },
          z: { type: 'integer', minimum: 0 },
          kind: { type: 'string', enum: ['panel', 'banner', 'title'] },
          transparency: {
            type: 'string',
            enum: ['opaque', 'uniform', 'frame_solid', 'center_solid'],
          },
          note: { type: 'string' },
        },
        required: ['source_index', 'bbox', 'z', 'kind', 'transparency', 'note'],
        additionalProperties: false,
      },
    },
    deleted: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          source_index: { type: 'integer', minimum: 0 },
          reason: { type: 'string' },
        },
        required: ['source_index', 'reason'],
        additionalProperties: false,
      },
    },
  },
  required: ['panels', 'deleted'],
  additionalProperties: false,
} as const

/** Gemini VL 审核 panel 清单:传第 15 步修补图 + 检测项(含粗排 z) */
export async function auditPanels(opts: {
  apiKey: string
  model: string
  temperature: number
  systemPrompt: string
  userPrompt: string
  /** 图1:去件图(mid_fill),bbox 以它为准 */
  imageDataUrl: string
  /** 图2:原图,辅助判断"承载体"(原来载着内容、现在还留在画面上) */
  originDataUrl?: string
  panels: { index: number; bbox: number[]; z: number }[]
  signal?: AbortSignal
}): Promise<{ panels: PanelAuditItem[]; deleted: PanelAuditDeletion[] }> {
  const headers = {
    Authorization: `Bearer ${opts.apiKey}`,
    'Content-Type': 'application/json',
    'X-OpenRouter-Title': 'Panel Audit',
  }
  const userText = `${opts.userPrompt}\n${JSON.stringify(opts.panels)}`
  // GPT(gpt-5.6 系)只在 responses API 有端点(与第 3 步检测同通道);
  // gemini 系走 chat/completions
  const useResponses = opts.model.startsWith('openai/')
  const response = useResponses
    ? await fetch('https://openrouter.ai/api/v1/responses', {
        method: 'POST',
        signal: opts.signal,
        headers,
        body: JSON.stringify({
          model: opts.model,
          instructions: opts.systemPrompt,
          input: [
            {
              type: 'message',
              role: 'user',
              content: [
                { type: 'input_text', text: userText },
                { type: 'input_image', image_url: opts.imageDataUrl,
                  detail: 'auto' },
                ...(opts.originDataUrl
                  ? [{ type: 'input_image', image_url: opts.originDataUrl,
                       detail: 'auto' }]
                  : []),
              ],
            },
          ],
          text: {
            format: {
              type: 'json_schema',
              name: 'panel_audit',
              strict: true,
              schema: PANEL_AUDIT_SCHEMA,
            },
          },
          store: false,
          stream: false,
        }),
      })
    : await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        signal: opts.signal,
        headers,
        body: JSON.stringify({
          model: opts.model,
          temperature: opts.temperature,
          messages: [
            { role: 'system', content: opts.systemPrompt },
            {
              role: 'user',
              content: [
                { type: 'text', text: userText },
                { type: 'image_url', image_url: { url: opts.imageDataUrl } },
                ...(opts.originDataUrl
                  ? [{ type: 'image_url',
                       image_url: { url: opts.originDataUrl } }]
                  : []),
              ],
            },
          ],
          response_format: {
            type: 'json_schema',
            json_schema: {
              name: 'panel_audit',
              strict: true,
              schema: PANEL_AUDIT_SCHEMA,
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
  const result = (await response.json()) as {
    error?: { message?: string }
    choices?: { message?: { content?: string } }[]
  }
  if (result.error) {
    throw new Error(result.error.message || 'OpenRouter 返回了失败状态')
  }
  const content = useResponses
    ? extractResponseText(result)
    : result.choices?.[0]?.message?.content
  if (!content) throw new Error('模型没有返回内容')
  let parsed: { panels?: PanelAuditItem[]; deleted?: PanelAuditDeletion[] }
  try {
    parsed = JSON.parse(content)
  } catch {
    throw new Error('模型返回的内容不是有效 JSON')
  }
  const panels = parsed.panels
  if (!Array.isArray(panels) || panels.length === 0) {
    throw new Error('模型没有返回 panels 数组')
  }
  return {
    panels: [...panels].sort((a, b) => a.z - b.z),
    deleted: parsed.deleted ?? [],
  }
}

/** 审核结果画框预览:kind 分色,标 z 与来源 */
export async function drawAuditOverlay(
  imageUrl: string,
  items: PanelAuditItem[],
): Promise<string> {
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const im = new Image()
    im.onload = () => resolve(im)
    im.onerror = () => reject(new Error('底图加载失败'))
    im.src = imageUrl
  })
  const W = img.naturalWidth
  const H = img.naturalHeight
  const canvas = document.createElement('canvas')
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 不可用')
  ctx.drawImage(img, 0, 0)
  const colors: Record<PanelKind, string> = {
    panel: '#ff2d2d',
    banner: '#ff9500',
    title: '#1677ff',
  }
  ctx.lineWidth = Math.max(2, Math.round(W / 400))
  ctx.font = `bold ${Math.max(12, Math.round(W / 55))}px sans-serif`
  for (const it of items) {
    const [cx, cy, w, h] = it.bbox
    const x = (cx - w / 2) * W
    const y = (cy - h / 2) * H
    ctx.strokeStyle = colors[it.kind]
    ctx.fillStyle = colors[it.kind]
    ctx.strokeRect(x, y, w * W, h * H)
    const tag =
      `z${it.z} ${it.kind}` +
      (it.transparency !== 'opaque' ? ` ${it.transparency}` : '') +
      (it.source_index < 0 ? ' +新' : ` #${it.source_index}`)
    ctx.fillText(tag, x + 4, y + Math.max(14, Math.round(W / 55)))
  }
  return canvas.toDataURL('image/png')
}
