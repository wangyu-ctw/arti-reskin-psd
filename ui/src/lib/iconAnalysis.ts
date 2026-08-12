// 第 6 步"分析icon":通过 OpenRouter 调 Gemini,分析 icon 图形、审核 bbox、生成 SAM2 正负点。
import type { DetectionItem } from './detection'

export interface AnalyzedIcon {
  index: number
  description: string
  positive_points: number[][]
  negative_points: number[][]
  /** 图形本体四周是否被光影效果包裹(外发光/光晕/投影围了一圈);
   * 为 true 时第 8 步提取同时采纳正负点,把光效切干净 */
  has_overflow_glow?: boolean
  /** 不再由模型返回:分析完成后由前端用检测框原值回填,供下游提取使用 */
  bbox: number[]
  /** 历史字段:轮廓核对已从 Gemini 职责中移除,新分析不再返回 */
  bbox_accurate?: boolean
  /** 历史字段:判删已从 Gemini 职责中移除,新分析不再返回 */
  should_delete?: boolean
}

const point = {
  type: 'array',
  items: { type: 'number', minimum: 0, maximum: 1 },
  minItems: 2,
  maxItems: 2,
}

export const ICON_ANALYSIS_SCHEMA = {
  type: 'object',
  properties: {
    icons: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          index: { type: 'integer', minimum: 0 },
          description: { type: 'string' },
          positive_points: { type: 'array', items: point, minItems: 1, maxItems: 3 },
          // minItems 抬高到 8:模型习惯贴下限给点(3→只给 3~4,6→只给 6;
          // 加 has_overflow_glow 字段后软约束顶不住,直接抬硬下限)
          negative_points: { type: 'array', items: point, minItems: 8, maxItems: 10 },
          has_overflow_glow: { type: 'boolean' },
        },
        required: ['index', 'description', 'positive_points', 'negative_points',
                   'has_overflow_glow'],
        additionalProperties: false,
      },
    },
  },
  required: ['icons'],
  additionalProperties: false,
}

/** 把 run 目录里的图片取回并转成 data URL(发给多模态模型用) */
export async function fetchImageAsDataUrl(url: string): Promise<string> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`获取图片失败: HTTP ${res.status}`)
  const blob = await res.blob()
  return await new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(new Error('图片编码失败'))
    reader.readAsDataURL(blob)
  })
}

export async function analyzeIcons(opts: {
  apiKey: string
  model: string
  temperature: number
  systemPrompt: string
  userPrompt: string
  imageDataUrl: string
  icons: DetectionItem[]
}): Promise<AnalyzedIcon[]> {
  const iconList = opts.icons.map((item, index) => ({ index, bbox: item.bbox }))

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${opts.apiKey}`,
      'Content-Type': 'application/json',
      'X-OpenRouter-Title': 'Icon Analysis',
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
              text: `${opts.userPrompt}\n${JSON.stringify(iconList)}`,
            },
            { type: 'image_url', image_url: { url: opts.imageDataUrl } },
          ],
        },
      ],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'icon_analysis',
          strict: true,
          schema: ICON_ANALYSIS_SCHEMA,
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
  if (result?.error) {
    throw new Error(result.error.message || 'OpenRouter 返回了失败状态')
  }
  const content = result?.choices?.[0]?.message?.content
  if (!content) throw new Error('模型没有返回内容')

  let parsed: { icons?: AnalyzedIcon[] }
  try {
    parsed = JSON.parse(content)
  } catch {
    throw new Error('模型返回的内容不是有效 JSON')
  }
  const icons = parsed.icons
  if (!Array.isArray(icons) || icons.length === 0) {
    throw new Error('模型没有返回 icons 数组')
  }
  if (icons.length !== opts.icons.length) {
    throw new Error(
      `分析结果数量(${icons.length})与输入 icon 数量(${opts.icons.length})不一致`,
    )
  }
  return icons
}
