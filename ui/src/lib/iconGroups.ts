// 第 8+ 步"素材化"的分组:通过 OpenRouter 调 VL 模型,识别每个 icon 是什么,
// 并把"相同的图标"(游戏币等资源、列表复用等)归为一组,组名/slug 供素材命名。
import type { DetectionItem } from './detection'

/** 一组相同的图标:name 描述它是什么/干什么(不含位置),bbox 是组内所有成员的检测框 */
export interface IconGroup {
  name: string
  /** 素材文件名用的唯一英文名(小写/数字/下划线),全组唯一 */
  slug: string
  /** 组内成员的检测框(归一化 [cx,cy,w,h]),与检测结果原值一致 */
  bbox: number[][]
  /** 组内成员在输入 icon 列表中的下标(前端回填 bbox 用,也便于溯源) */
  indices: number[]
}

// 模型只返回分组下标,bbox 由前端用检测框原值回填——
// 坐标不过模型的手,杜绝抄写误差(几何权威原则的延续)
export const ICON_GROUP_SCHEMA = {
  type: 'object',
  properties: {
    groups: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          slug: { type: 'string', pattern: '^[a-z0-9_]+$' },
          indices: {
            type: 'array',
            items: { type: 'integer', minimum: 0 },
            minItems: 1,
          },
        },
        required: ['name', 'slug', 'indices'],
        additionalProperties: false,
      },
    },
  },
  required: ['groups'],
  additionalProperties: false,
}

export async function analyzeIconGroups(opts: {
  apiKey: string
  model: string
  temperature: number
  systemPrompt: string
  userPrompt: string
  imageDataUrl: string
  icons: DetectionItem[]
}): Promise<IconGroup[]> {
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
          name: 'icon_groups',
          strict: true,
          schema: ICON_GROUP_SCHEMA,
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

  let parsed: { groups?: { name: string; slug: string; indices: number[] }[] }
  try {
    parsed = JSON.parse(content)
  } catch {
    throw new Error('模型返回的内容不是有效 JSON')
  }
  const groups = parsed.groups
  if (!Array.isArray(groups) || groups.length === 0) {
    throw new Error('模型没有返回 groups 数组')
  }

  // slug 唯一性校验
  const slugs = new Set<string>()
  for (const g of groups) {
    if (slugs.has(g.slug)) throw new Error(`素材名 slug 重复:${g.slug}`)
    slugs.add(g.slug)
  }

  // 完整性校验:每个输入下标必须恰好出现一次(不多、不少、不重复)
  const seen = new Set<number>()
  for (const g of groups) {
    for (const idx of g.indices) {
      if (idx < 0 || idx >= opts.icons.length) {
        throw new Error(`分组里出现了不存在的下标 ${idx}`)
      }
      if (seen.has(idx)) {
        throw new Error(`下标 ${idx} 被分进了多个组`)
      }
      seen.add(idx)
    }
  }
  if (seen.size !== opts.icons.length) {
    const missing = opts.icons
      .map((_, i) => i)
      .filter((i) => !seen.has(i))
    throw new Error(`有 ${missing.length} 个 icon 未被分组:下标 ${missing.join(', ')}`)
  }

  // bbox 用检测框原值回填
  return groups.map((g) => ({
    name: g.name,
    slug: g.slug,
    indices: g.indices,
    bbox: g.indices.map((i) => opts.icons[i].bbox),
  }))
}
