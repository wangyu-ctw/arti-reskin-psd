// 第 8 步"修正":提取完成后,把源图 + 抠图结果交给 VL 质检,
// 找出底板/背景粘连、过于保守等问题,产出修正 bbox + 正负点,供 SAM2 重抠。
import type { DetectionItem } from './detection'

/** 质检模型:用户指定 gemini 3.1 pro */
export const CUTOUT_FIX_MODEL = 'google/gemini-3.1-pro-preview'

export interface CutoutFix {
  index: number
  issue: string
  /** 修正后的框(已做"只准扩大"合并,完整包含原框) */
  bbox: number[]
  positive_points: number[][]
  negative_points: number[][]
}

const point = {
  type: 'array',
  items: { type: 'number', minimum: 0, maximum: 1 },
  minItems: 2,
  maxItems: 2,
}

export const CUTOUT_FIX_SCHEMA = {
  type: 'object',
  properties: {
    fixes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          index: { type: 'integer', minimum: 0 },
          issue: { type: 'string' },
          bbox: {
            type: 'array',
            items: { type: 'number', minimum: 0, maximum: 1 },
            minItems: 4,
            maxItems: 4,
          },
          positive_points: { type: 'array', items: point, minItems: 0, maxItems: 4 },
          negative_points: { type: 'array', items: point, minItems: 0, maxItems: 6 },
        },
        required: ['index', 'issue', 'bbox', 'positive_points', 'negative_points'],
        additionalProperties: false,
      },
    },
  },
  required: ['fixes'],
  additionalProperties: false,
}

export const CUTOUT_FIX_SYSTEM_PROMPT = `你是游戏 UI 抠图质检员。你会收到两张图和一份 icon 检测框列表(归一化 YOLO 格式 [center_x, center_y, width, height],数值 0～1,附下标 index):
- 图1:抠图的源图;
- 图2:自动抠图结果,透明区域已填充品红色 RGB(255,0,255) 以便识别——品红区就是"被抠掉的部分",非品红区就是"被保留的部分"。
请逐一对照图1,检查图2中每个 icon 的抠图质量,找出以下问题:
1. 底板/背景粘连:保留区混入了 icon 之外的内容——共用面板的底色、卡片格子的一角、邻近元素的碎片。注意:专属底座不算粘连(为该图标量身设计、图形占底板一半以上、离开该图标就没有独立含义的底座,如圆形徽章底、技能框、菱形底座,属于 icon 本体,必须保留);
2. 过于保守:icon 本体被误抠掉了一部分(缺角、缺描边、专属底座被切掉);
3. 附属装饰被抠掉:紧贴或环绕图标的装饰元素——角标、丝带、飘带、缎带、树叶、星星、花纹、边饰等,属于 icon 的组成部分。对照图1逐个确认:图1中该 icon 带有的装饰,在图2中变成品红(被抠掉)的,必须加回——在被抠掉的装饰上给出正点,装饰超出原框的,扩大 bbox 把装饰完整框入;
4. 其它明显异常(整个 icon 没抠出来、抠成了别的东西)。
只对有问题的 icon 输出修正项,每项包含:
- index:该 icon 在列表中的下标;
- issue:一句话说明问题;
- bbox:修正后的框。只准扩大不准缩小:新框必须完整包含原框;框没有问题就原样返回原框;
- positive_points:0~4 个正点,落在"应保留却被抠掉/需要确认保留"的部位(icon 本体、专属底座、附属装饰都算);
- negative_points:0~6 个负点,落在"应抠掉却被保留"的粘连区域(面板底色、邻近碎片);
- 正负点坐标一律为整图归一化 0~1;每个修正项至少要给出一个点(正或负),否则修正无从执行。
没有问题的 icon 不要输出;全部完好则 fixes 为空数组。严禁为凑数虚报问题。`


/** 取图并转 data URL;bg 提供时先平铺到该纯色底上(给 VL 看透明图用) */
export async function fetchImageDataUrl(
  url: string,
  bg?: [number, number, number],
): Promise<string> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`获取图片失败: HTTP ${res.status}`)
  const blob = await res.blob()
  if (!bg) {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = () => reject(new Error('图片编码失败'))
      reader.readAsDataURL(blob)
    })
  }
  const img = await createImageBitmap(blob)
  const canvas = document.createElement('canvas')
  canvas.width = img.width
  canvas.height = img.height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 不可用')
  ctx.fillStyle = `rgb(${bg[0]},${bg[1]},${bg[2]})`
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(img, 0, 0)
  return canvas.toDataURL('image/png')
}

export async function analyzeCutoutFixes(opts: {
  apiKey: string
  sourceDataUrl: string
  cutoutDataUrl: string
  icons: DetectionItem[]
}): Promise<CutoutFix[]> {
  const iconList = opts.icons.map((item, index) => ({ index, bbox: item.bbox }))

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${opts.apiKey}`,
      'Content-Type': 'application/json',
      'X-OpenRouter-Title': 'Cutout QA',
    },
    body: JSON.stringify({
      model: CUTOUT_FIX_MODEL,
      temperature: 0,
      messages: [
        { role: 'system', content: CUTOUT_FIX_SYSTEM_PROMPT },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: `请质检以下 icon 的抠图质量:\n${JSON.stringify(iconList)}\n图1为源图,图2为抠图结果(品红=被抠掉)。`,
            },
            { type: 'image_url', image_url: { url: opts.sourceDataUrl } },
            { type: 'image_url', image_url: { url: opts.cutoutDataUrl } },
          ],
        },
      ],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'cutout_fixes',
          strict: true,
          schema: CUTOUT_FIX_SCHEMA,
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

  let parsed: { fixes?: CutoutFix[] }
  try {
    parsed = JSON.parse(content)
  } catch {
    throw new Error('模型返回的内容不是有效 JSON')
  }
  const fixes = Array.isArray(parsed.fixes) ? parsed.fixes : []

  // 清洗:下标合法、至少一个点、bbox 做"只准扩大"合并(几何权威)
  const cleaned: CutoutFix[] = []
  for (const f of fixes) {
    if (f.index < 0 || f.index >= opts.icons.length) continue
    if (!f.positive_points.length && !f.negative_points.length) continue
    const orig = opts.icons[f.index].bbox
    const union = (a: number[], b: number[]) => {
      const ax0 = a[0] - a[2] / 2, ay0 = a[1] - a[3] / 2
      const ax1 = a[0] + a[2] / 2, ay1 = a[1] + a[3] / 2
      const bx0 = b[0] - b[2] / 2, by0 = b[1] - b[3] / 2
      const bx1 = b[0] + b[2] / 2, by1 = b[1] + b[3] / 2
      const x0 = Math.min(ax0, bx0), y0 = Math.min(ay0, by0)
      const x1 = Math.max(ax1, bx1), y1 = Math.max(ay1, by1)
      return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    }
    cleaned.push({
      ...f,
      bbox: Array.isArray(f.bbox) && f.bbox.length === 4 ? union(orig, f.bbox) : orig,
    })
  }
  return cleaned
}
