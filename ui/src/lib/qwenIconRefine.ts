/**
 * 绿底精修实验:把 icon 裁块发给 qwen-image-3(OpenRouter 图像生成),
 * 让它在纯绿底(#00FF00)上重绘并精修 icon,前端抠绿得到透明 PNG,与 SAM2 结果对比。
 */

// OpenRouter 无 qwen 图像输出模型;可选:google/gemini-3-pro-image(质量优先)、
// google/gemini-3.1-flash-image(快且便宜)、openai/gpt-5.4-image-2
export const QWEN_IMAGE_MODEL = 'google/gemini-3-pro-image'

/** 整图版提示词:一次生成,除 icon 外全部涂纯绿,位置布局原样保留,之后按已知 bbox 自行裁切。 */
export function greenFullPrompt(iconCount: number): string {
  return `Recreate this game UI screenshot at the same size and layout, but keep ONLY the icons — every other pixel must become flat, uniform pure green (#00FF00).
Strict rules:
- Icons are the small functional symbols/pictograms (coins, gems, gears, arrows, close buttons, rarity badges, plus signs, etc.). There are about ${iconCount} of them. Keep EVERY icon at its exact original position and scale — do not move, resize or rearrange anything.
- This is restoration, not reinterpretation: each icon keeps its exact design, shape, colors, gradients and outline. Repair edges that were occluded by other UI elements.
- Everything that is not an icon becomes pure green: panels, backgrounds, bars, buttons' base plates, text areas, and especially the circular sockets / dark backing plates / recesses BEHIND icons, and any shadows cast on the background. An icon's own built-in outline or glow stays; the backing plate under it goes green.
- The green must be flat #00FF00 everywhere, no gradients or textures.
Output a single image with the same aspect ratio as the input.`
}

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('图片加载失败'))
    img.src = url
  })
}

/** 按归一化 bbox 从整图裁块(外扩 padRatio),小图放大到长边 ≥ minOut 提升生成质量。 */
export function cropByBbox(
  img: HTMLImageElement,
  bbox: number[],
  padRatio = 0.12,
  minOut = 448,
): string {
  const [cx, cy, w, h] = bbox
  const bw = w * img.naturalWidth
  const bh = h * img.naturalHeight
  const pad = Math.max(bw, bh) * padRatio
  const x1 = Math.max(0, (cx - w / 2) * img.naturalWidth - pad)
  const y1 = Math.max(0, (cy - h / 2) * img.naturalHeight - pad)
  const x2 = Math.min(img.naturalWidth, (cx + w / 2) * img.naturalWidth + pad)
  const y2 = Math.min(img.naturalHeight, (cy + h / 2) * img.naturalHeight + pad)
  const cw = Math.max(1, x2 - x1)
  const ch = Math.max(1, y2 - y1)
  const scale = Math.max(1, minOut / Math.max(cw, ch))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(cw * scale)
  canvas.height = Math.round(ch * scale)
  const ctx = canvas.getContext('2d')!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(img, x1, y1, cw, ch, 0, 0, canvas.width, canvas.height)
  return canvas.toDataURL('image/png')
}

/** 整图降采样(控制输入体积/费用),返回 dataURL。 */
export function downscaleImage(img: HTMLImageElement, maxSide = 1600): string {
  const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(img.naturalWidth * scale)
  canvas.height = Math.round(img.naturalHeight * scale)
  const ctx = canvas.getContext('2d')!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
  return canvas.toDataURL('image/png')
}

/** 调图像模型按提示词编辑图片,返回输出图的 dataURL。 */
export async function qwenGreenRefine(
  apiKey: string,
  cropDataUrl: string,
  prompt: string,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    signal,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: QWEN_IMAGE_MODEL,
      modalities: ['image', 'text'],
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            { type: 'image_url', image_url: { url: cropDataUrl } },
          ],
        },
      ],
    }),
  })
  if (!response.ok) {
    throw new Error(`qwen-image-3 请求失败:HTTP ${response.status} ${await response.text()}`)
  }
  const data = (await response.json()) as {
    choices?: { message?: { images?: { image_url?: { url?: string } }[] } }[]
  }
  const url = data.choices?.[0]?.message?.images?.[0]?.image_url?.url
  if (!url) throw new Error('响应中没有图片(模型可能不支持图像输出)')
  return url
}

/** 抠绿:绿色优势像素变透明,边缘半透明过渡 + 去绿溢色。 */
export async function removeGreen(dataUrl: string): Promise<string> {
  const img = await loadImage(dataUrl)
  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(img, 0, 0)
  const im = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const d = im.data
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2]
    const diff = g - Math.max(r, b)
    if (diff > 48) {
      d[i + 3] = 0
    } else if (diff > 16) {
      // 边缘过渡:部分透明 + 压掉绿溢色
      d[i + 3] = Math.round(d[i + 3] * (1 - (diff - 16) / 32))
      d[i + 1] = Math.max(r, b)
    } else if (diff > 4) {
      d[i + 1] = Math.max(r, b, g - diff) // 仅去溢色
    }
  }
  ctx.putImageData(im, 0, 0)
  return canvas.toDataURL('image/png')
}
