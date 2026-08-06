/**
 * 文字层还原:原图与去字图逐像素取差值——差异大的像素就是被去掉的文字,
 * RGB 取原图像素,alpha 由差值幅度映射(带噪声门限:去字图是整图重生成,
 * 全图都有轻微像素漂移,低于门限的差值一律视为噪声压成全透明)。
 */
const NOISE_FLOOR = 26 // Chebyshev 差值低于此视为重生成噪声
const ALPHA_GAIN = 5 // (diff - NOISE_FLOOR) * GAIN → alpha,快速拉到不透明

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('图片加载失败'))
    img.src = url
  })
}

export async function extractTextFront(
  originUrl: string,
  textBackUrl: string,
): Promise<Blob> {
  const [origin, textBack] = await Promise.all([
    loadImage(originUrl),
    loadImage(textBackUrl),
  ])
  const w = origin.naturalWidth
  const h = origin.naturalHeight

  const draw = (img: HTMLImageElement) => {
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')!
    ctx.imageSmoothingQuality = 'high'
    // 去字图经过 16 对齐缩放,尺寸可能与原图不同,统一拉回原图尺寸
    ctx.drawImage(img, 0, 0, w, h)
    return ctx
  }
  const originCtx = draw(origin)
  const backData = draw(textBack).getImageData(0, 0, w, h).data
  const im = originCtx.getImageData(0, 0, w, h)
  const d = im.data

  for (let i = 0; i < d.length; i += 4) {
    const diff = Math.max(
      Math.abs(d[i] - backData[i]),
      Math.abs(d[i + 1] - backData[i + 1]),
      Math.abs(d[i + 2] - backData[i + 2]),
    )
    d[i + 3] = diff <= NOISE_FLOOR ? 0 : Math.min(255, (diff - NOISE_FLOOR) * ALPHA_GAIN)
  }
  originCtx.putImageData(im, 0, 0)

  return new Promise((resolve, reject) => {
    originCtx.canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('文字层导出失败'))),
      'image/png',
    )
  })
}
