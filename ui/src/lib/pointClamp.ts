import type { AnalyzedIcon } from './iconAnalysis'

/**
 * 取点域钳制:对 Gemini 返回的正负点做纯几何后处理,强制落入规则定义的合法域。
 * 不看图、不增删点,只把"看对了但坐标写歪了"的点投影回它本来想去的域:
 *
 * 框局部坐标 dx=(x-cx)/w, dy=(y-cy)/h(框内 |dx|,|dy|≤0.5):
 * - 中央菱形(|dx|+|dy| ≤ 0.5)= 图形主体的几何代理,负点禁入;
 * - 角点合法域 = 框内且菱形外的四个角落三角区;
 * - 边点合法域 = 框外、越出框边 5%~15% 的环带;
 * - 正点合法域 = 框内、距框边 ≥10% 的内缩区(贴边正点拉回)。
 */

const DIAMOND = 0.5 // 菱形边界:|dx|+|dy| = 0.5
const TRI_TARGET = 0.56 // 投影出菱形后的目标 L1 半径(略过界,留安全余量)
const TRI_CENTROID = 1 / 3 // 角落三角区质心坐标(±1/3, ±1/3),L1=0.667
const EDGE_MIN = 0.05 // 边点越界下限(相对框宽高)
const EDGE_MAX = 0.15 // 边点越界上限
const POS_INSET = 0.4 // 正点内缩边界(=距框边 10%)

function clampNeg(dx: number, dy: number, idx: number): [number, number, boolean] {
  const inBoxX = Math.abs(dx) <= 0.5
  const inBoxY = Math.abs(dy) <= 0.5

  if (inBoxX && inBoxY) {
    // 框内:角点候选。菱形内 → 沿所在象限向角落三角区质心混合投影
    const r = Math.abs(dx) + Math.abs(dy)
    if (r > DIAMOND) return [dx, dy, false] // 已在三角区,合法
    // 象限符号(轴上点用序号轮转打散,避免多个点挤到同一角)
    const sx = dx !== 0 ? Math.sign(dx) : idx % 2 === 0 ? 1 : -1
    const sy = dy !== 0 ? Math.sign(dy) : idx % 4 < 2 ? 1 : -1
    const cx0 = sx * TRI_CENTROID
    const cy0 = sy * TRI_CENTROID
    // 与质心同象限时 L1 沿混合线性:解出恰好越过 TRI_TARGET 的混合系数
    const rc = Math.abs(cx0) + Math.abs(cy0) // = 0.667
    const rp = Math.abs(dx) * (Math.sign(dx) === sx ? 1 : -1) +
               Math.abs(dy) * (Math.sign(dy) === sy ? 1 : -1)
    const t = Math.min(1, Math.max(0, (TRI_TARGET - rp) / (rc - rp)))
    return [dx + t * (cx0 - dx), dy + t * (cy0 - dy), true]
  }

  // 框外:边点。取越界主导轴,越界量压进 [5%, 15%] 环带,交叉轴收回框投影范围内
  const ox = Math.max(0, Math.abs(dx) - 0.5)
  const oy = Math.max(0, Math.abs(dy) - 0.5)
  let nx = dx
  let ny = dy
  let moved = false
  if (ox >= oy) {
    const o = Math.min(EDGE_MAX, Math.max(EDGE_MIN, ox))
    if (o !== ox) moved = true
    nx = Math.sign(dx) * (0.5 + o)
    if (Math.abs(ny) > 0.45) {
      ny = Math.sign(ny) * 0.45 // 斜出角落的拉回边中线方向
      moved = true
    }
  } else {
    const o = Math.min(EDGE_MAX, Math.max(EDGE_MIN, oy))
    if (o !== oy) moved = true
    ny = Math.sign(dy) * (0.5 + o)
    if (Math.abs(nx) > 0.45) {
      nx = Math.sign(nx) * 0.45
      moved = true
    }
  }
  return [nx, ny, moved]
}

function clampPos(dx: number, dy: number): [number, number, boolean] {
  const nx = Math.min(POS_INSET, Math.max(-POS_INSET, dx))
  const ny = Math.min(POS_INSET, Math.max(-POS_INSET, dy))
  return [nx, ny, nx !== dx || ny !== dy]
}

/** 返回钳制后的新数组与被修正的点数。 */
export function clampIconPoints(icons: AnalyzedIcon[]): {
  icons: AnalyzedIcon[]
  moved: number
} {
  let moved = 0
  const out = icons.map((icon) => {
    const [cx, cy, w, h] = (icon.bbox ?? []).map(Number)
    if (![cx, cy, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return icon

    const toLocal = (p: number[]) => [(p[0] - cx) / w, (p[1] - cy) / h] as const
    const toGlobal = (dx: number, dy: number) => [
      Math.min(1, Math.max(0, cx + dx * w)),
      Math.min(1, Math.max(0, cy + dy * h)),
    ]

    const neg = (icon.negative_points ?? []).map((p, i) => {
      const [dx, dy] = toLocal(p)
      const [nx, ny, m] = clampNeg(dx, dy, i)
      if (m) moved += 1
      return m ? toGlobal(nx, ny) : p
    })
    const pos = (icon.positive_points ?? []).map((p) => {
      const [dx, dy] = toLocal(p)
      const [nx, ny, m] = clampPos(dx, dy)
      if (m) moved += 1
      return m ? toGlobal(nx, ny) : p
    })
    return { ...icon, negative_points: neg, positive_points: pos }
  })
  return { icons: out, moved }
}
