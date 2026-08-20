import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import { Card, Input, Pagination, Radio, Spin, Tag } from 'antd'

const CHECKERBOARD: CSSProperties = {
  backgroundImage:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

type SampleMeta = Record<string, unknown>

interface Sample {
  stem: string
  split: string
}

interface ViewerColumn {
  key: string
  label: string
  /** 该列的文件名;null = 此样本无该层(显示"空") */
  file: (meta: SampleMeta) => string | null
  /** 图下方的小字标注(如覆盖率) */
  note?: (meta: SampleMeta) => string | null
}

export interface ViewerConfig {
  title: string
  /** ~/Desktop/训练数据 下的数据集目录名 */
  dataset: string
  columns: ViewerColumn[]
  /** 首列样本名下方的信息行 */
  sampleInfo: (meta: SampleMeta) => string
}

// ---- 六槽整图分解(scripts/make_six_slot_data.py) ----
const SIX_SLOTS = ['bg', 'panel', 'controls', 'assets', 'panel_f', 'icon', 'text']
export const sixSlotConfig: ViewerConfig = {
  title: '六槽分层数据浏览(v3 · panel/panel_f 按堆叠分界)',
  dataset: 'six_slot_v3',
  columns: SIX_SLOTS.map((slot, i) => ({
    key: slot,
    label: slot === 'controls' ? 'controls(bar+button)' : slot,
    file: (meta) =>
      (meta.present as Record<string, boolean>)[slot]
        ? `layer_0${i}_${slot}.png`
        : null,
    note: (meta) =>
      `覆盖 ${((meta.coverage as Record<string, number>)[slot] * 100).toFixed(1)}%`,
  })),
  sampleInfo: (m) => {
    const [w, h] = m.size as number[]
    return `${w}×${h} · 叠回误差 ${m.recompose_error as number}`
  },
}

// ---- panel z 分层(scripts/make_layered_panel_data.py) ----
const PANEL_MAX_Z = 7
export const panelZConfig: ViewerConfig = {
  title: 'panel 分层数据浏览(layered_panel_v3 · 自底向上紧凑)',
  dataset: 'layered_panel_v3',
  columns: [
    {
      key: 'bg',
      label: 'bg',
      file: () => 'layer_00_bg.png',
    },
    ...Array.from({ length: PANEL_MAX_Z }, (_, i) => ({
      key: `z${i}`,
      label: `panel z${i}`,
      file: (meta: SampleMeta) =>
        i < (meta.levels as number)
          ? `layer_${String(i + 1).padStart(2, '0')}_panel.png`
          : null,
    })),
  ],
  sampleInfo: (m) => {
    const [w, h] = m.size as number[]
    return `${w}×${h} · ${m.levels as number} 层 ${m.panels as number} 块 · 叠回误差 ${m.recompose_error as number}`
  },
}

// ---- panelz v2 LoRA 推理结果(infer_panelz_val.py):固定 bg+z0..z4 全帧输出 ----
export const panelZPredConfig: ViewerConfig = {
  title: 'panelz 推理结果(val · 无后缀=v2,@v3=自底向上紧凑)',
  dataset: 'panelz_v2_pred',
  columns: [
    {
      key: 'bg',
      label: 'bg',
      file: () => 'layer_00_bg.png',
      note: (m) => `覆盖 ${(((m.coverage as Record<string, number>).bg ?? 0) * 100).toFixed(1)}%`,
    },
    ...Array.from({ length: 5 }, (_, i) => ({
      key: `z${i}`,
      label: `panel z${i}`,
      file: () => `layer_${String(i + 1).padStart(2, '0')}_panel.png`,
      note: (m: SampleMeta) =>
        `覆盖 ${(((m.coverage as Record<string, number>)[`z${i}`] ?? 0) * 100).toFixed(1)}%`,
    })),
  ],
  sampleInfo: (m) => {
    const [w, h] = m.size as number[]
    return `${w}×${h} · ${m.elapsed_sec as number}s · steps ${m.steps as number} · cfg ${m.true_cfg as number} · seed ${m.seed as number}`
  },
}

// ---- 六槽 LoRA 推理结果(infer_six_slot_val.py,与训练数据同构) ----
export const sixSlotPredConfig: ViewerConfig = {
  ...sixSlotConfig,
  title: '六槽推理结果(checkpoint-3000 · val)',
  dataset: 'six_slot_pred',
  sampleInfo: (m) => {
    const [w, h] = m.size as number[]
    return `${w}×${h} · ${m.elapsed_sec as number}s · steps ${m.steps as number} · cfg ${m.true_cfg as number} · seed ${m.seed as number}`
  },
}

const PAGE_SIZE = 20

/**
 * 训练数据浏览通用页(仿 seedseek 表格):行=样本、首列=composite 输入、
 * 其余列由 config 定义,20 组/页。数据经 /__train-data 直接读本机
 * ~/Desktop/训练数据/<dataset>(vite dev 中间件,不经 RunPod)。
 */
export default function TrainDataViewerPage({ config }: { config: ViewerConfig }) {
  const [samples, setSamples] = useState<Sample[] | null>(null)
  const [loadError, setLoadError] = useState('')
  const [filter, setFilter] = useState('')
  const [splitFilter, setSplitFilter] = useState<'all' | 'val' | 'train'>('all')
  const [page, setPage] = useState(1)
  const [metas, setMetas] = useState<Record<string, SampleMeta>>({})

  const fileUrl = (s: Sample, file: string) =>
    `/__train-data/${config.dataset}/${s.split}/${encodeURIComponent(s.stem)}/${file}`
  const sampleKey = (s: Sample) => `${s.split}/${s.stem}`

  useEffect(() => {
    fetch(`/__train-data/${config.dataset}/list`)
      .then(async (r) => {
        const data = (await r.json()) as { samples?: Sample[]; error?: string }
        if (!r.ok || !data.samples) throw new Error(data.error ?? `HTTP ${r.status}`)
        setSamples(data.samples)
      })
      .catch((e) => setLoadError(e instanceof Error ? e.message : String(e)))
  }, [config.dataset])

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase()
    if (!samples) return []
    return samples.filter(
      (s) =>
        (splitFilter === 'all' || s.split === splitFilter) &&
        (!q || s.stem.toLowerCase().includes(q)),
    )
  }, [samples, filter, splitFilter])

  const pageSamples = useMemo(
    () => filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE),
    [filtered, page],
  )

  // 当前页样本的 meta 按需加载(层列的取舍依赖 meta,加载完才渲染格子)
  useEffect(() => {
    for (const s of pageSamples) {
      if (metas[sampleKey(s)]) continue
      fetch(fileUrl(s, 'meta.json'))
        .then((r) => r.json())
        .then((m: SampleMeta) =>
          setMetas((prev) => ({ ...prev, [sampleKey(s)]: m })))
        .catch(() => {})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageSamples.map(sampleKey).join('|')])

  const pager = samples ? (
    <div className="flex items-center gap-4">
      <Pagination
        current={page}
        pageSize={PAGE_SIZE}
        total={filtered.length}
        showSizeChanger={false}
        showQuickJumper
        onChange={setPage}
      />
      <span className="text-[12px] text-black/45">
        共 {filtered.length} 组
        {filtered.length !== samples.length ? `(已过滤,总 ${samples.length})` : ''}
      </span>
    </div>
  ) : null

  return (
    <main className="min-h-screen bg-[#f5f6f8] p-4">
      <Card
        title={<span className="text-[16px] font-bold">{config.title}</span>}
        extra={
          <a href="/" className="text-[13px]">
            返回主流程
          </a>
        }
        className="mb-4 shadow-sm"
      >
        <div className="flex items-center gap-4">
          <Radio.Group
            optionType="button"
            buttonStyle="solid"
            value={splitFilter}
            onChange={(e) => {
              setSplitFilter(e.target.value)
              setPage(1)
            }}
            options={[
              { value: 'all', label: '全部' },
              { value: 'val', label: 'val' },
              { value: 'train', label: 'train' },
            ]}
          />
          <Input
            className="w-80"
            allowClear
            placeholder="按名称过滤(如 binan / 副本 / U132)"
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value)
              setPage(1)
            }}
          />
          {pager}
        </div>
      </Card>

      {loadError ? (
        <Card className="shadow-sm">
          <div className="text-[13px] text-[#cf1322]">
            数据加载失败：{loadError}(dev server 读的是本机
            ~/Desktop/训练数据/{config.dataset},根目录可用 TRAIN_DATA_DIR 环境变量改)
          </div>
        </Card>
      ) : !samples ? (
        <Card className="shadow-sm">
          <div className="flex items-center gap-3 py-4">
            <Spin />
            <span className="text-[12px] text-black/45">加载样本清单…</span>
          </div>
        </Card>
      ) : (
        <Card className="shadow-sm" styles={{ body: { overflow: 'auto' } }}>
          {/* border-separate:sticky 列配 collapse 边框会在滚动时错位 */}
          <table className="border-separate border-spacing-0">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 min-w-72 border border-black/10 bg-white p-2 text-[12px]">
                  样本(composite 输入)
                </th>
                {config.columns.map((col) => (
                  <th key={col.key} className="min-w-72 border border-black/10 p-2 text-[13px]">
                    {col.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageSamples.map((s) => {
                const meta = metas[sampleKey(s)]
                return (
                  <tr key={sampleKey(s)}>
                    <td className="sticky left-0 z-10 border border-black/10 bg-white p-2 align-top">
                      <a href={fileUrl(s, 'composite.png')} target="_blank" rel="noreferrer">
                        <img
                          src={fileUrl(s, 'composite.png')}
                          alt={s.stem}
                          loading="lazy"
                          className="h-auto w-72"
                        />
                      </a>
                      <div className="mt-1 max-w-72 break-all text-[11px] text-black/55">
                        {s.split === 'val' ? <Tag color="gold">val</Tag> : null}
                        {s.stem}
                      </div>
                      {meta ? (
                        <div className="text-[11px] text-black/40">
                          {config.sampleInfo(meta)}
                        </div>
                      ) : null}
                    </td>
                    {config.columns.map((col) => {
                      if (!meta) {
                        return (
                          <td key={col.key} className="border border-black/10 p-2 align-top">
                            <Spin size="small" />
                          </td>
                        )
                      }
                      const file = col.file(meta)
                      const note = file ? col.note?.(meta) : null
                      return (
                        <td key={col.key} className="border border-black/10 p-2 align-top">
                          {file ? (
                            <>
                              <a href={fileUrl(s, file)} target="_blank" rel="noreferrer">
                                <div style={CHECKERBOARD}>
                                  <img
                                    src={fileUrl(s, file)}
                                    alt={`${s.stem} ${col.key}`}
                                    loading="lazy"
                                    className="h-auto w-72"
                                  />
                                </div>
                              </a>
                              {note ? (
                                <div className="mt-1 text-[11px] text-black/40">{note}</div>
                              ) : null}
                            </>
                          ) : (
                            <span className="text-[11px] text-black/30">空</span>
                          )}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
          <div className="mt-3">{pager}</div>
        </Card>
      )}
    </main>
  )
}
