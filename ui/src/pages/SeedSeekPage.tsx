import { useMemo, useRef, useState, type CSSProperties } from 'react'
import {
  Button,
  Card,
  Checkbox,
  Input,
  InputNumber,
  Radio,
  Select,
  Spin,
  Upload,
  message,
} from 'antd'
import { UploadOutlined } from '@ant-design/icons'
import { submitTask, uploadImage, waitTask, type TaskType } from '../lib/runpodApi'
import stepDefaults from '../config/stepDefaults'

const { TextArea } = Input

const CHECKERBOARD: CSSProperties = {
  backgroundImage:
    'conic-gradient(#e5e5e5 0 25%, #ffffff 0 50%, #e5e5e5 0 75%, #ffffff 0)',
  backgroundSize: '16px 16px',
}

const FILL_PARAMS = {
  prompt: stepDefaults.iconBack.prompt,
  steps: stepDefaults.iconBack.steps,
  guidance: stepDefaults.iconBack.guidance,
  grow_mask: stepDefaults.iconBack.growMask,
  mask_blur: stepDefaults.iconBack.maskBlur,
  max_pixels: stepDefaults.iconBack.maxPixels,
  fill_holes: stepDefaults.iconBack.fillHoles,
}

interface SeekMode {
  label: string
  taskType: TaskType
  /** 管线固定参数(image/mask/output 等),由模式决定,不进参数 UI */
  fixed: Record<string, unknown>
  /** 可调参数模板(seed 由列注入) */
  params: Record<string, unknown>
  /** 结果取哪个文件 */
  output: string
  /** 需要复用主流程 run(依赖 run 目录里已有的素材文件,不能用新上传的图) */
  needsRun: boolean
}

const MODES: Record<string, SeekMode> = {
  text_back: {
    label: '去文字',
    taskType: 'text_back',
    fixed: {},
    output: 'text_back.png',
    needsRun: false,
    params: {
      prompt: stepDefaults.textBack.prompt,
      steps: stepDefaults.textBack.steps,
      protect: true,
      protect_grow: 8,
      max_pixels: 1048576,
    },
  },
  fill_icon: {
    label: '去icon补洞(需主流程 run)',
    taskType: 'flux_fill',
    fixed: { image: 'text_back.png', mask_from: 'icons.png', output: 'seedseek_fill.png' },
    output: 'seedseek_fill.png',
    needsRun: true,
    params: { ...FILL_PARAMS },
  },
  fill_mid: {
    label: '修补中景层(需主流程 run)',
    taskType: 'flux_fill',
    fixed: {
      image: 'icon_back.png',
      mask_from_holes: 'mid_hole.png',
      output: 'seedseek_fill.png',
    },
    output: 'seedseek_fill.png',
    needsRun: true,
    params: { ...FILL_PARAMS },
  },
}

/** 从 /api/runs/<run_id>/files/<file> 形态的地址里识别主流程 run */
function parseRunUrl(url: string): { runId: string; file: string } | null {
  const m = url.match(/\/api\/runs\/([^/]+)\/files\/([^/?#]+)/)
  return m ? { runId: m[1], file: m[2] } : null
}

/** 解析 seed 输入:支持 "1-8"、"3,7,42"、混合 "1-3,9,20-22" */
function parseSeeds(text: string): number[] {
  const out: number[] = []
  for (const part of text.split(',')) {
    const seg = part.trim()
    if (!seg) continue
    const range = seg.match(/^(\d+)\s*-\s*(\d+)$/)
    if (range) {
      const a = parseInt(range[1], 10)
      const b = parseInt(range[2], 10)
      for (let i = Math.min(a, b); i <= Math.max(a, b); i += 1) out.push(i)
    } else if (/^\d+$/.test(seg)) {
      out.push(parseInt(seg, 10))
    }
  }
  return [...new Set(out)]
}

interface ImageRow {
  name: string
  runId: string
  previewUrl: string
  /** true=复用主流程 run(素材齐全);false=新上传(仅去文字模式可用) */
  fromRun: boolean
}

interface CellState {
  url?: string
  error?: string
  skipped?: boolean
}

export default function SeedSeekPage() {
  // 从主流程跳转时通过 query 预填:?task=text_back&img=<encoded url>
  const initial = useMemo(() => {
    const sp = new URLSearchParams(window.location.search)
    const raw = sp.get('task') ?? ''
    const t = raw === 'flux_fill' ? 'fill_icon' : raw
    return {
      task: t in MODES ? t : 'text_back',
      img: sp.get('img') ?? '',
    }
  }, [])

  const [rows, setRows] = useState<ImageRow[]>([])
  const [urlText, setUrlText] = useState(initial.img)
  const [loadingImages, setLoadingImages] = useState(false)

  const [task, setTask] = useState<string>(initial.task)
  const [params, setParams] = useState<Record<string, unknown>>(
    (MODES[initial.task] ?? MODES.text_back).params,
  )
  const [seedText, setSeedText] = useState('1-6')

  const [cells, setCells] = useState<Record<string, CellState>>({})
  // 三态评级:good=还行(计分) / neutral=再看看(默认,不计) / bad=不行(该 seed 整体淘汰)
  const [ratings, setRatings] = useState<Record<string, 'good' | 'neutral' | 'bad'>>({})
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const stopRef = useRef(false)
  // 实时淘汰名单(评分 onChange 直接写,生成循环逐格检查)与当前在等的任务
  const bannedRef = useRef<Set<number>>(new Set())
  const currentRef = useRef<{ seed: number; abort: AbortController } | null>(null)
  // 本轮会话目录名:所有结果集中存到 pod 的 /workspace/seedseek/<session>/
  const sessionRef = useRef(
    new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14),
  )

  const seeds = useMemo(() => parseSeeds(seedText), [seedText])

  const setRating = (key: string, seed: number, value: 'good' | 'neutral' | 'bad') => {
    setRatings((prev) => ({ ...prev, [key]: value }))
    if (value === 'bad') {
      bannedRef.current.add(seed)
      // 这个 seed 正在生成:立即中断等待(GPU 上当次任务会自行跑完,结果丢弃)
      if (currentRef.current?.seed === seed) currentRef.current.abort.abort()
    } else {
      // 反悔:同一 seed 只可能有一行标过"不行"(其余行已禁用),直接解除
      bannedRef.current.delete(seed)
    }
  }

  const switchTask = (t: string) => {
    setTask(t)
    setParams(MODES[t]?.params ?? {})
  }

  const addFiles = async (files: File[]) => {
    setLoadingImages(true)
    try {
      for (const file of files) {
        const info = await uploadImage(file)
        setRows((prev) => [
          ...prev,
          {
            name: file.name,
            runId: info.run_id,
            previewUrl: URL.createObjectURL(file),
            fromRun: false,
          },
        ])
      }
    } catch (error) {
      void message.error(error instanceof Error ? error.message : '上传失败')
    } finally {
      setLoadingImages(false)
    }
  }

  const addUrls = async () => {
    const urls = urlText.split('\n').map((s) => s.trim()).filter(Boolean)
    if (!urls.length) return
    setLoadingImages(true)
    try {
      for (const url of urls) {
        // 主流程 run 的文件地址:直接复用该 run(补洞素材齐全),不建新 run
        const runRef = parseRunUrl(url)
        if (runRef) {
          setRows((prev) => [
            ...prev,
            {
              name: `${runRef.runId.slice(-6)}/${runRef.file}`,
              runId: runRef.runId,
              previewUrl: url,
              fromRun: true,
            },
          ])
          continue
        }
        const resp = await fetch(url)
        if (!resp.ok) throw new Error(`加载失败 HTTP ${resp.status}: ${url}`)
        const blob = await resp.blob()
        const name = url.split('/').pop()?.split('?')[0] || 'image.png'
        const file = new File([blob], name, { type: blob.type || 'image/png' })
        const info = await uploadImage(file)
        setRows((prev) => [
          ...prev,
          { name, runId: info.run_id, previewUrl: URL.createObjectURL(file), fromRun: false },
        ])
      }
      setUrlText('')
    } catch (error) {
      void message.error(error instanceof Error ? error.message : 'URL 加载失败')
    } finally {
      setLoadingImages(false)
    }
  }

  const runAll = async () => {
    if (!rows.length || !seeds.length || running) return
    stopRef.current = false
    setRunning(true)
    setCells({})
    setRatings({})
    bannedRef.current = new Set()
    const total = rows.length * seeds.length
    let done = 0
    try {
      for (const row of rows) {
        for (const seed of seeds) {
          if (stopRef.current) return
          const key = `${row.runId}:${seed}`
          // 该 seed 已被标"不行":余下生成全部跳过
          if (bannedRef.current.has(seed)) {
            setCells((prev) => ({ ...prev, [key]: { skipped: true } }))
            done += 1
            continue
          }
          const m = MODES[task]
          if (m.needsRun && !row.fromRun) {
            setCells((prev) => ({
              ...prev,
              [key]: { error: '该模式需要主流程 run 的文件地址(新上传的图没有补洞素材)' },
            }))
            done += 1
            continue
          }
          setProgress(`${done}/${total} · ${row.name} × seed ${seed}`)
          const abort = new AbortController()
          currentRef.current = { seed, abort }
          try {
            const { task_id } = await submitTask(m.taskType, row.runId, {
              ...m.fixed,
              ...params,
              seed,
            })
            await waitTask(task_id, { intervalMs: 1500, signal: abort.signal })
            // 输出文件名固定,必须在下一个 seed 覆盖前立刻取走快照
            const resp = await fetch(
              `/api/runs/${row.runId}/files/${m.output}?t=${Date.now()}`,
            )
            if (!resp.ok) throw new Error(`取结果失败 HTTP ${resp.status}`)
            const blob = await resp.blob()
            const url = URL.createObjectURL(blob)
            setCells((prev) => ({ ...prev, [key]: { url } }))
            // 集中落盘:/workspace/seedseek/<session>/<图名>_seed<N>.png(失败不影响流程)
            const stem = row.name.replace(/\.[^.]+$/, '').replace(/[^\w一-鿿-]+/g, '_')
            void fetch(
              `/api/seedseek/${sessionRef.current}_${task}/${stem}_seed${seed}.png`,
              { method: 'POST', body: blob },
            ).catch(() => {})
          } catch (error) {
            if (bannedRef.current.has(seed)) {
              // 等待途中被标"不行"而中断:按跳过处理,不算错误
              setCells((prev) => ({ ...prev, [key]: { skipped: true } }))
            } else {
              setCells((prev) => ({
                ...prev,
                [key]: { error: error instanceof Error ? error.message : '失败' },
              }))
            }
          } finally {
            currentRef.current = null
          }
          done += 1
        }
      }
      setProgress(`完成 ${done}/${total}`)
    } finally {
      setRunning(false)
    }
  }

  // 计分:统计"还行"票;任何一行标了"不行"的 seed 整体淘汰,不进入排名
  const tally = useMemo(() => {
    const count: Record<number, number> = {}
    const banned = new Set<number>()
    for (const seed of seeds) count[seed] = 0
    for (const row of rows) {
      for (const seed of seeds) {
        const r = ratings[`${row.runId}:${seed}`]
        if (r === 'good') count[seed] += 1
        if (r === 'bad') banned.add(seed)
      }
    }
    const eligible = seeds.filter((s) => !banned.has(s))
    const max = Math.max(0, ...eligible.map((s) => count[s]))
    const best = eligible.filter((s) => count[s] === max && max > 0)
    return { count, banned, max, best }
  }, [ratings, rows, seeds])

  return (
    <main className="min-h-screen bg-[#f5f6f8] p-4">
      <Card
        title={<span className="text-[16px] font-bold">找 seed 工具</span>}
        extra={
          <a href="/" className="text-[13px]">
            返回主流程
          </a>
        }
        className="mb-4 shadow-sm"
      >
        <div className="flex flex-col gap-4">
          {/* 1. 图片来源 */}
          <div className="flex items-start gap-4">
            <div>
              <div className="mb-1 text-[13px] font-bold">上传图片(可多张)</div>
              <Upload
                multiple
                showUploadList={false}
                beforeUpload={(_file, fileList) => {
                  // antd 对多选会逐个回调,只在首个文件时批量处理一次
                  if (_file === fileList[0]) void addFiles(fileList as File[])
                  return false
                }}
              >
                <Button icon={<UploadOutlined />} loading={loadingImages}>
                  选择图片
                </Button>
              </Upload>
            </div>
            <div className="flex-1">
              <div className="mb-1 text-[13px] font-bold">
                或输入图片地址(每行一个,支持 /api/runs/.../files/xxx.png)
              </div>
              <div className="flex gap-2">
                <TextArea
                  rows={2}
                  value={urlText}
                  onChange={(e) => setUrlText(e.target.value)}
                  placeholder="https://... 或 /api/runs/<run_id>/files/origin.png"
                />
                <Button onClick={() => void addUrls()} loading={loadingImages}>
                  加载
                </Button>
              </div>
            </div>
          </div>

          {/* 2. 任务与动态参数 */}
          <div>
            <div className="mb-1 flex items-center gap-3">
              <span className="text-[13px] font-bold">任务类型</span>
              <Select
                className="w-44"
                value={task}
                onChange={switchTask}
                disabled={running}
                options={Object.entries(MODES).map(([value, m]) => ({
                  value,
                  label: m.label,
                }))}
              />
              <span className="text-[11px] text-black/45">
                参数自动加载当前配置,可改;seed 由列注入;补洞类模式的素材(源图/mask)由模式自动指定
              </span>
            </div>
            <div className="grid grid-cols-3 gap-3">
              {Object.entries(params).map(([key, value]) => (
                <div key={key} className={typeof value === 'string' && value.length > 60 ? 'col-span-3' : ''}>
                  <div className="mb-0.5 text-[11px] text-black/50">{key}</div>
                  {typeof value === 'boolean' ? (
                    <Checkbox
                      checked={value}
                      disabled={running}
                      onChange={(e) =>
                        setParams((p) => ({ ...p, [key]: e.target.checked }))
                      }
                    />
                  ) : typeof value === 'number' ? (
                    <InputNumber
                      className="w-full"
                      value={value}
                      disabled={running}
                      onChange={(v) => setParams((p) => ({ ...p, [key]: v ?? 0 }))}
                    />
                  ) : String(value).length > 60 ? (
                    <TextArea
                      rows={3}
                      value={String(value)}
                      disabled={running}
                      onChange={(e) =>
                        setParams((p) => ({ ...p, [key]: e.target.value }))
                      }
                    />
                  ) : (
                    <Input
                      value={String(value)}
                      disabled={running}
                      onChange={(e) =>
                        setParams((p) => ({ ...p, [key]: e.target.value }))
                      }
                    />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* 3. seed 输入与执行 */}
          <div className="flex items-end gap-3">
            <div className="flex-1">
              <div className="mb-1 text-[13px] font-bold">
                seeds(范围 1-8 或逗号列表 3,7,42,可混用)
              </div>
              <Input
                value={seedText}
                onChange={(e) => setSeedText(e.target.value)}
                disabled={running}
              />
            </div>
            <span className="pb-1 text-[12px] text-black/45">
              共 {seeds.length} 个 seed × {rows.length} 张图 ={' '}
              {seeds.length * rows.length} 次生成
            </span>
            {running ? (
              <Button danger onClick={() => (stopRef.current = true)}>
                停止
              </Button>
            ) : null}
            <Button
              type="primary"
              loading={running}
              disabled={!rows.length || !seeds.length}
              onClick={() => void runAll()}
            >
              批量生成
            </Button>
          </div>
          {progress ? (
            <div className="text-[12px] text-black/55">
              {progress}
              <span className="ml-3 text-black/40">
                结果同步保存至 pod:/workspace/seedseek/{sessionRef.current}_{task}/
              </span>
            </div>
          ) : null}
        </div>
      </Card>

      {/* 4. 结果表格 */}
      {rows.length > 0 && seeds.length > 0 ? (
        <Card className="shadow-sm" styles={{ body: { overflow: 'auto' } }}>
          {/* border-separate:sticky 列配 collapse 边框会在滚动时错位 */}
          <table className="border-separate border-spacing-0">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 min-w-80 border border-black/10 bg-white p-2 text-[12px]">
                  图片
                </th>
                {seeds.map((seed) => (
                  <th
                    key={seed}
                    className="min-w-80 border border-black/10 p-2 text-[13px]"
                  >
                    seed {seed}
                    {tally.banned.has(seed) ? (
                      <span className="ml-1 text-[11px] font-normal text-[#cf1322]">
                        已淘汰
                      </span>
                    ) : (
                      <span className="ml-1 text-[11px] font-normal text-black/45">
                        ({tally.count[seed] ?? 0} 票)
                      </span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.runId}>
                  <td className="sticky left-0 z-10 border border-black/10 bg-white p-2 align-top">
                    <img
                      src={row.previewUrl}
                      alt={row.name}
                      className="h-auto w-80"
                    />
                    <div className="mt-1 max-w-40 break-all text-[11px] text-black/55">
                      {row.name}
                    </div>
                  </td>
                  {seeds.map((seed) => {
                    const key = `${row.runId}:${seed}`
                    const cell = cells[key]
                    return (
                      <td key={seed} className="border border-black/10 p-2 align-top">
                        {cell?.url ? (
                          <>
                            <a href={cell.url} target="_blank" rel="noreferrer">
                              <div style={CHECKERBOARD}>
                                <img
                                  src={cell.url}
                                  alt={`seed ${seed}`}
                                  className="h-auto w-80"
                                />
                              </div>
                            </a>
                            <div className="mt-1">
                              <Radio.Group
                                size="small"
                                optionType="button"
                                buttonStyle="solid"
                                value={ratings[key] ?? 'neutral'}
                                disabled={
                                  tally.banned.has(seed) &&
                                  (ratings[key] ?? 'neutral') !== 'bad'
                                }
                                onChange={(e) => setRating(key, seed, e.target.value)}
                                options={[
                                  { value: 'good', label: '还行' },
                                  { value: 'neutral', label: '再看看' },
                                  { value: 'bad', label: '不行' },
                                ]}
                              />
                            </div>
                          </>
                        ) : cell?.skipped ? (
                          <span className="text-[11px] text-black/30">
                            已淘汰,跳过生成
                          </span>
                        ) : cell?.error ? (
                          <div className="max-w-40 break-all text-[11px] text-[#cf1322]">
                            {cell.error}
                          </div>
                        ) : running ? (
                          <Spin size="small" />
                        ) : (
                          <span className="text-[11px] text-black/30">待生成</span>
                        )}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>

          <div className="mt-3 text-[14px]">
            {tally.max > 0 ? (
              <>
                本轮最佳 seed:
                <span className="mx-1 font-bold text-[#1677ff]">
                  {tally.best.join('、')}
                </span>
                (各 {tally.max} 个"还行"
                {tally.banned.size > 0
                  ? `;${[...tally.banned].join('、')} 因被标"不行"淘汰`
                  : ''}
                )
              </>
            ) : (
              <span className="text-black/45">
                标"还行"计票;任何一行标了"不行"的 seed 整体淘汰
              </span>
            )}
          </div>
        </Card>
      ) : null}
    </main>
  )
}
