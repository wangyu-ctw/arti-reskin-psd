/** 新管线(/pipeline2)前端 store:纯遥控器——每步提交一个 p2_* 任务
 * (编排与像素操作全在服务端 Python,见 service/app/pipeline2.py),
 * 完成后读 run 目录的 p2_state.json 渲染。与旧管线 store/config 完全隔离
 * (只读旧 localStorage 的 OpenRouter apiKey,不写)。 */
import { create } from 'zustand'
import { submitTask, uploadImage, waitTask, type TaskType } from '../lib/runpodApi'
import defaults from '../config/pipeline2Defaults.json'
import inventoryPromptRaw from '../config/pipeline2InventoryPrompt.md?raw'

const OLD_SETTINGS_KEY = 'gpt56-sol-image-analyzer.settings.v1'

type StepStatus = 'idle' | 'running' | 'done' | 'error'

export interface P2Layer {
  name: string
  file: string
  coverage: number
  keep: boolean
}

export interface P2Element {
  id: string
  sourceLayer: string
  sourceFile: string
  bbox: number[]
  cls: string
  conf: number
  type?: string
  cover?: string
  verdict?: string
  mergedInto?: string
  skip?: boolean
  extract?: { method: 'crop' | 'sam2_combined'; file: string; bbox: number[] }
}

interface P2ServerState {
  imageSize?: { w: number; h: number }
  slotLayers?: P2Layer[]
  panelzLayers?: P2Layer[]
  elements?: P2Element[]
  missing?: { layer: string; type: string; bbox: number[]; note: string; file?: string }[]
  gptSummary?: string
  extractStats?: string
  yoloDropped?: number
  yoloRescued?: number
  originInventory?: P2InventoryItem[]
  originYolo?: P2InventoryItem[]
  layerYolo?: (P2InventoryItem & { sourceLayer: string })[]
  inventoryStats?: P2InventoryStats | null
  assetsSummary?: { count: number; byLayer: Record<string, number>; file: string }
  cascadeSummary?: P2CascadeSummary
  psdSummary?: P2PsdSummary | null
}

export interface P2CascadeStep {
  layer: string
  kind: string
  cut: number
  rescued: string[]
  /** temp 下沉时以未解决 item 身份认领的素材(串槽内容归位) */
  claimed?: string[]
  overlays: number
  absorbed: number
  assets: string[]
  tempFile: string
  tempCoverage: number
}

export interface P2CascadeAsset {
  id: string
  file: string
  cls: string
  bbox: number[]
  sourceLayer: string
  zIndex: number
  method: string
  stackedOn?: string
  recoveredFrom?: string
}

export interface P2CascadeManifest {
  /** 级联落盘时间戳:级联重跑后文件名不变,图片 URL 用它换版防缓存混读 */
  generatedAt?: number
  /** 生成画布帧尺寸(素材像素/bbox 的原生帧;原图经 1024 桶+16 对齐) */
  canvas?: { w: number; h: number }
  steps: P2CascadeStep[]
  assets: P2CascadeAsset[]
  debris: string
  background: { file: string; zIndex: number }
  report: {
    lost: { id: string; cls: string; bbox: number[] }[]
    ghostSuspects: { id: string; bgDiff: number }[]
    panelSoftCheck: { cls: string; bbox: number[] }[]
    recoveredBg: number
    recoveredOrigin: number
    stackRelations: number
  }
}

export interface P2BatchItem {
  name: string
  runId: string
  /** 当前工序文案(完成后为「完成」) */
  step: string
  status: 'running' | 'done' | 'error'
  error?: string
  assets?: number
  psdLayers?: number
  diffMean?: number
}

export interface P2PsdSummary {
  file: string
  preview: string
  layers: number
  groups: number
  /** 全透明素材跳过数(正常应为 0) */
  skipped?: number
  sizeMB: number
  /** 与原图的 RGB 平均绝对差(0~255) */
  diffMean: number
  /** 差异像素占比(任一通道差 >12 的像素百分数) */
  diffPct: number
  generatedAt: number
  elapsed: number
}

export interface P2CascadeSummary {
  count: number
  byLayer: Record<string, number>
  lost: number
  ghosts: number
  relations: number
  file: string
  debris: string
  background: string
}

export interface P2SixSlotParams {
  steps: number
  seed: number
  trueCfg: number
  resolution: number
}

export interface P2PanelzParams extends P2SixSlotParams {
  layers: number
}

export interface P2YoloParams {
  model: string
  imgsz: number
  conf: number
  iou: number
  dedupIou: number
}

export interface P2GptParams {
  model: string
  prompt: string
  /** 推理强度(none~max)与速度模式(balanced/latency/throughput),同旧管线第三步 */
  effort?: string
  speed?: string
}

export interface P2InventoryItem {
  cls: string
  bbox: number[]
  conf: number
  status?: string
}

export interface P2InventoryStats {
  union: number
  afterDedup: number
  vlDup: number
  vlDiscard: number
  vlMissing: number
  final: number
}

interface P2State {
  apiKey: string
  runpodTarget: string
  sixSlotParams: P2SixSlotParams
  panelzParams: P2PanelzParams
  yoloParams: P2YoloParams
  gptParams: P2GptParams
  detectParams: P2GptParams
  runId: string
  originUrl: string
  imageSize: { w: number; h: number } | null
  status: Record<string, StepStatus>
  errors: Record<string, string>
  slotLayers: P2Layer[]
  panelzLayers: P2Layer[]
  elements: P2Element[]
  missing: { layer: string; type: string; bbox: number[]; note: string; file?: string }[]
  gptSummary: string
  extractStats: string
  yoloDropped: number
  yoloRescued: number
  originInventory: P2InventoryItem[]
  originYolo: P2InventoryItem[]
  layerYolo: (P2InventoryItem & { sourceLayer: string })[]
  inventoryStats: P2InventoryStats | null
  assetsSummary: { count: number; byLayer: Record<string, number>; file: string } | null
  cascadeSummary: P2CascadeSummary | null
  cascadeManifest: P2CascadeManifest | null
  psdSummary: P2PsdSummary | null
  /** 批量执行:独立 run,不影响页面当前 run 的状态 */
  batch: { running: boolean; items: P2BatchItem[] }
  /** 第 6 步:前端 canvas 绘制开关(不经后端、不落盘) */
  recomposeReady: boolean
  /** 第 5 步素材 hover 联动:拼回图只显示该素材 */
  hoveredElementId: string | null
  tick: number

  setApiKey: (k: string) => void
  patchSixSlot: (p: Partial<P2SixSlotParams>) => void
  patchPanelz: (p: Partial<P2PanelzParams>) => void
  patchYolo: (p: Partial<P2YoloParams>) => void
  patchGpt: (p: Partial<P2GptParams>) => void
  patchDetect: (p: Partial<P2GptParams>) => void
  clearRun: () => void
  setRunpodTarget: (t: string) => void
  fetchRunpodTarget: () => Promise<void>
  applyRunpodTarget: () => Promise<void>
  uploadOriginal: (file: File) => Promise<void>
  restoreRun: (runId: string) => Promise<void>
  runPsd: () => Promise<void>
  runBatch: (files: File[]) => Promise<void>
  runDetect: () => Promise<void>
  runInventory: () => Promise<void>
  runLayerYolo: (scope: 'six' | 'panelz') => Promise<void>
  runSixSlot: () => Promise<void>
  runPanelz: () => Promise<void>
  runYolo: () => Promise<void>
  runGpt: () => Promise<void>
  runExtract: () => Promise<void>
  runAssets: () => Promise<void>
  runCascade: () => Promise<void>
  runRecompose: () => void
  setHoveredElement: (id: string | null) => void
}

interface SharedSettings {
  apiKey?: string
  runpodTarget?: string
}

/** 与老管线共用同一个 localStorage 设置字段(apiKey 只读,runpodTarget 读写) */
function loadSharedSettings(): SharedSettings {
  try {
    const raw = localStorage.getItem(OLD_SETTINGS_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as SharedSettings
    return {
      apiKey: typeof parsed.apiKey === 'string' ? parsed.apiKey : undefined,
      runpodTarget:
        typeof parsed.runpodTarget === 'string' ? parsed.runpodTarget : undefined,
    }
  } catch {
    return {}
  }
}

function saveSharedSettings(patch: SharedSettings): void {
  try {
    let cur: Record<string, unknown> = {}
    const raw = localStorage.getItem(OLD_SETTINGS_KEY)
    if (raw) cur = JSON.parse(raw) as Record<string, unknown>
    localStorage.setItem(OLD_SETTINGS_KEY, JSON.stringify({ ...cur, ...patch }))
  } catch {
    /* 存储失败不阻塞 */
  }
}

const sharedSaved = loadSharedSettings()

// 启动对齐(同老管线):localStorage 有地址就推给 dev server 中间件;
// 没有就取中间件当前值回填 store 与 localStorage
if (sharedSaved.runpodTarget) {
  fetch('/__runpod-target', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target: sharedSaved.runpodTarget }),
  }).catch(() => {})
} else {
  fetch('/__runpod-target')
    .then((r) => r.json())
    .then((d: { target?: string }) => {
      if (d.target) {
        usePipeline2Store.setState({ runpodTarget: d.target })
        saveSharedSettings({ runpodTarget: d.target })
      }
    })
    .catch(() => {})
}

const fileUrl = (runId: string, f: string) => `/api/runs/${runId}/files/${f}`

export const usePipeline2Store = create<P2State>((set, get) => {
  const setStep = (key: string, s: StepStatus, err = '') =>
    set((st) => ({
      status: { ...st.status, [key]: s },
      errors: { ...st.errors, [key]: err },
      tick: st.tick + 1,
    }))

  /** 从 run 目录读服务端状态并入 store */
  const pullState = async () => {
    const { runId } = get()
    const resp = await fetch(`${fileUrl(runId, 'p2_state.json')}?t=${Date.now()}`)
    if (!resp.ok) return
    const s = (await resp.json()) as P2ServerState
    set({
      imageSize: s.imageSize ?? get().imageSize,
      slotLayers: s.slotLayers ?? [],
      panelzLayers: s.panelzLayers ?? [],
      elements: s.elements ?? [],
      missing: s.missing ?? [],
      gptSummary: s.gptSummary ?? '',
      extractStats: s.extractStats ?? '',
      yoloDropped: s.yoloDropped ?? 0,
      yoloRescued: s.yoloRescued ?? 0,
      originInventory: s.originInventory ?? [],
      originYolo: s.originYolo ?? [],
      layerYolo: s.layerYolo ?? [],
      inventoryStats: s.inventoryStats ?? null,
      assetsSummary: s.assetsSummary ?? null,
      cascadeSummary: s.cascadeSummary ?? null,
      psdSummary: s.psdSummary ?? null,
    })
  }

  const pullCascadeManifest = async () => {
    const { runId, cascadeSummary } = get()
    if (!runId || !cascadeSummary) {
      set({ cascadeManifest: null })
      return
    }
    try {
      const resp = await fetch(
        `${fileUrl(runId, cascadeSummary.file)}?t=${Date.now()}`)
      if (resp.ok) set({ cascadeManifest: (await resp.json()) as P2CascadeManifest })
    } catch {
      /* 账本读取失败不阻塞 */
    }
  }

  /** 一步 = 提交任务 → 等完成 → 拉状态 */
  const runStep = async (
    key: string, type: TaskType, params: Record<string, unknown>,
    after?: (result: unknown) => void,
  ) => {
    const { runId } = get()
    if (!runId || get().status[key] === 'running') return
    setStep(key, 'running')
    try {
      const { task_id } = await submitTask(type, runId, params)
      const result = await waitTask(task_id, { intervalMs: 3000 })
      await pullState()
      after?.(result)
      setStep(key, 'done')
    } catch (e) {
      setStep(key, 'error', e instanceof Error ? e.message : String(e))
    }
  }

  return {
    apiKey: sharedSaved.apiKey ?? '',
    runpodTarget: sharedSaved.runpodTarget ?? '',
    sixSlotParams: { ...defaults.sixSlot },
    panelzParams: { ...defaults.panelz },
    yoloParams: {
      model: defaults.yolo.model, imgsz: defaults.yolo.imgsz,
      conf: defaults.yolo.conf, iou: defaults.yolo.iou,
      dedupIou: defaults.yolo.dedupIou,
    },
    gptParams: { model: defaults.gpt.model, prompt: defaults.gpt.prompt },
    detectParams: {
      model: defaults.gpt.model,
      prompt: inventoryPromptRaw.trim(),
      effort: defaults.gpt.effort,
      speed: defaults.gpt.speed,
    },
    runId: '',
    originUrl: '',
    imageSize: null,
    status: {},
    errors: {},
    slotLayers: [],
    panelzLayers: [],
    elements: [],
    missing: [],
    gptSummary: '',
    extractStats: '',
    yoloDropped: 0,
    yoloRescued: 0,
    originInventory: [],
    originYolo: [],
    layerYolo: [],
    inventoryStats: null,
    assetsSummary: null,
    cascadeSummary: null,
    cascadeManifest: null,
    psdSummary: null,
    batch: { running: false, items: [] },
    recomposeReady: false,
    hoveredElementId: null,
    tick: 0,

    setApiKey: (k) => set({ apiKey: k }),

    patchSixSlot: (p) => set((st) => ({ sixSlotParams: { ...st.sixSlotParams, ...p } })),
    patchPanelz: (p) => set((st) => ({ panelzParams: { ...st.panelzParams, ...p } })),
    patchYolo: (p) => set((st) => ({ yoloParams: { ...st.yoloParams, ...p } })),
    patchGpt: (p) => set((st) => ({ gptParams: { ...st.gptParams, ...p } })),
    patchDetect: (p) => set((st) => ({ detectParams: { ...st.detectParams, ...p } })),

    clearRun: () => set({
      runId: '', originUrl: '', imageSize: null, status: {}, errors: {},
      slotLayers: [], panelzLayers: [], elements: [], missing: [],
      gptSummary: '', extractStats: '', yoloDropped: 0,
      originInventory: [], originYolo: [], layerYolo: [], inventoryStats: null,
      assetsSummary: null, cascadeSummary: null, cascadeManifest: null,
      psdSummary: null,
      recomposeReady: false, hoveredElementId: null,
    }),

    setRunpodTarget: (t) => set({ runpodTarget: t }),

    // 与老管线同逻辑:localStorage(共用字段)为权威,应用时双写
    fetchRunpodTarget: async () => {
      if (get().runpodTarget) return
      try {
        const resp = await fetch('/__runpod-target')
        const data = (await resp.json()) as { target?: string }
        if (data.target) set({ runpodTarget: data.target })
      } catch {
        /* dev server 不可用时留空 */
      }
    },

    applyRunpodTarget: async () => {
      const t = get().runpodTarget.trim().replace(/\/+$/, '')
      if (!t) return
      const resp = await fetch('/__runpod-target', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target: t }),
      })
      const data = (await resp.json()) as { target?: string; error?: string }
      if (!resp.ok || !data.target) throw new Error(data.error ?? '设置失败')
      set({ runpodTarget: data.target })
      saveSharedSettings({ runpodTarget: data.target })
    },

    uploadOriginal: async (file) => {
      if (get().status.upload === 'running') return
      setStep('upload', 'running')
      try {
        const info = await uploadImage(file)
        set({
          runId: info.run_id,
          originUrl: fileUrl(info.run_id, 'origin.png'),
          imageSize: null, slotLayers: [], panelzLayers: [], elements: [],
          missing: [], gptSummary: '', extractStats: '',
          recomposeReady: false, hoveredElementId: null,
        })
        setStep('upload', 'done')
      } catch (e) {
        setStep('upload', 'error', e instanceof Error ? e.message : String(e))
      }
    },

    restoreRun: async (runId) => {
      if (!runId) return
      set({
        runId, originUrl: fileUrl(runId, 'origin.png'),
        imageSize: null, slotLayers: [], panelzLayers: [], elements: [],
        missing: [], gptSummary: '', extractStats: '',
          recomposeReady: false, hoveredElementId: null,
      })
      await pullState()
      setStep('upload', 'done')
      // 按恢复到的数据反推各步完成状态,解锁后续按钮
      const st = get()
      if (st.originYolo.length) setStep('detect', 'done')
      if (st.inventoryStats) setStep('inventory', 'done')
      if (st.slotLayers.length) setStep('sixSlot', 'done')
      if (st.panelzLayers.length) setStep('panelz', 'done')
      if (st.elements.length) setStep('yolo', 'done')
      if (st.gptSummary) setStep('gpt', 'done')
      if (st.extractStats || st.elements.some((e) => e.extract)) {
        setStep('extract', 'done')
      }
      if (st.assetsSummary) setStep('assets', 'done')
      if (st.cascadeSummary) {
        setStep('cascade', 'done')
        await pullCascadeManifest()
      }
      if (st.psdSummary) setStep('psd', 'done')
    },

    runDetect: async () => {
      const y = get().yoloParams
      await runStep('detect', 'p2_detect', {
        yolo_model: y.model, imgsz: y.imgsz, conf: y.conf, iou: y.iou,
      })
    },

    runLayerYolo: async (scope) => {
      const y = get().yoloParams
      await runStep(scope === 'six' ? 'sixYolo' : 'panelzYolo', 'p2_layer_yolo', {
        scope, yolo_model: y.model, imgsz: y.imgsz, conf: y.conf, iou: y.iou,
      })
    },

    runInventory: async () => {
      const { apiKey, detectParams, yoloParams } = get()
      if (!apiKey.trim()) {
        setStep('inventory', 'error', '缺少 OpenRouter API Key')
        return
      }
      await runStep('inventory', 'p2_inventory', {
        api_key: apiKey,
        model: detectParams.model,
        prompt: detectParams.prompt,
        effort: detectParams.effort ?? 'high',
        speed: detectParams.speed ?? 'balanced',
        dedup_iou: yoloParams.dedupIou,
      })
    },

    runSixSlot: async () => {
      const p = get().sixSlotParams
      await runStep('sixSlot', 'p2_sixslot', {
        steps: p.steps, seed: p.seed, true_cfg: p.trueCfg,
        resolution: p.resolution,
      })
      // 六槽完成后自动接 panelz(panel 层为空时后端会报错,吞掉即可)
      if (get().status.sixSlot === 'done') {
        await get().runPanelz()
      }
    },

    runPanelz: async () => {
      const p = get().panelzParams
      await runStep('panelz', 'p2_panelz', {
        layers: p.layers, steps: p.steps, seed: p.seed,
        true_cfg: p.trueCfg, resolution: p.resolution,
      })
    },

    runYolo: async () => {
      const p = get().yoloParams
      await runStep('yolo', 'p2_yolo', {
        model: p.model, imgsz: p.imgsz, conf: p.conf, iou: p.iou,
        dedup_iou: p.dedupIou,
      })
    },

    runGpt: async () => {
      const { apiKey } = get()
      if (!apiKey.trim()) {
        setStep('gpt', 'error', '缺少 OpenRouter API Key(可在旧管线第 2 步配置,或在本页填写)')
        return
      }
      const g = get().gptParams
      await runStep('gpt', 'p2_gpt', {
        api_key: apiKey, model: g.model, prompt: g.prompt,
      })
    },

    runExtract: async () => {
      await runStep('extract', 'p2_extract', {})
    },

    runAssets: async () => {
      await runStep('assets', 'p2_assets', {})
    },

    runCascade: async () => {
      await runStep('cascade', 'p2_cascade', {})
      await pullCascadeManifest()
    },

    runPsd: async () => {
      await runStep('psd', 'p2_psd', {})
    },

    // 批量:每张图独立 run 全链(⓪⁺∥① → ①ᵇ → ② → ③ → ④),逐图串行;
    // 不触碰页面当前 run 的任何状态,key 只随 ② 请求传、不落盘
    runBatch: async (files) => {
      if (get().batch.running || !files.length) return
      const items: P2BatchItem[] = files.map((f) => ({
        name: f.name, runId: '', step: '排队', status: 'running' as const,
      }))
      set({ batch: { running: true, items } })
      const patch = (i: number, p: Partial<P2BatchItem>) =>
        set((st) => ({
          batch: {
            running: st.batch.running,
            items: st.batch.items.map((it, j) => (j === i ? { ...it, ...p } : it)),
          },
        }))
      const wait = (taskId: string) => waitTask(taskId, { intervalMs: 3000 })
      for (let i = 0; i < files.length; i++) {
        try {
          patch(i, { step: '上传' })
          const { run_id } = await uploadImage(files[i])
          patch(i, { runId: run_id })
          const { yoloParams: y, sixSlotParams: sp, panelzParams: pz,
                  detectParams: dp, apiKey } = get()
          patch(i, { step: '⓪⁺检测 ∥ ①六槽' })
          const [tDetect, tSix] = await Promise.all([
            submitTask('p2_detect', run_id, {
              yolo_model: y.model, imgsz: y.imgsz, conf: y.conf, iou: y.iou,
            }),
            submitTask('p2_sixslot', run_id, {
              steps: sp.steps, seed: sp.seed, true_cfg: sp.trueCfg,
              resolution: sp.resolution,
            }),
          ])
          await Promise.all([wait(tDetect.task_id), wait(tSix.task_id)])
          patch(i, { step: '①ᵇ panelz' })
          const tPz = await submitTask('p2_panelz', run_id, {
            layers: pz.layers, steps: pz.steps, seed: pz.seed,
            true_cfg: pz.trueCfg, resolution: pz.resolution,
          })
          await wait(tPz.task_id)
          patch(i, { step: '② 汇总审核' })
          const tInv = await submitTask('p2_inventory', run_id, {
            api_key: apiKey, model: dp.model, prompt: dp.prompt,
            effort: dp.effort ?? 'high', speed: dp.speed ?? 'balanced',
            dedup_iou: y.dedupIou,
          })
          await wait(tInv.task_id)
          patch(i, { step: '③ 级联切取' })
          const tCas = await submitTask('p2_cascade', run_id, {})
          const cas = (await wait(tCas.task_id)) as { count?: number } | null
          patch(i, { step: '④ 生成PSD', assets: cas?.count })
          const tPsd = await submitTask('p2_psd', run_id, {})
          const psd = (await wait(tPsd.task_id)) as
            { layers?: number; diffMean?: number } | null
          patch(i, {
            step: '完成', status: 'done',
            psdLayers: psd?.layers, diffMean: psd?.diffMean,
          })
        } catch (e) {
          patch(i, {
            status: 'error',
            error: e instanceof Error ? e.message : String(e),
          })
        }
      }
      set((st) => ({ batch: { ...st.batch, running: false } }))
    },

    // 拼回改为前端 canvas 实时绘制,这里只打开开关并刷新
    runRecompose: () => {
      set({ recomposeReady: true })
      setStep('recompose', 'done')
    },

    setHoveredElement: (id) => set({ hoveredElementId: id }),
  }
})
