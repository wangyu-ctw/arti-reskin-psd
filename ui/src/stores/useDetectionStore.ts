import { create } from 'zustand'
import {
  API_URL,
  MODEL,
  REASONING_OPTIONS,
  SPEED_OPTIONS,
  STORAGE_KEY,
  buildUserContent,
  createCacheKey,
  DETECTION_SCHEMA,
  extractResponseText,
  detectOverlay,
  humanizeError,
  pickDetections,
  type StructuredResult,
} from '../lib/detection'
import {
  submitTask,
  uploadImage,
  waitTask,
  type CreateRunResult,
} from '../lib/runpodApi'
import stepDefaults from '../config/stepDefaults'
import {
  analyzeIcons,
  fetchImageAsDataUrl,
  type AnalyzedIcon,
} from '../lib/iconAnalysis'
import { analyzeIconGroups, type IconGroup } from '../lib/iconGroups'
import { clampIconPoints } from '../lib/pointClamp'
import { extractTextFront } from '../lib/textFront'
import { analyzeCutoutFixes, fetchImageDataUrl } from '../lib/iconRefine'

export const DEFAULT_TEXT_BACK_PROMPT = stepDefaults.textBack.prompt

export const DEFAULT_ICON_BACK_PROMPT = stepDefaults.iconBack.prompt

export const DEFAULT_MID_FILL_PROMPT = stepDefaults.midFill.prompt

// 检测结果缓存(每次成功生成后写入,面板上可一键导入)
const RESULT_CACHE_KEY = 'gpt56-sol-image-analyzer.result.v1'

interface ResultCache {
  outputText: string
  structuredResult: StructuredResult | null
  usageModel: string
  usageTokens: string
  savedAt: number
}

function saveResultCache(data: Omit<ResultCache, 'savedAt'>): void {
  try {
    localStorage.setItem(
      RESULT_CACHE_KEY,
      JSON.stringify({ ...data, savedAt: Date.now() }),
    )
  } catch {
    /* 缓存失败不影响主流程 */
  }
}

type TextBackStatus = 'idle' | 'running' | 'done' | 'error'

// 第 7 步提icon:按 icon 像素长边分三档,各档独立参数
export type IconTier = 'small' | 'medium' | 'large'

export interface IconTierParams {
  paddingRatio: number
  minPadding: number
  maskThreshold: number
  featherRadius: number
  cropScale: number
  refine: boolean
  multimask: boolean
  fillHoles: boolean
}

// 第 9~11 步:中景层提取(assets/bar/button,从第 8 步结果图 icon_back.png 提取)
export type MidKey = 'assets' | 'bar' | 'button'

export interface MidExtractParams {
  paddingRatio: number
  minPadding: number
  maskThreshold: number
  featherRadius: number
  cropScale: number
  refine: boolean
  multimask: boolean
  fillHoles: boolean
}

const midIdle = () => ({ assets: 'idle', bar: 'idle', button: 'idle' }) as Record<MidKey, TextBackStatus>
const midEmpty = () => ({ assets: '', bar: '', button: '' }) as Record<MidKey, string>

type StatusType = '' | 'working' | 'success' | 'error'

// localStorage 只存 apiKey 和 runpod 地址;
// 各步骤默认值统一走 src/config/stepDefaults(长提示词在独立文本文件里,由该模块拼装)
interface SavedSettings {
  apiKey?: string
  runpodTarget?: string
}

interface DetectionState {
  apiKey: string
  runpodTarget: string
  systemPrompt: string
  userPrompt: string
  yoloResult: string
  // 第 3 步 YOLO 权重选择(daemon 多模型注册表的 key)
  yoloModel: string
  isYoloRunning: boolean
  yoloError: string
  reasoningEffort: string
  speedMode: string
  file: File | null
  previewUrl: string
  // RunPod 上的 run 信息(POST /runs 的返回),后续任务都引用这里的 run_id
  runInfo: CreateRunResult | null
  isUploading: boolean
  uploadError: string
  // 第 5 步:去文字(text_back)
  textBackPrompt: string
  textBackSeed: number
  textBackSteps: number
  // 保护合成:只有 YOLO text 框内取重生成像素,其余保留原图,icon 不可能被误删
  textBackProtect: boolean
  textBackProtectGrow: number
  textBackStatus: TextBackStatus
  textBackImageUrl: string
  textBackError: string
  // 本地上传的去字图:不上传 RunPod,仅供第 3 步双图检测使用,优先于生成结果
  textBackLocalFile: File | null
  textBackLocalUrl: string
  // 历史 run_id(最近在前,最多 10 条,localStorage 持久化)
  runHistory: string[]
  // 第 6 步:分析 icon(OpenRouter 调 Gemini)
  iconAnalysisModel: string
  iconAnalysisTemperature: number
  iconAnalysisSystemPrompt: string
  iconAnalysisUserPrompt: string
  iconAnalysisStatus: TextBackStatus
  iconAnalysisError: string
  analyzedIcons: AnalyzedIcon[] | null
  // 域钳制结果说明(修正了几个越域点)
  iconClampInfo: string
  // 第 8+ 步:素材化(分组 → 每组一张高清透明素材,Qwen 双图重绘)
  iconGroupModel: string
  iconGroupTemperature: number
  iconGroupSystemPrompt: string
  iconGroupUserPrompt: string
  iconGroupStatus: TextBackStatus
  iconGroupError: string
  iconGroups: IconGroup[] | null
  iconAssetStatus: TextBackStatus
  iconAssetError: string
  // 原图参照:开=双图重绘(慢 ~40% 但修复更准),关=单图
  iconAssetUseRef: boolean
  iconAssetSummary: string
  // 叠放显示数据:每组素材文件与各成员回贴矩形(源图像素坐标)
  iconAssetItems: {
    slug: string
    file: string
    status: string
    members: { member: number; paste_x: number; paste_y: number;
               paste_w: number; paste_h: number }[]
  }[]
  iconAssetSourceSize: [number, number] | null
  // 第 7 步:提 icon(sam2 抠图)
  // 提icon源图:text_back=去字图 / origin=原图 / auto=双源逐 icon 按 SAM2 自评分择优
  iconSource: 'text_back' | 'origin' | 'auto'
  // 第 6 步文字层:原图与去字图差值还原,生成后静默上传 pod(text_front.png)
  textFrontStatus: TextBackStatus
  textFrontImageUrl: string
  textFrontError: string
  iconTierParams: Record<IconTier, IconTierParams>
  iconSmallMaxSide: number
  iconLargeMinSide: number
  iconStatus: TextBackStatus
  iconImageUrl: string
  iconError: string
  // 第 8 步"修正":VL 质检抠图问题(粘连/保守),SAM2 带正负点重抠
  iconRefineQaStatus: TextBackStatus
  iconRefineQaError: string
  iconRefineQaInfo: string
  // 第 8 步:去 icon(flux_fill 修补)
  iconBackPrompt: string
  iconBackSeed: number
  iconBackSteps: number
  iconBackGuidance: number
  iconBackGrowMask: number
  iconBackMaskBlur: number
  iconBackMaxPixels: number
  iconBackFillHoles: boolean
  iconBackStatus: TextBackStatus
  iconBackImageUrl: string
  iconBackError: string
  // 第 9~11 步:中景层提取
  midParams: Record<MidKey, MidExtractParams>
  midStatus: Record<MidKey, TextBackStatus>
  midImageUrl: Record<MidKey, string>
  midError: Record<MidKey, string>
  // 第 12 步:中景层破洞图(icon_back 减去 assets/bar/button 的 mask)
  midHoleStatus: TextBackStatus
  midHoleImageUrl: string
  midHoleError: string
  midHoleGrow: number // 洞外扩(腐蚀保留区)像素,0=按图层 alpha 原样挖
  // 第 13 步:修补(对第 12 步破洞图的透明区做 flux_fill,默认挂 icon_back LoRA)
  midFillStatus: TextBackStatus
  midFillImageUrl: string
  midFillError: string
  midFillPrompt: string
  midFillSeed: number
  midFillSteps: number
  midFillGuidance: number
  midFillGrowMask: number
  midFillMaskBlur: number
  midFillMaxPixels: number
  midFillFillHoles: boolean
  // 第 15 步:前中景对比(6/8/10/11/12/14 图层本地拼合,与原图滑杆对比,不上传)
  compareStatus: TextBackStatus
  compareImageUrl: string
  compareError: string
  compareMissing: string
  bboxType: string
  structuredResult: StructuredResult | null
  outputText: string
  outputIsError: boolean
  statusType: StatusType
  statusText: string
  usageModel: string
  usageTokens: string
  usageVisible: boolean
  isSending: boolean

  setField: <K extends keyof DetectionState>(
    key: K,
    value: DetectionState[K],
  ) => void
  setFile: (file: File | null) => void
  setTextBackLocalFile: (file: File | null) => void
  /** 按 run_id 恢复一次历史执行的全部结果;返回 null 表示成功,否则为错误信息 */
  restoreRun: (runId: string) => Promise<string | null>
  setRunpodTarget: (target: string) => void
  uploadOrigin: () => Promise<void>
  runYolo: () => Promise<void>
  setIconTierParam: <K extends keyof IconTierParams>(
    tier: IconTier,
    key: K,
    value: IconTierParams[K],
  ) => void
  runTextBack: () => Promise<void>
  runAnalyzeIcons: () => Promise<void>
  /** 一键清除所有负点(走"不使用负点数据"这条路) */
  clearNegativePoints: () => void
  runAnalyzeGroups: () => Promise<void>
  runExtractIcons: () => Promise<void>
  /** 内部共用:按给定 borders 执行 SAM2 提取(提取/修正共用参数装配) */
  submitIconExtraction: (borders: Record<string, unknown>[]) => Promise<void>
  runRefineIcons: () => Promise<void>
  runIconAsset: () => Promise<void>
  runIconBack: () => Promise<void>
  setMidParam: <K extends keyof MidExtractParams>(
    cat: MidKey,
    key: K,
    value: MidExtractParams[K],
  ) => void
  runMidOne: (cat: MidKey) => Promise<void>
  runMidExtract: () => Promise<void>
  runMidHole: () => Promise<void>
  runMidFill: () => Promise<void>
  runTextFront: () => Promise<void>
  runCompare: () => Promise<void>
  submit: () => Promise<void>
  cancel: () => void
  clearOutput: () => void
  importCachedResult: () => boolean
}

let abortController: AbortController | null = null

/** 取图片自然尺寸(浏览器缓存友好,用于分档判定) */
function loadImageSize(url: string): Promise<[number, number]> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve([img.naturalWidth, img.naturalHeight])
    img.onerror = () => reject(new Error('图片加载失败'))
    img.src = url
  })
}

/** 第 7 步分析点位的传递策略:只有小档 icon 传正点,负点及中/大档一律不传 */
function analyzedToBorders(
  analyzed: AnalyzedIcon[],
  imgW: number,
  imgH: number,
  smallMaxSide: number,
): Record<string, unknown>[] {
  return analyzed.map((a) => {
    const side = Math.max(a.bbox[2] * imgW, a.bbox[3] * imgH)
    return side <= smallMaxSide
      ? { bbox: a.bbox, positive_points: a.positive_points }
      : { bbox: a.bbox }
  })
}

function loadSavedSettings(): SavedSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as SavedSettings
    return {
      apiKey: typeof parsed.apiKey === 'string' ? parsed.apiKey : undefined,
      runpodTarget:
        typeof parsed.runpodTarget === 'string' ? parsed.runpodTarget : undefined,
    }
  } catch {
    return {}
  }
}

function saveSettings(patch: SavedSettings): void {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ ...loadSavedSettings(), ...patch }),
    )
  } catch {
    /* 存储失败不应阻止 API 请求。 */
  }
}

// 历史 run_id 记录(最近的在前,去重,最多 10 条)
const RUN_HISTORY_KEY = 'gpt56-sol-image-analyzer.runHistory.v1'
const RUN_HISTORY_MAX = 10

function loadRunHistory(): string[] {
  try {
    const raw = localStorage.getItem(RUN_HISTORY_KEY)
    const parsed = raw ? (JSON.parse(raw) as unknown) : []
    return Array.isArray(parsed)
      ? parsed.filter((v): v is string => typeof v === 'string').slice(0, RUN_HISTORY_MAX)
      : []
  } catch {
    return []
  }
}

function pushRunHistory(runId: string): string[] {
  const next = [runId, ...loadRunHistory().filter((id) => id !== runId)].slice(
    0,
    RUN_HISTORY_MAX,
  )
  try {
    localStorage.setItem(RUN_HISTORY_KEY, JSON.stringify(next))
  } catch {
    /* 存储失败不影响主流程 */
  }
  return next
}

// 检测结果静默回传 pod:写为 run 目录下的 structure1.json(覆盖式,只保留一份)。
// 隐性行为,失败不打扰用户;没有 run 目录(如纯本地去字图调试)时跳过。
function syncStructureToPod(structured: StructuredResult | null): void {
  const runInfo = useDetectionStore.getState().runInfo
  if (!runInfo || !structured) return
  fetch(`/api/runs/${runInfo.run_id}/files/structure1.json`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(structured),
  }).catch(() => {
    /* 静默同步,失败仅跳过 */
  })
}

// runpod 地址变更后同步给 Vite dev server(/api 代理转发的目标),防抖避免打字期间刷请求
let syncTimer: ReturnType<typeof setTimeout> | undefined
function syncTargetToDevServer(target: string): void {
  clearTimeout(syncTimer)
  if (!/^https?:\/\//.test(target)) return
  syncTimer = setTimeout(() => {
    fetch('/__runpod-target', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target }),
    }).catch(() => {
      /* dev server 不在时忽略 */
    })
  }, 600)
}

const saved = loadSavedSettings()

// 启动时对齐 runpod 地址:localStorage 有值就推给 dev server,
// 没有就用 dev server 的当前值(vite.config.ts 里的默认地址)填充 store
if (saved.runpodTarget) {
  syncTargetToDevServer(saved.runpodTarget)
} else {
  fetch('/__runpod-target')
    .then((r) => r.json())
    .then((d: { target?: string }) => {
      if (d.target) {
        useDetectionStore.setState({ runpodTarget: d.target })
        saveSettings({ runpodTarget: d.target })
      }
    })
    .catch(() => {
      /* dev server 不在时忽略 */
    })
}

export const useDetectionStore = create<DetectionState>((set, get) => ({
  apiKey: saved.apiKey ?? '',
  runpodTarget: saved.runpodTarget ?? '',
  systemPrompt: stepDefaults.detection.systemPrompt,
  userPrompt: stepDefaults.detection.userPrompt,
  yoloResult: '',
  yoloModel: 'game0804_p2',
  isYoloRunning: false,
  yoloError: '',
  reasoningEffort: stepDefaults.detection.reasoningEffort,
  speedMode: stepDefaults.detection.speedMode,
  file: null,
  previewUrl: '',
  runInfo: null,
  isUploading: false,
  uploadError: '',
  textBackPrompt: stepDefaults.textBack.prompt,
  textBackSeed: stepDefaults.textBack.seed,
  textBackSteps: stepDefaults.textBack.steps,
  textBackProtect: true,
  textBackProtectGrow: 8,
  textBackStatus: 'idle',
  textBackImageUrl: '',
  textBackError: '',
  textBackLocalFile: null,
  textBackLocalUrl: '',
  runHistory: loadRunHistory(),
  iconAnalysisModel: stepDefaults.iconAnalysis.model,
  iconAnalysisTemperature: stepDefaults.iconAnalysis.temperature,
  iconAnalysisSystemPrompt: stepDefaults.iconAnalysis.systemPrompt,
  iconAnalysisUserPrompt: stepDefaults.iconAnalysis.userPrompt,
  iconAnalysisStatus: 'idle',
  iconAnalysisError: '',
  analyzedIcons: null,
  iconClampInfo: '',
  iconGroupModel: stepDefaults.iconGroups.model,
  iconGroupTemperature: stepDefaults.iconGroups.temperature,
  iconGroupSystemPrompt: stepDefaults.iconGroups.systemPrompt,
  iconGroupUserPrompt: stepDefaults.iconGroups.userPrompt,
  iconGroupStatus: 'idle',
  iconGroupError: '',
  iconGroups: null,
  iconAssetStatus: 'idle',
  iconAssetError: '',
  iconAssetUseRef: false,
  iconAssetItems: [],
  iconAssetSourceSize: null,
  iconAssetSummary: '',
  iconSource: 'auto',
  textFrontStatus: 'idle',
  textFrontImageUrl: '',
  textFrontError: '',
  iconTierParams: {
    small: { ...stepDefaults.iconExtract.small },
    medium: { ...stepDefaults.iconExtract.medium },
    large: { ...stepDefaults.iconExtract.large },
  },
  iconSmallMaxSide: stepDefaults.iconExtract.smallMaxSide,
  iconLargeMinSide: stepDefaults.iconExtract.largeMinSide,
  iconStatus: 'idle',
  iconImageUrl: '',
  iconRefineQaStatus: 'idle',
  iconRefineQaError: '',
  iconRefineQaInfo: '',
  iconError: '',
  iconBackPrompt: stepDefaults.iconBack.prompt,
  iconBackSeed: stepDefaults.iconBack.seed,
  iconBackSteps: stepDefaults.iconBack.steps,
  iconBackGuidance: stepDefaults.iconBack.guidance,
  iconBackGrowMask: stepDefaults.iconBack.growMask,
  iconBackMaskBlur: stepDefaults.iconBack.maskBlur,
  iconBackMaxPixels: stepDefaults.iconBack.maxPixels,
  iconBackFillHoles: stepDefaults.iconBack.fillHoles,
  iconBackStatus: 'idle',
  iconBackImageUrl: '',
  iconBackError: '',
  midParams: {
    assets: { ...stepDefaults.assetsExtract },
    bar: { ...stepDefaults.barExtract },
    button: { ...stepDefaults.buttonExtract },
  },
  midStatus: midIdle(),
  midImageUrl: midEmpty(),
  midError: midEmpty(),
  midHoleStatus: 'idle',
  midHoleImageUrl: '',
  midHoleError: '',
  midHoleGrow: 0,
  midFillStatus: 'idle',
  midFillImageUrl: '',
  midFillError: '',
  midFillPrompt: stepDefaults.midFill.prompt,
  midFillSeed: stepDefaults.midFill.seed,
  midFillSteps: stepDefaults.midFill.steps,
  midFillGuidance: stepDefaults.midFill.guidance,
  midFillGrowMask: stepDefaults.midFill.growMask,
  midFillMaskBlur: stepDefaults.midFill.maskBlur,
  midFillMaxPixels: stepDefaults.midFill.maxPixels,
  midFillFillHoles: stepDefaults.midFill.fillHoles,
  compareStatus: 'idle',
  compareImageUrl: '',
  compareError: '',
  compareMissing: '',
  bboxType: 'text',
  structuredResult: null,
  outputText: '',
  outputIsError: false,
  statusType: '',
  statusText: '等待请求',
  usageModel: '',
  usageTokens: '',
  usageVisible: false,
  isSending: false,

  setField: (key, value) => set({ [key]: value } as Partial<DetectionState>),

  setFile: (file) => {
    const prev = get().previewUrl
    if (prev) URL.revokeObjectURL(prev)
    // 换了原图,本地去字图也随之失效
    get().setTextBackLocalFile(null)
    set({
      file,
      previewUrl: file ? URL.createObjectURL(file) : '',
      structuredResult: null,
      runInfo: null,
      uploadError: '',
      textBackStatus: 'idle',
      textBackImageUrl: '',
      textBackError: '',
      iconAnalysisStatus: 'idle',
      iconAnalysisError: '',
      analyzedIcons: null, iconClampInfo: '', iconGroupStatus: 'idle', iconGroupError: '', iconGroups: null, iconAssetStatus: 'idle', iconAssetError: '', iconAssetSummary: '', iconAssetItems: [], iconAssetSourceSize: null,
      iconStatus: 'idle',
      iconImageUrl: '',
      iconError: '',
      textFrontStatus: 'idle',
      textFrontImageUrl: '',
      textFrontError: '',
      iconBackStatus: 'idle',
      iconBackImageUrl: '',
      iconBackError: '',
      midStatus: midIdle(),
      midImageUrl: midEmpty(),
      midError: midEmpty(),
      midHoleStatus: 'idle',
      midHoleImageUrl: '',
      midHoleError: '',
      midFillStatus: 'idle',
      midFillImageUrl: '',
      midFillError: '',
      compareStatus: 'idle',
      compareImageUrl: '',
      compareError: '',
      compareMissing: '',
    })
    // 选中图片即自动上传到 RunPod,创建 run 目录
    if (file) void get().uploadOrigin()
  },

  restoreRun: async (runId) => {
    // 第 6 步分析icon的结果暂不恢复(按约定略过)
    try {
      const metaResp = await fetch(`/api/runs/${encodeURIComponent(runId)}`)
      if (!metaResp.ok) return `找不到该 run(HTTP ${metaResp.status})`
      const meta = (await metaResp.json()) as { original_filename?: string }
      const filesResp = await fetch(`/api/runs/${encodeURIComponent(runId)}/files`)
      if (!filesResp.ok) return `读取文件清单失败(HTTP ${filesResp.status})`
      const { files } = (await filesResp.json()) as { files: string[] }
      const has = (name: string) => files.includes(name)
      if (!has('origin.png')) return '该 run 目录里没有 origin.png,无法恢复'

      const fileUrl = (name: string) =>
        `/api/runs/${encodeURIComponent(runId)}/files/${name}?t=${Date.now()}`

      const originBlob = await (await fetch(fileUrl('origin.png'))).blob()
      const file = new File([originBlob], meta.original_filename || 'origin.png', {
        type: originBlob.type || 'image/png',
      })

      let structured: StructuredResult | null = null
      if (has('structure1.json')) {
        try {
          structured = (await (await fetch(fileUrl('structure1.json'))).json()) as StructuredResult
        } catch {
          structured = null
        }
      }
      let yoloText = ''
      if (has('yolo.txt')) {
        yoloText = (await (await fetch(fileUrl('yolo.txt'))).text()).trim()
      }

      const prev = get().previewUrl
      if (prev) URL.revokeObjectURL(prev)
      get().setTextBackLocalFile(null)

      set({
        file,
        previewUrl: URL.createObjectURL(file),
        runInfo: {
          run_id: runId,
          run_dir: `/workspace/servData/${runId}/`,
          origin: `/workspace/servData/${runId}/origin.png`,
        },
        isUploading: false,
        uploadError: '',
        yoloResult: yoloText,
        isYoloRunning: false,
        yoloError: '',
        structuredResult: structured,
        outputText: structured ? JSON.stringify(structured, null, 2) : '',
        outputIsError: false,
        statusType: structured ? 'success' : '',
        statusText: structured ? '已恢复' : '等待请求',
        usageModel: '',
        usageTokens: '',
        usageVisible: false,
        isSending: false,
        textBackStatus: has('text_back.png') ? 'done' : 'idle',
        textBackImageUrl: has('text_back.png') ? fileUrl('text_back.png') : '',
        textBackError: '',
        iconAnalysisStatus: 'idle',
        iconAnalysisError: '',
        analyzedIcons: null, iconClampInfo: '', iconGroupStatus: 'idle', iconGroupError: '', iconGroups: null, iconAssetStatus: 'idle', iconAssetError: '', iconAssetSummary: '', iconAssetItems: [], iconAssetSourceSize: null,
        iconStatus: has('icons.png') ? 'done' : 'idle',
        iconImageUrl: has('icons.png') ? fileUrl('icons.png') : '',
        textFrontStatus: has('text_front.png') ? 'done' : 'idle',
        textFrontImageUrl: has('text_front.png') ? fileUrl('text_front.png') : '',
        textFrontError: '',
        iconError: '',
        iconBackStatus: has('icon_back.png') ? 'done' : 'idle',
        iconBackImageUrl: has('icon_back.png') ? fileUrl('icon_back.png') : '',
        iconBackError: '',
        midStatus: {
          assets: has('assets.png') ? 'done' : 'idle',
          bar: has('bar.png') ? 'done' : 'idle',
          button: has('button.png') ? 'done' : 'idle',
        },
        midImageUrl: {
          assets: has('assets.png') ? fileUrl('assets.png') : '',
          bar: has('bar.png') ? fileUrl('bar.png') : '',
          button: has('button.png') ? fileUrl('button.png') : '',
        },
        midError: midEmpty(),
        midHoleStatus: has('mid_hole.png') ? 'done' : 'idle',
        midHoleImageUrl: has('mid_hole.png') ? fileUrl('mid_hole.png') : '',
        midHoleError: '',
        midFillStatus: has('mid_fill.png') ? 'done' : 'idle',
        midFillImageUrl: has('mid_fill.png') ? fileUrl('mid_fill.png') : '',
        midFillError: '',
        runHistory: pushRunHistory(runId),
      })
      return null
    } catch (error) {
      return error instanceof Error ? error.message : '恢复失败'
    }
  },

  setTextBackLocalFile: (file) => {
    const prev = get().textBackLocalUrl
    if (prev) URL.revokeObjectURL(prev)
    set({
      textBackLocalFile: file,
      textBackLocalUrl: file ? URL.createObjectURL(file) : '',
    })
  },

  setRunpodTarget: (runpodTarget) => {
    set({ runpodTarget })
    saveSettings({ runpodTarget })
    syncTargetToDevServer(runpodTarget.trim())
  },

  runTextBack: async () => {
    const { runInfo, textBackPrompt, textBackSeed, textBackSteps, textBackStatus } = get()
    if (!runInfo || textBackStatus === 'running') return
    // 重新生成去字图后,基于它的 icon 分析和去 icon 结果一定过期;
    // icon 提取仅在"从去文字图提取"模式下过期,从原图提取的结果不受影响
    set({ textBackStatus: 'running', textBackError: '',
          textFrontStatus: 'idle', textFrontImageUrl: '', textFrontError: '',
          iconAnalysisStatus: 'idle', iconAnalysisError: '', analyzedIcons: null, iconClampInfo: '', iconGroupStatus: 'idle', iconGroupError: '', iconGroups: null, iconAssetStatus: 'idle', iconAssetError: '', iconAssetSummary: '', iconAssetItems: [], iconAssetSourceSize: null,
          iconBackStatus: 'idle', iconBackImageUrl: '', iconBackError: '',
          ...(get().iconSource === 'origin'
            ? {}
            : { iconStatus: 'idle' as const, iconImageUrl: '', iconError: '' }) })
    try {
      const { task_id } = await submitTask('text_back', runInfo.run_id, {
        prompt: textBackPrompt.trim() || DEFAULT_TEXT_BACK_PROMPT,
        seed: textBackSeed,
        steps: textBackSteps,
        protect: get().textBackProtect,
        protect_grow: get().textBackProtectGrow,
      })
      await waitTask(task_id, { intervalMs: 2000 })
      // 等待期间换了图,丢弃过期结果
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        textBackStatus: 'done',
        // 带时间戳防浏览器缓存,重新生成后能刷出新图
        textBackImageUrl: `/api/runs/${runInfo.run_id}/files/text_back.png?t=${Date.now()}`,
      })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        textBackStatus: 'error',
        textBackError: error instanceof Error ? error.message : '去文字失败',
      })
    }
  },

  runAnalyzeIcons: async () => {
    const {
      runInfo,
      structuredResult,
      textBackStatus,
      apiKey,
      iconAnalysisStatus,
      iconAnalysisModel,
      iconAnalysisTemperature,
      iconAnalysisSystemPrompt,
      iconAnalysisUserPrompt,
    } = get()
    // 与第 5 步同口径:discard 的审计记录不进入分析
    const icons = pickDetections(structuredResult, 'icon')
    if (!runInfo || textBackStatus !== 'done' || icons.length === 0) return
    if (iconAnalysisStatus === 'running') return
    if (!apiKey.trim()) {
      set({ iconAnalysisStatus: 'error',
            iconAnalysisError: '请先在第 2 步填写 OpenRouter API Key' })
      return
    }
    // 重新分析后,基于旧分析的提取/去 icon 结果都过期
    set({ iconAnalysisStatus: 'running', iconAnalysisError: '',
          iconStatus: 'idle', iconImageUrl: '', iconError: '',
          iconBackStatus: 'idle', iconBackImageUrl: '', iconBackError: '' })
    try {
      const imageDataUrl = await fetchImageAsDataUrl(
        `/api/runs/${runInfo.run_id}/files/text_back.png`,
      )
      const analyzed = await analyzeIcons({
        apiKey: apiKey.trim(),
        model: iconAnalysisModel.trim(),
        temperature: iconAnalysisTemperature,
        systemPrompt: iconAnalysisSystemPrompt,
        userPrompt: iconAnalysisUserPrompt,
        imageDataUrl,
        icons,
      })
      // 等待期间换了图,丢弃过期结果
      if (get().runInfo?.run_id !== runInfo.run_id) return
      // 轮廓不再由模型修正:bbox 用检测框原值回填,供下游提取使用
      const withBbox = analyzed.map((a, i) => ({
        ...a,
        bbox: icons[a.index]?.bbox ?? icons[i]?.bbox ?? a.bbox,
      }))
      // 取点域钳制:落错域的点投影回合法域(纯几何,不看图、不增删点)
      const { icons: clamped, moved } = clampIconPoints(withBbox)
      set({
        iconAnalysisStatus: 'done',
        analyzedIcons: clamped,
        iconClampInfo:
          moved > 0 ? `域钳制修正了 ${moved} 个越域点位` : '所有点位均在合法域内',
      })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        iconAnalysisStatus: 'error',
        iconAnalysisError: error instanceof Error ? error.message : '分析失败',
      })
    }
  },

  clearNegativePoints: () => {
    const { analyzedIcons } = get()
    if (!analyzedIcons?.length) return
    set({
      analyzedIcons: analyzedIcons.map((a) => ({ ...a, negative_points: [] })),
      iconClampInfo: '已一键清除全部负点(提取将只用框+正点)',
    })
  },

  runAnalyzeGroups: async () => {
    const {
      runInfo,
      structuredResult,
      textBackStatus,
      apiKey,
      iconGroupStatus,
      iconGroupModel,
      iconGroupTemperature,
      iconGroupSystemPrompt,
      iconGroupUserPrompt,
    } = get()
    const icons = pickDetections(structuredResult, 'icon')
    if (!runInfo || textBackStatus !== 'done' || icons.length === 0) return
    if (iconGroupStatus === 'running') return
    if (!apiKey.trim()) {
      set({ iconGroupStatus: 'error',
            iconGroupError: '请先在第 2 步填写 OpenRouter API Key' })
      return
    }
    set({ iconGroupStatus: 'running', iconGroupError: '' })
    try {
      const imageDataUrl = await fetchImageAsDataUrl(
        `/api/runs/${runInfo.run_id}/files/text_back.png`,
      )
      const groups = await analyzeIconGroups({
        apiKey: apiKey.trim(),
        model: iconGroupModel.trim(),
        temperature: iconGroupTemperature,
        systemPrompt: iconGroupSystemPrompt,
        userPrompt: iconGroupUserPrompt,
        imageDataUrl,
        icons,
      })
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({ iconGroupStatus: 'done', iconGroups: groups })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        iconGroupStatus: 'error',
        iconGroupError: error instanceof Error ? error.message : '分组失败',
      })
    }
  },

  runExtractIcons: async () => {
    const { runInfo, structuredResult, textBackStatus, iconStatus, iconSource } = get()
    // 有第 7 步分析结果时:只有小档 icon 传正点,负点及中/大档一律不传;
    // 无分析结果退回原始检测框(与第 5 步同口径,过滤 discard)
    const { analyzedIcons, iconSmallMaxSide } = get()
    if (!runInfo) return
    // 涉及去文字图的模式(text_back / auto)需要第 2 步已出结果;纯原图不需要
    if (iconSource !== 'origin' && textBackStatus !== 'done') return
    let borders: Record<string, unknown>[]
    if (analyzedIcons?.length) {
      const srcName = iconSource === 'origin' ? 'origin.png' : 'text_back.png'
      const [iw, ih] = await loadImageSize(
        `/api/runs/${runInfo.run_id}/files/${srcName}`,
      )
      borders = analyzedToBorders(analyzedIcons, iw, ih, iconSmallMaxSide)
    } else {
      borders = pickDetections(structuredResult, 'icon').map((d) => ({
        bbox: d.bbox,
      }))
    }
    if (borders.length === 0) return
    if (iconSource === 'auto') {
      // 双源择优:被文字压住的 icon 锁定去字图(原图上的文字像素会污染候选),
      // 其余逐 icon 由 SAM2 自评分在去字图/原图间择优
      const texts = pickDetections(structuredResult, 'text')
      borders = borders.map((b) => ({
        ...b,
        source:
          detectOverlay([b as never], texts).length > 0 ? 'primary' : 'auto',
      }))
    }
    if (iconStatus === 'running') return
    // 手动重新提取后,上一轮质检结论过期
    set({ iconRefineQaStatus: 'idle', iconRefineQaError: '', iconRefineQaInfo: '' })
    await get().submitIconExtraction(borders)
  },

  submitIconExtraction: async (borders) => {
    const {
      runInfo,
      iconSource,
      iconTierParams,
      iconSmallMaxSide,
      iconLargeMinSide,
    } = get()
    if (!runInfo || borders.length === 0) return
    // icon 重新提取后,旧的去 icon 结果已过期
    set({ iconStatus: 'running', iconError: '',
          iconBackStatus: 'idle', iconBackImageUrl: '', iconBackError: '' })
    try {
      // 中档 = 全局默认;小/大档以 size_rules 按像素长边覆盖
      // (二轮精化/多候选/封孔也随档位走)
      const toRule = (p: IconTierParams) => ({
        padding_ratio: p.paddingRatio,
        min_padding: p.minPadding,
        mask_threshold: p.maskThreshold,
        feather_radius: p.featherRadius,
        crop_scale: p.cropScale,
        refine: p.refine,
        multimask: p.multimask,
        fill_holes: p.fillHoles,
      })
      const { task_id } = await submitTask('sam2', runInfo.run_id, {
        image: iconSource === 'origin' ? 'origin.png' : 'text_back.png',
        ...(iconSource === 'auto' ? { alt_image: 'origin.png' } : {}),
        output: 'icons.png',
        borders,
        ...toRule(iconTierParams.medium),
        size_rules: [
          { max_side: iconSmallMaxSide, ...toRule(iconTierParams.small) },
          { min_side: iconLargeMinSide, ...toRule(iconTierParams.large) },
        ],
      })
      await waitTask(task_id, { intervalMs: 2000 })
      // 等待期间换了图,丢弃过期结果
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        iconStatus: 'done',
        iconImageUrl: `/api/runs/${runInfo.run_id}/files/icons.png?t=${Date.now()}`,
      })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        iconStatus: 'error',
        iconError: error instanceof Error ? error.message : '提取失败',
      })
    }
  },

  runRefineIcons: async () => {
    const {
      runInfo,
      structuredResult,
      iconStatus,
      iconSource,
      apiKey,
      iconRefineQaStatus,
    } = get()
    const icons = pickDetections(structuredResult, 'icon')
    if (!runInfo || iconStatus !== 'done' || icons.length === 0) return
    if (iconRefineQaStatus === 'running') return
    if (!apiKey.trim()) {
      set({ iconRefineQaStatus: 'error',
            iconRefineQaError: '请先在第 2 步填写 OpenRouter API Key' })
      return
    }
    set({ iconRefineQaStatus: 'running', iconRefineQaError: '', iconRefineQaInfo: '' })
    try {
      const srcName = iconSource === 'origin' ? 'origin.png' : 'text_back.png'
      // 抠图结果透明区平铺品红,让质检模型能"看见"哪些被抠掉了
      const [sourceDataUrl, cutoutDataUrl] = await Promise.all([
        fetchImageDataUrl(`/api/runs/${runInfo.run_id}/files/${srcName}`),
        fetchImageDataUrl(`/api/runs/${runInfo.run_id}/files/icons.png`, [255, 0, 255]),
      ])
      const fixes = await analyzeCutoutFixes({
        apiKey: apiKey.trim(),
        sourceDataUrl,
        cutoutDataUrl,
        icons,
      })
      if (get().runInfo?.run_id !== runInfo.run_id) return
      if (fixes.length === 0) {
        set({ iconRefineQaStatus: 'done',
              iconRefineQaInfo: '质检通过:未发现需要修正的 icon' })
        return
      }
      // 全量重抠:有修正的 icon 换上修正框+修正点(质检看的是当前结果,
      // 其点位优先);其余按第 7 步传点策略(仅小档正点)或 box-only
      const { analyzedIcons, iconSmallMaxSide } = get()
      const fixMap = new Map(fixes.map((f) => [f.index, f]))
      let base: Record<string, unknown>[]
      if (analyzedIcons?.length) {
        const [iw, ih] = await loadImageSize(
          `/api/runs/${runInfo.run_id}/files/${srcName}`,
        )
        base = analyzedToBorders(analyzedIcons, iw, ih, iconSmallMaxSide)
      } else {
        base = icons.map((d) => ({ bbox: d.bbox }))
      }
      let borders: Record<string, unknown>[] = base.map((b, i) => {
        const f = fixMap.get(i)
        return f
          ? { bbox: f.bbox,
              positive_points: f.positive_points,
              negative_points: f.negative_points }
          : b
      })
      if (iconSource === 'auto') {
        const texts = pickDetections(structuredResult, 'text')
        borders = borders.map((b) => ({
          ...b,
          source:
            detectOverlay([b as never], texts).length > 0 ? 'primary' : 'auto',
        }))
      }
      set({
        iconRefineQaInfo:
          `修正 ${fixes.length} 个：` +
          fixes.map((f) => `#${f.index} ${f.issue}`).join('；'),
      })
      await get().submitIconExtraction(borders)
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({ iconRefineQaStatus: 'done' })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        iconRefineQaStatus: 'error',
        iconRefineQaError: error instanceof Error ? error.message : '修正失败',
      })
    }
  },

  runIconAsset: async () => {
    const { runInfo, iconGroups, iconStatus, iconAssetStatus } = get()
    if (!runInfo || !iconGroups?.length) return
    if (iconStatus !== 'done') return // 需要 icons.png(提icon完成)
    if (iconAssetStatus === 'running') return
    set({ iconAssetStatus: 'running', iconAssetError: '' })
    try {
      const task = await submitTask('icon_asset', runInfo.run_id, {
        groups: iconGroups.map((g) => ({
          name: g.name, slug: g.slug, bbox: g.bbox,
        })),
        use_ref: get().iconAssetUseRef,
      })
      const result = (await waitTask(task.task_id)) as {
        count?: number
        ok?: number
        statuses?: Record<string, number>
        assets?: DetectionState['iconAssetItems']
        source_size?: [number, number]
      }
      if (get().runInfo?.run_id !== runInfo.run_id) return
      const bad = Object.entries(result?.statuses ?? {})
        .filter(([k]) => k !== 'ok')
        .map(([k, v]) => `${k}×${v}`)
        .join(' ')
      set({
        iconAssetStatus: 'done',
        iconAssetSummary: `${result?.ok ?? 0}/${result?.count ?? 0} 组成功${bad ? `（${bad}）` : ''}`,
        iconAssetItems: result?.assets ?? [],
        iconAssetSourceSize: result?.source_size ?? null,
      })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        iconAssetStatus: 'error',
        iconAssetError: error instanceof Error ? error.message : '素材化失败',
      })
    }
  },

  runIconBack: async () => {
    const {
      runInfo,
      iconStatus,
      iconBackStatus,
      iconBackPrompt,
      iconBackSeed,
      iconBackSteps,
      iconBackGuidance,
      iconBackGrowMask,
      iconBackMaskBlur,
      iconBackMaxPixels,
      iconBackFillHoles,
    } = get()
    if (!runInfo || iconStatus !== 'done') return
    if (iconBackStatus === 'running') return
    // icon_back.png 重新生成后,基于它的中景层提取和破洞图全部过期
    set({ iconBackStatus: 'running', iconBackError: '',
          midStatus: midIdle(), midImageUrl: midEmpty(), midError: midEmpty(),
          midHoleStatus: 'idle', midHoleImageUrl: '', midHoleError: '' })
    try {
      const { task_id } = await submitTask('flux_fill', runInfo.run_id, {
        image: 'text_back.png',
        mask_from: 'icons.png',
        hole_output: 'icon_hole.png',
        output: 'icon_back.png',
        prompt: iconBackPrompt.trim() || DEFAULT_ICON_BACK_PROMPT,
        seed: iconBackSeed,
        steps: iconBackSteps,
        guidance: iconBackGuidance,
        grow_mask: iconBackGrowMask,
        mask_blur: iconBackMaskBlur,
        max_pixels: iconBackMaxPixels,
        fill_holes: iconBackFillHoles,
      })
      await waitTask(task_id, { intervalMs: 2000 })
      // 等待期间换了图,丢弃过期结果
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        iconBackStatus: 'done',
        iconBackImageUrl: `/api/runs/${runInfo.run_id}/files/icon_back.png?t=${Date.now()}`,
      })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        iconBackStatus: 'error',
        iconBackError: error instanceof Error ? error.message : '去 icon 失败',
      })
    }
  },

  runYolo: async () => {
    const { runInfo, isYoloRunning } = get()
    if (!runInfo || isYoloRunning) return
    set({ isYoloRunning: true, yoloError: '' })
    try {
      const { task_id } = await submitTask('yolo', runInfo.run_id, {
        model: get().yoloModel,
      })
      const result = (await waitTask(task_id, { intervalMs: 1500 })) as {
        lines?: string[]
        count?: number
      }
      // 等待期间换了图,丢弃过期结果
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({ yoloResult: (result.lines ?? []).join('\n'), isYoloRunning: false })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        isYoloRunning: false,
        yoloError: error instanceof Error ? error.message : 'YOLO 检测失败',
      })
    }
  },

  setIconTierParam: (tier, key, value) =>
    set({
      iconTierParams: {
        ...get().iconTierParams,
        [tier]: { ...get().iconTierParams[tier], [key]: value },
      },
    }),

  setMidParam: (cat, key, value) =>
    set({
      midParams: {
        ...get().midParams,
        [cat]: { ...get().midParams[cat], [key]: value },
      },
    }),

  runMidOne: async (cat) => {
    const { runInfo, structuredResult, iconBackStatus, midStatus, midParams } = get()
    if (!runInfo || iconBackStatus !== 'done') return
    if (midStatus[cat] === 'running') return
    const borders = pickDetections(structuredResult, cat)
    if (borders.length === 0) return
    const p = midParams[cat]
    set({
      midStatus: { ...get().midStatus, [cat]: 'running' },
      midError: { ...get().midError, [cat]: '' },
      // 中景层变了,旧破洞图和修补图过期
      midHoleStatus: 'idle',
      midHoleImageUrl: '',
      midHoleError: '',
      midFillStatus: 'idle',
      midFillImageUrl: '',
      midFillError: '',
    })
    try {
      // 串行分层:assets 直接从去icon图提;button 从"挖掉 assets 的破洞图"提;
      // bar 从"再挖掉 button 的破洞图"提。前序层未提取时自动降级用已有的最深层。
      const subtractFor: Record<MidKey, MidKey[]> = {
        assets: [],
        button: ['assets'],
        bar: ['assets', 'button'],
      }
      let image = 'icon_back.png'
      const doneLayers = subtractFor[cat].filter(
        (k) => get().midStatus[k] === 'done',
      )
      if (doneLayers.length > 0) {
        const holeName = cat === 'button' ? 'mid_hole_a.png' : 'mid_hole_ab.png'
        const { task_id: holeId } = await submitTask('mid_hole', runInfo.run_id, {
          image: 'icon_back.png',
          sources: doneLayers.map((k) => `${k}.png`),
          output: holeName,
          fill_rgb: [0, 0, 0],
        })
        await waitTask(holeId, { intervalMs: 1000 })
        image = holeName
      }
      const { task_id } = await submitTask('sam2', runInfo.run_id, {
        image,
        output: `${cat}.png`,
        borders,
        padding_ratio: p.paddingRatio,
        min_padding: p.minPadding,
        mask_threshold: p.maskThreshold,
        feather_radius: p.featherRadius,
        crop_scale: p.cropScale,
        refine: p.refine,
        multimask: p.multimask,
        fill_holes: p.fillHoles,
      })
      await waitTask(task_id, { intervalMs: 2000 })
      // 等待期间换了图,丢弃过期结果
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        midStatus: { ...get().midStatus, [cat]: 'done' },
        midImageUrl: {
          ...get().midImageUrl,
          [cat]: `/api/runs/${runInfo.run_id}/files/${cat}.png?t=${Date.now()}`,
        },
      })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        midStatus: { ...get().midStatus, [cat]: 'error' },
        midError: {
          ...get().midError,
          [cat]: error instanceof Error ? error.message : '提取失败',
        },
      })
    }
  },

  runMidExtract: async () => {
    const { runInfo, iconBackStatus, structuredResult, runMidOne } = get()
    if (!runInfo || iconBackStatus !== 'done') return
    // 严格串行:assets → button → bar,后一步都在前一步挖洞后的图上提取,
    // 已移除的元素不再干扰 SAM2 对下一层的判断
    if (pickDetections(structuredResult, 'assets').length > 0) {
      await runMidOne('assets')
    }
    if (pickDetections(structuredResult, 'button').length > 0) {
      await runMidOne('button')
    }
    if (pickDetections(structuredResult, 'bar').length > 0) {
      await runMidOne('bar')
    }
  },

  runMidHole: async () => {
    const { runInfo, iconBackStatus, midStatus, midHoleStatus } = get()
    if (!runInfo || iconBackStatus !== 'done') return
    if (midHoleStatus === 'running') return
    if (!Object.values(midStatus).some((s) => s === 'done')) return
    // 破洞图重新生成后,旧修补结果过期
    set({ midHoleStatus: 'running', midHoleError: '',
          midFillStatus: 'idle', midFillImageUrl: '', midFillError: '' })
    try {
      const { task_id } = await submitTask('mid_hole', runInfo.run_id, {
        grow: get().midHoleGrow,
      })
      await waitTask(task_id, { intervalMs: 1500 })
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        midHoleStatus: 'done',
        midHoleImageUrl: `/api/runs/${runInfo.run_id}/files/mid_hole.png?t=${Date.now()}`,
      })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        midHoleStatus: 'error',
        midHoleError: error instanceof Error ? error.message : '生成失败',
      })
    }
  },

  runMidFill: async () => {
    const {
      runInfo,
      midHoleStatus,
      midFillStatus,
      midFillPrompt,
      midFillSeed,
      midFillSteps,
      midFillGuidance,
      midFillGrowMask,
      midFillMaskBlur,
      midFillMaxPixels,
      midFillFillHoles,
    } = get()
    if (!runInfo || midHoleStatus !== 'done') return
    if (midFillStatus === 'running') return
    set({ midFillStatus: 'running', midFillError: '' })
    try {
      // 修补第 12 步破洞图:底图用 icon_back(洞外像素一致且全不透明),
      // 修补区取 mid_hole.png 的透明区;默认挂 icon_back_fill LoRA
      const { task_id } = await submitTask('flux_fill', runInfo.run_id, {
        image: 'icon_back.png',
        mask_from_holes: 'mid_hole.png',
        output: 'mid_fill.png',
        prompt: midFillPrompt.trim() || DEFAULT_MID_FILL_PROMPT,
        seed: midFillSeed,
        steps: midFillSteps,
        guidance: midFillGuidance,
        grow_mask: midFillGrowMask,
        mask_blur: midFillMaskBlur,
        max_pixels: midFillMaxPixels,
        fill_holes: midFillFillHoles,
      })
      await waitTask(task_id, { intervalMs: 2000 })
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        midFillStatus: 'done',
        midFillImageUrl: `/api/runs/${runInfo.run_id}/files/mid_fill.png?t=${Date.now()}`,
      })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        midFillStatus: 'error',
        midFillError: error instanceof Error ? error.message : '修补失败',
      })
    }
  },

  runTextFront: async () => {
    const { runInfo, textBackStatus, textFrontStatus, textBackImageUrl } = get()
    if (!runInfo || textBackStatus !== 'done') return
    if (textFrontStatus === 'running') return
    set({ textFrontStatus: 'running', textFrontError: '' })
    try {
      const base = `/api/runs/${runInfo.run_id}/files`
      // 原图恒取 pod(本地 previewUrl 可能已被释放);去字图优先当前展示的那份
      const blob = await extractTextFront(
        `${base}/origin.png`,
        textBackImageUrl || `${base}/text_back.png`,
      )
      const put = await fetch(`${base}/text_front.png`, {
        method: 'POST',
        body: blob,
      })
      if (!put.ok) throw new Error(`上传失败:HTTP ${put.status}`)
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        textFrontStatus: 'done',
        textFrontImageUrl: `${base}/text_front.png?t=${Date.now()}`,
      })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        textFrontStatus: 'error',
        textFrontError: error instanceof Error ? error.message : '生成失败',
      })
    }
  },

  runCompare: async () => {
    const { runInfo, compareStatus } = get()
    if (!runInfo || compareStatus === 'running') return
    set({ compareStatus: 'running', compareError: '', compareMissing: '' })
    const loadImg = (url: string) =>
      new Promise<HTMLImageElement>((resolve, reject) => {
        const img = new Image()
        img.onload = () => resolve(img)
        img.onerror = () => reject(new Error('load failed'))
        img.src = url
      })
    try {
      const base = `/api/runs/${runInfo.run_id}/files`
      const t = Date.now()
      const origin = await loadImg(`${base}/origin.png?t=${t}`)
      const canvas = document.createElement('canvas')
      canvas.width = origin.naturalWidth
      canvas.height = origin.naturalHeight
      const ctx = canvas.getContext('2d')!
      ctx.imageSmoothingQuality = 'high'

      // 图层序(自底向上):修补底图 → bar → button → assets → icons → 文字层
      // (与串行提取的移除顺序互逆:先移除的压在上面)
      const layers = [
        ['mid_fill.png', '14修补'],
        ['bar.png', '12提bar'],
        ['button.png', '11提button'],
        ['assets.png', '10提assets'],
        ['icons.png', '8提icon'],
        // ['text_front.png', '6文字层'],
      ] as const
      const missing: string[] = []
      let drawn = 0
      for (const [file, label] of layers) {
        try {
          const img = await loadImg(`${base}/${file}?t=${t}`)
          // 各层可能是 16 对齐尺寸,统一拉伸到原图画布
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
          drawn += 1
        } catch {
          missing.push(label)
        }
      }
      if (drawn === 0) throw new Error('六个图层一个都不存在,请先完成前置步骤')

      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('导出失败'))), 'image/png'),
      )
      if (get().runInfo?.run_id !== runInfo.run_id) return
      const prev = get().compareImageUrl
      if (prev.startsWith('blob:')) URL.revokeObjectURL(prev)
      set({
        compareStatus: 'done',
        compareImageUrl: URL.createObjectURL(blob),
        compareMissing: missing.length ? `缺少图层:${missing.join('、')}` : '',
      })
    } catch (error) {
      if (get().runInfo?.run_id !== runInfo.run_id) return
      set({
        compareStatus: 'error',
        compareError: error instanceof Error ? error.message : '拼合失败',
      })
    }
  },

  uploadOrigin: async () => {
    const file = get().file
    if (!file) return
    set({ isUploading: true, uploadError: '', runInfo: null })
    try {
      const info = await uploadImage(file)
      // 等待期间用户换了图,丢弃过期结果
      if (get().file !== file) return
      set({ runInfo: info, isUploading: false, runHistory: pushRunHistory(info.run_id) })
    } catch (error) {
      if (get().file !== file) return
      set({
        uploadError:
          error instanceof Error ? error.message : '上传失败,请检查 RunPod 地址',
        isUploading: false,
      })
    }
  },

  cancel: () => abortController?.abort(),

  clearOutput: () =>
    set({
      outputText: '',
      outputIsError: false,
      usageVisible: false,
      structuredResult: null,
      statusType: '',
      statusText: '等待请求',
    }),

  importCachedResult: () => {
    try {
      const raw = localStorage.getItem(RESULT_CACHE_KEY)
      if (!raw) return false
      const cached = JSON.parse(raw) as ResultCache
      if (typeof cached.outputText !== 'string' || !cached.outputText) return false
      set({
        outputText: cached.outputText,
        outputIsError: false,
        structuredResult: cached.structuredResult ?? null,
        statusType: 'success',
        statusText: '已导入缓存',
        usageModel: cached.usageModel ?? '',
        usageTokens: cached.usageTokens ?? '',
        usageVisible: Boolean(cached.usageModel),
      })
      // 导入缓存同样触发静默回传
      syncStructureToPod(cached.structuredResult ?? null)
      return true
    } catch {
      return false
    }
  },

  submit: async () => {
    const state = get()
    if (state.isSending) return

    const apiKey = state.apiKey.trim()
    const systemPrompt = state.systemPrompt.trim()
    const userPrompt = state.userPrompt.trim()
    const yoloResult = state.yoloResult.trim()
    const combinedUserPrompt = yoloResult
      ? `${userPrompt}\n\nYOLO结果：\n${yoloResult}`
      : userPrompt

    const fail = (text: string) =>
      set({
        outputText: text,
        outputIsError: true,
        statusType: 'error',
        statusText: '请求失败',
        usageVisible: false,
      })

    if (!apiKey || !systemPrompt || !userPrompt) {
      fail('请完整填写 API Key、系统提示词和用户提示词。')
      return
    }
    if (!state.file) {
      fail('请至少选择一张图片。')
      return
    }
    // 检测需要原图 + 去字图双图:本地上传的去字图优先,否则要求第 2 步已完成
    if (!state.textBackLocalFile && (!state.runInfo || state.textBackStatus !== 'done')) {
      fail('请先完成第 2 步去文字，或在第 2 步上传本地去字图。')
      return
    }

    saveSettings({ apiKey })
    abortController = new AbortController()
    set({
      isSending: true,
      outputText: '',
      outputIsError: false,
      statusType: 'working',
      statusText: '处理中',
      usageVisible: false,
      structuredResult: null,
    })

    try {
      // 单图协议:只传去字图(保护合成后,除文字区外与原图逐像素一致,
      // 原图不再提供,避免模型认错图;text 类由 YOLO 行透传)。本地上传的优先。
      let textBackFile: File
      if (state.textBackLocalFile) {
        textBackFile = state.textBackLocalFile
      } else {
        const tbResp = await fetch(
          `/api/runs/${state.runInfo!.run_id}/files/text_back.png`,
        )
        if (!tbResp.ok) {
          throw new Error(`获取去文字图失败：HTTP ${tbResp.status}`)
        }
        textBackFile = new File([await tbResp.blob()], 'text_back.png', {
          type: 'image/png',
        })
      }
      const content = await buildUserContent(combinedUserPrompt, [textBackFile])
      const cacheKey = createCacheKey(systemPrompt, combinedUserPrompt)
      const provider: Record<string, unknown> = { require_parameters: true }
      if (state.speedMode !== 'balanced') provider.sort = state.speedMode

      const requestBody = {
        model: MODEL,
        instructions: systemPrompt,
        input: [{ type: 'message', role: 'user', content }],
        text: {
          format: {
            type: 'json_schema',
            name: 'ui_detection_result',
            strict: true,
            schema: DETECTION_SCHEMA,
          },
        },
        prompt_cache_key: cacheKey,
        prompt_cache_options: { mode: 'explicit', ttl: '30m' },
        reasoning: { effort: state.reasoningEffort, exclude: true },
        provider,
        store: false,
        stream: false,
      }

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          'X-OpenRouter-Title': 'GPT-5.6 Sol UI Detection',
        },
        body: JSON.stringify(requestBody),
        signal: abortController.signal,
      })

      if (!response.ok) {
        const rawBody = await response.text()
        let detail = rawBody
        try {
          const body = JSON.parse(rawBody)
          detail = body?.error?.message || body?.message || JSON.stringify(body)
        } catch {
          /* 保留非 JSON 的原始错误文本 */
        }
        throw new Error(`HTTP ${response.status}${detail ? `：${detail}` : ''}`)
      }

      const result = await response.json()
      if (result?.error || result?.status === 'failed') {
        throw new Error(result?.error?.message || 'OpenRouter 返回了失败状态。')
      }

      const responseText = extractResponseText(result)
      if (!responseText) {
        throw new Error('模型已结束响应，但没有返回结构化结果。')
      }

      let structuredResult: unknown
      try {
        structuredResult = JSON.parse(responseText)
      } catch {
        throw new Error('模型返回的内容不是有效 JSON。请重试或检查模型端点。')
      }

      const validResult =
        structuredResult &&
        typeof structuredResult === 'object' &&
        !Array.isArray(structuredResult)
          ? (structuredResult as StructuredResult)
          : null

      const reasoningLabel =
        REASONING_OPTIONS.find((o) => o.value === state.reasoningEffort)?.label ??
        state.reasoningEffort
      const speedLabel =
        SPEED_OPTIONS.find((o) => o.value === state.speedMode)?.label ??
        state.speedMode

      let usageTokens = 'Responses API'
      if (result.usage?.total_tokens != null) {
        const details = result.usage.input_tokens_details || {}
        const parts = [
          `总计 ${Number(result.usage.total_tokens).toLocaleString()} tokens`,
        ]
        if (details.cached_tokens > 0) {
          parts.push(`缓存命中 ${Number(details.cached_tokens).toLocaleString()}`)
        }
        if (details.cache_write_tokens > 0) {
          parts.push(
            `缓存写入 ${Number(details.cache_write_tokens).toLocaleString()}`,
          )
        }
        usageTokens = parts.join(' · ')
      }

      const outputText = JSON.stringify(structuredResult, null, 2)
      const usageModel = `${result.model || MODEL} · ${reasoningLabel} · ${speedLabel}`
      set({
        outputText,
        outputIsError: false,
        structuredResult: validResult,
        statusType: 'success',
        statusText: '已完成',
        usageModel,
        usageTokens,
        usageVisible: true,
      })
      // 每次成功生成后写入 localStorage,下次可"导入缓存"复用
      saveResultCache({
        outputText,
        structuredResult: validResult,
        usageModel,
        usageTokens,
      })
      // 静默回传 pod:structure1.json 落到 run 目录
      syncStructureToPod(validResult)
    } catch (error) {
      if ((error as { name?: string })?.name === 'AbortError') {
        const current = get().outputText
        set({
          outputText: current || '请求已取消。',
          outputIsError: !current ? true : get().outputIsError,
          statusType: 'error',
          statusText: '已取消',
        })
      } else {
        set({
          outputText: humanizeError(error),
          outputIsError: true,
          statusType: 'error',
          statusText: '请求失败',
          usageVisible: false,
        })
      }
    } finally {
      abortController = null
      set({ isSending: false })
    }
  },
}))
