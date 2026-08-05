// RunPod GPU 服务的调用封装。
// 所有请求走 /api 前缀,由 Vite dev server 转发到当前 RunPod 地址(见 vite.config.ts)。

export interface CreateRunResult {
  run_id: string
  run_dir: string
  origin: string
}

export interface TaskInfo {
  task_id: string
  type: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  created_at: number
  started_at: number | null
  finished_at: number | null
  result: unknown
  error: string | null
  queue_position?: number | null
}

export type TaskType =
  | 'hello'
  | 'text_back'
  | 'text_back_cold'
  | 'comfy_workflow'
  | 'flux_fill'
  | 'mid_hole'
  | 'omnipsd'
  | 'yolo'
  | 'sam2'

async function toJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

/** 读当前 RunPod 代理地址 */
export async function getRunpodTarget(): Promise<string> {
  const data = await toJson<{ target: string }>(await fetch('/__runpod-target'))
  return data.target
}

/** 改 RunPod 代理地址(持久化在 Vite dev server 侧) */
export async function setRunpodTarget(target: string): Promise<string> {
  const data = await toJson<{ target: string }>(
    await fetch('/__runpod-target', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target }),
    }),
  )
  return data.target
}

/** 上传原图,创建 run 目录,原图存为 <run_dir>/origin.png */
export async function uploadImage(file: File): Promise<CreateRunResult> {
  const fd = new FormData()
  fd.append('file', file)
  return toJson(await fetch('/api/runs', { method: 'POST', body: fd }))
}

/** 提交任务,秒回 task_id(GPU 上严格 FIFO 串行执行) */
export async function submitTask(
  type: TaskType,
  runId?: string,
  params: Record<string, unknown> = {},
): Promise<{ task_id: string; status: string; queue_position: number | null }> {
  return toJson(
    await fetch('/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, run_id: runId, params }),
    }),
  )
}

/** 查一次任务状态 */
export async function getTask(taskId: string): Promise<TaskInfo> {
  return toJson(await fetch(`/api/tasks/${taskId}`))
}

/** 轮询直到任务结束,成功返回 result,失败抛错 */
export async function waitTask(
  taskId: string,
  opts: { intervalMs?: number; signal?: AbortSignal; onUpdate?: (t: TaskInfo) => void } = {},
): Promise<unknown> {
  const { intervalMs = 1500, signal, onUpdate } = opts
  for (;;) {
    if (signal?.aborted) throw new Error('aborted')
    const task = await getTask(taskId)
    onUpdate?.(task)
    if (task.status === 'succeeded') return task.result
    if (task.status === 'failed') throw new Error(task.error ?? 'task failed')
    await new Promise((r) => setTimeout(r, intervalMs))
  }
}
