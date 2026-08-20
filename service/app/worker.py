"""GPU worker:FIFO 队列,支持单/双泳道。

- 单卡(默认):一条 worker 线程,全部任务严格 FIFO,GPU 串行;
- 双卡(SERVICE_LANE_MODE=dual,run.sh 按 GPU 数设定):两条泳道并行——
  gpu0 泳道跑 ComfyUI/FLUX 系任务,gpu1 泳道跑 YOLO/SAM2/CPU 任务;
  泳道内部仍 FIFO。跨泳道的复合任务(panel_peel 等)归 gpu0,
  中途调 sam2/yolo daemon 时由 daemon 自身的锁串行,安全;
- HTTP 层提交任务立即返回 task_id,通过轮询查状态,规避 RunPod Proxy 长请求超时。
"""
import os
import queue
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Optional

# 双泳道下走 gpu1 的任务类型(检测/分割/纯 CPU);其余默认 gpu0(Comfy 系)
# p2_sixslot/p2_panelz 是分钟级 qwen 生成,走 gpu0 主泳道;
# 轻任务(检测/审核/抠取)走 gpu1 泳道,与生成并行不排队
GPU1_TYPES = {"yolo", "sam2", "hello", "mid_hole", "panel_asset", "element_extract",
              "p2_detect", "p2_yolo", "p2_gpt", "p2_extract", "p2_assets",
              "p2_layer_yolo", "p2_inventory", "p2_cascade", "p2_psd", "p2_recompose",
              "icon_asset", "qwen_layered"}


class Task:
    def __init__(self, task_type: str, payload: dict):
        self.id = uuid.uuid4().hex[:16]
        self.type = task_type
        self.payload = payload
        self.status = "queued"  # queued -> running -> succeeded / failed
        self.result: Any = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def to_dict(self, queue_position: Optional[int] = None) -> dict:
        d = {
            "task_id": self.id,
            "type": self.type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }
        if queue_position is not None:
            d["queue_position"] = queue_position
        return d


class GPUWorker:
    def __init__(self):
        self._dual = os.environ.get("SERVICE_LANE_MODE", "single") == "dual"
        self._lanes = ("gpu0", "gpu1") if self._dual else ("gpu0",)
        self._queues: dict[str, "queue.Queue[Task]"] = {
            lane: queue.Queue() for lane in self._lanes}
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._handlers: dict[str, Callable[[dict], Any]] = {}
        self._current: dict[str, Optional[Task]] = {lane: None for lane in self._lanes}
        self._threads = [
            threading.Thread(target=self._loop, args=(lane,),
                             name=f"gpu-worker-{lane}", daemon=True)
            for lane in self._lanes]
        self._started = False

    def _lane_of(self, task_type: str) -> str:
        if self._dual and task_type in GPU1_TYPES:
            return "gpu1"
        return "gpu0"

    # ---- handler 注册 ----

    def register(self, task_type: str, handler: Callable[[dict], Any]) -> None:
        self._handlers[task_type] = handler

    def known_types(self) -> list[str]:
        return sorted(self._handlers)

    # ---- 生命周期 ----

    def start(self) -> None:
        if not self._started:
            self._started = True
            for t in self._threads:
                t.start()

    # ---- 提交 / 查询 ----

    def submit(self, task_type: str, payload: dict) -> Task:
        if task_type not in self._handlers:
            raise KeyError(f"unknown task type: {task_type}")
        task = Task(task_type, payload)
        with self._lock:
            self._tasks[task.id] = task
        self._queues[self._lane_of(task_type)].put(task)
        return task

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def queue_position(self, task: Task) -> Optional[int]:
        """排队中的任务在队列里的位置(0 = 下一个执行)。非 queued 状态返回 None。"""
        if task.status != "queued":
            return None
        lane = self._lane_of(task.type)
        with self._lock:
            pending = [t for t in self._queues[lane].queue if t.status == "queued"]
        try:
            return pending.index(task)
        except ValueError:
            return None

    def stats(self) -> dict:
        with self._lock:
            running = {lane: (t.to_dict() if t else None)
                       for lane, t in self._current.items()}
            any_running = next((v for v in running.values() if v), None)
            return {
                "queued": sum(q.qsize() for q in self._queues.values()),
                # 兼容旧字段:任一泳道在跑就展示它
                "running": any_running,
                "total_tasks": len(self._tasks),
                "lanes": {lane: {"queued": self._queues[lane].qsize(),
                                 "running": running[lane]}
                          for lane in self._lanes},
            }

    # ---- worker 主循环 ----

    def _loop(self, lane: str) -> None:
        q = self._queues[lane]
        while True:
            task = q.get()
            with self._lock:
                self._current[lane] = task
            task.status = "running"
            task.started_at = time.time()
            try:
                handler = self._handlers[task.type]
                task.result = handler(task.payload)
                task.status = "succeeded"
            except Exception:
                task.status = "failed"
                task.error = traceback.format_exc()
            finally:
                task.finished_at = time.time()
                with self._lock:
                    self._current[lane] = None
                q.task_done()


# 全局唯一 worker 实例(run.sh 用 --workers 1 启动,进程内单例即全局单例)
worker = GPUWorker()
