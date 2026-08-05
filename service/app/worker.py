"""GPU worker:单线程 + FIFO 队列。

- 全服务只有这一个 worker 线程,queue.Queue 保证严格 FIFO;
- 同一时刻最多一个 handler 在跑,即一张 RTX PRO 6000 同时只有一个任务占用 GPU;
- HTTP 层提交任务立即返回 task_id,通过轮询查状态,规避 RunPod Proxy 长请求超时。
"""
import queue
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Optional


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
        self._queue: "queue.Queue[Task]" = queue.Queue()
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._handlers: dict[str, Callable[[dict], Any]] = {}
        self._current: Optional[Task] = None
        self._thread = threading.Thread(target=self._loop, name="gpu-worker", daemon=True)
        self._started = False

    # ---- handler 注册 ----

    def register(self, task_type: str, handler: Callable[[dict], Any]) -> None:
        self._handlers[task_type] = handler

    def known_types(self) -> list[str]:
        return sorted(self._handlers)

    # ---- 生命周期 ----

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    # ---- 提交 / 查询 ----

    def submit(self, task_type: str, payload: dict) -> Task:
        if task_type not in self._handlers:
            raise KeyError(f"unknown task type: {task_type}")
        task = Task(task_type, payload)
        with self._lock:
            self._tasks[task.id] = task
        self._queue.put(task)
        return task

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def queue_position(self, task: Task) -> Optional[int]:
        """排队中的任务在队列里的位置(0 = 下一个执行)。非 queued 状态返回 None。"""
        if task.status != "queued":
            return None
        with self._lock:
            pending = [t for t in self._queue.queue if t.status == "queued"]
        try:
            return pending.index(task)
        except ValueError:
            return None

    def stats(self) -> dict:
        with self._lock:
            current = self._current
            return {
                "queued": self._queue.qsize(),
                "running": current.to_dict() if current else None,
                "total_tasks": len(self._tasks),
            }

    # ---- worker 主循环 ----

    def _loop(self) -> None:
        while True:
            task = self._queue.get()
            with self._lock:
                self._current = task
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
                    self._current = None
                self._queue.task_done()


# 全局唯一 worker 实例(run.sh 用 --workers 1 启动,进程内单例即全局单例)
worker = GPUWorker()
