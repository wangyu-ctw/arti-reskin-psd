"""HTTP 入口。

接口一览:
    GET  /hello                     helloworld,验证服务通
    GET  /health                    服务状态 + 队列情况
    POST /runs                      上传一张原图 -> 创建 run_id 目录
    GET  /runs/{run_id}             查看 run 的 meta.json
    POST /tasks                     提交任务 {type, run_id?, params?} -> 立即返回 task_id
    GET  /tasks/{task_id}           轮询任务状态 / 结果

长任务全部走 POST /tasks + 轮询,HTTP 请求本身秒回,规避 RunPod Proxy 超时。
仅自用,不加 CORS 中间件(浏览器跨域请求会被默认拒绝)。
"""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import comfy, sam2c, storage, tasks, yoloc
from .config import ALLOWED_IMAGE_EXTS, DATA_ROOT
from .worker import worker

app = FastAPI(title="runpod-gpu-service", docs_url=None, redoc_url=None)


@app.on_event("startup")
def _startup() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    tasks.register_all()
    worker.start()


# ---- 基础 ----

@app.get("/hello")
def hello():
    return {"message": "hello world"}


@app.get("/health")
def health():
    import urllib.request as _ur
    try:
        with _ur.urlopen("http://127.0.0.1:8195/health", timeout=1) as r:
            import json as _json
            qwen_up = bool(_json.loads(r.read()).get("ok"))
    except Exception:
        qwen_up = False
    return {"status": "ok", "data_root": str(DATA_ROOT), "worker": worker.stats(),
            "task_types": worker.known_types(), "comfyui_up": comfy.is_up(),
            "sam2_up": sam2c.is_up(), "yolo_up": yoloc.is_up(),
            "qwen_layered_up": qwen_up}


# ---- run 管理 ----

@app.post("/runs")
async def create_run(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(400, f"unsupported file type: {ext!r}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    try:
        meta = storage.create_run(file.filename or f"upload{ext}", data, ext)
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")
    return {
        "run_id": meta["run_id"],
        "run_dir": f"{DATA_ROOT / meta['run_id']}/",
        "origin": meta["origin_path"],
    }


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    try:
        return storage.read_meta(run_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/runs/{run_id}/files")
def list_run_files(run_id: str):
    """列出 run 目录下的文件名(前端"恢复"功能据此决定恢复哪些步骤)。"""
    try:
        run_dir = storage.get_run_dir(run_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    # 递归列出(含子目录,如 bar/),返回相对路径
    return {"files": sorted(
        str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()
    )}


@app.get("/runs/{run_id}/files/{filename:path}")
def get_run_file(run_id: str, filename: str):
    """读取 run 目录下的文件(如 text_back.png),给前端展示用。"""
    try:
        run_dir = storage.get_run_dir(run_id).resolve()
    except ValueError as e:
        raise HTTPException(404, str(e))
    path = (run_dir / filename).resolve()
    # 防目录穿越:必须还在 run 目录内
    if not str(path).startswith(str(run_dir) + "/"):
        raise HTTPException(403, "path escapes run dir")
    if not path.is_file():
        raise HTTPException(404, f"file not found: {filename}")
    # run 文件会被覆写(text_back 残字补洞、icons 修正重抠等),禁止浏览器缓存
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.post("/runs/{run_id}/files/{filename:path}")
async def put_run_file(run_id: str, filename: str, request: Request):
    """把请求体原样写入 run 目录下的文件(存在则覆盖),如前端回传 structure1.json。
    支持一层子目录(如 bar/xxx.png,13+ 步产物集中存放),自动建目录。"""
    try:
        run_dir = storage.get_run_dir(run_id).resolve()
    except ValueError as e:
        raise HTTPException(404, str(e))
    path = (run_dir / filename).resolve()
    # 防目录穿越:必须还在 run 目录内,且最多一层子目录
    if not str(path).startswith(str(run_dir) + "/"):
        raise HTTPException(403, "path escapes run dir")
    if path.parent != run_dir and path.parent.parent != run_dir:
        raise HTTPException(403, "at most one subdirectory level allowed")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")
    path.write_bytes(body)
    return {"path": str(path), "bytes": len(body)}


SEEDSEEK_ROOT = Path("/workspace/seedseek")


@app.post("/seedseek/{subpath:path}")
async def put_seedseek_file(subpath: str, request: Request):
    """找 seed 工具的结果集中存放:写入 /workspace/seedseek/<会话>/<文件>。

    只允许两级相对路径(会话目录/文件名),防目录穿越。
    """
    path = (SEEDSEEK_ROOT / subpath).resolve()
    if SEEDSEEK_ROOT.resolve() not in path.parents:
        raise HTTPException(403, "path must stay inside /workspace/seedseek")
    if len(path.relative_to(SEEDSEEK_ROOT.resolve()).parts) > 2:
        raise HTTPException(403, "at most <session>/<filename>")
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"path": str(path), "bytes": len(body)}


# ---- 任务提交 / 轮询 ----

class SubmitTaskRequest(BaseModel):
    type: str                 # hello / omnipsd / yolo / sam2
    run_id: Optional[str] = None
    params: dict = {}


@app.post("/tasks")
def submit_task(req: SubmitTaskRequest):
    payload = dict(req.params)
    if req.run_id is not None:
        try:
            storage.get_run_dir(req.run_id)  # 提前校验,排错友好
        except ValueError as e:
            raise HTTPException(404, str(e))
        payload["run_id"] = req.run_id
    try:
        task = worker.submit(req.type, payload)
    except KeyError as e:
        raise HTTPException(400, str(e.args[0]))
    if req.run_id is not None:
        storage.append_task_record(req.run_id, {"task_id": task.id, "type": task.type,
                                                "created_at": task.created_at})
    return {"task_id": task.id, "status": task.status,
            "queue_position": worker.queue_position(task)}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = worker.get(task_id)
    if task is None:
        raise HTTPException(404, f"task not found: {task_id}")
    return task.to_dict(queue_position=worker.queue_position(task))
