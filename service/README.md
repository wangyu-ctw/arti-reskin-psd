# RunPod GPU Service

上传整个 `service/` 目录到 `/workspace/`,然后:

```bash
bash /workspace/service/run.sh
```

首次启动会自动 `pip install`,之后监听 `0.0.0.0:8888`(RunPod 暴露 8888 端口即可访问)。

## 架构

- **异步任务模式**:`POST /tasks` 立即返回 `task_id`,用 `GET /tasks/{task_id}` 轮询结果。HTTP 请求本身都是秒回,规避 RunPod Proxy 的长请求超时。
- **GPU 串行**:全服务只有一个 worker 线程消费 FIFO 队列(`app/worker.py`),同一时刻最多一个任务占用 GPU。`run.sh` 用 `--workers 1` 启动,保证队列全局唯一。
- **数据目录**:所有业务数据在 `/workspace/servData/`。每次 `POST /runs` 上传一张原图,创建一个微秒级时间戳目录(即 `run_id`),后续 OmniPSD / YOLO / SAM2 任务都带同一个 `run_id`,输出写回该目录:

```
/workspace/servData/20260723_153045_123456/
├── origin.png        上传的原图(统一转成 PNG)
├── meta.json         run 元信息 + 任务记录
├── omnipsd/          各任务输出(后续实现)
├── yolo/
└── sam2/
```

- 仅自用,未开启 CORS。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/hello` | helloworld |
| GET | `/health` | 服务状态 + 队列情况 |
| POST | `/runs` | 上传原图(multipart `file` 字段),存为 `<run_dir>/origin.png`,返回 `run_id` 和 `run_dir` |
| GET | `/runs/{run_id}` | 查看 run 的 meta |
| POST | `/tasks` | 提交任务 `{"type": "...", "run_id": "...", "params": {...}}` |
| GET | `/tasks/{task_id}` | 轮询任务状态 / 结果 |

任务状态流转:`queued → running → succeeded / failed`。排队中的任务带 `queue_position`(0 = 下一个执行)。

## 使用示例

```bash
BASE=https://<pod-id>-8888.proxy.runpod.net   # 或 http://<ip>:8888

curl $BASE/hello
curl $BASE/health

# 1. 上传原图,拿 run_id
RUN_ID=$(curl -s -F "file=@test.png" $BASE/runs | python3 -c "import sys,json;print(json.load(sys.stdin)['run_id'])")

# 2. 提交任务(hello 是演示任务,模拟占用 GPU 3 秒)
TASK_ID=$(curl -s -X POST $BASE/tasks -H "Content-Type: application/json" \
  -d "{\"type\": \"hello\", \"run_id\": \"$RUN_ID\", \"params\": {\"name\": \"runpod\", \"sleep\": 3}}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['task_id'])")

# 3. 轮询结果
curl $BASE/tasks/$TASK_ID
```

## 后续接新模型

在 [app/tasks.py](app/tasks.py) 里实现对应 handler(`handle_omnipsd` / `handle_yolo` / `handle_sam2`):

- 签名 `handler(payload: dict) -> 可JSON序列化对象`,在 worker 线程里串行执行,可放心独占 GPU;
- 用 `storage.get_run_dir(payload["run_id"])` 拿目录,输出写到 `run_dir / "yolo"` 等子目录;
- 模型权重建议模块级加载一次、常驻显存,避免每个任务重复加载。
