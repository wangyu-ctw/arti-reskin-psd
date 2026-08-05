import { useState } from 'react'
import { App, Button, Card, Input, Modal, Tag, Upload } from 'antd'
import type { UploadProps } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { formatBytes, isSupportedImage, MAX_FILE_BYTES } from '../lib/detection'
import RunpodSettingPopover from './RunpodSettingPopover'

const { Dragger } = Upload

export default function UploadPanel() {
  const { message } = App.useApp()
  const { file, previewUrl, runInfo, isUploading, uploadError, setFile, uploadOrigin, restoreRun } =
    useDetectionStore()
  const runHistory = useDetectionStore((s) => s.runHistory)
  const [restoreOpen, setRestoreOpen] = useState(false)
  const [restoreId, setRestoreId] = useState('')
  const [restoring, setRestoring] = useState(false)

  const doRestore = async () => {
    const runId = restoreId.trim()
    if (!runId) {
      message.warning('请输入 run_id')
      return
    }
    setRestoring(true)
    try {
      const err = await restoreRun(runId)
      if (err) {
        message.error(`恢复失败：${err}`)
      } else {
        message.success(`已恢复 run ${runId} 的执行结果`)
        setRestoreOpen(false)
        setRestoreId('')
      }
    } finally {
      setRestoring(false)
    }
  }

  const draggerProps: UploadProps = {
    accept: 'image/png,image/jpeg,image/webp,image/gif',
    multiple: false,
    showUploadList: false,
    beforeUpload: (candidate) => {
      if (!isSupportedImage(candidate)) {
        message.error(`“${candidate.name}”不是支持的图片格式。`)
        return Upload.LIST_IGNORE
      }
      if (candidate.size > MAX_FILE_BYTES) {
        message.error(`“${candidate.name}”超过单张图片 25 MB 限制。`)
        return Upload.LIST_IGNORE
      }
      setFile(candidate)
      return false
    },
  }

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              1
            </span>
            上传图片
          </span>
          <div className="flex items-center gap-2">
            <Button size="small" onClick={() => setRestoreOpen(true)}>
              恢复
            </Button>
            <RunpodSettingPopover />
          </div>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {file ? (
        <div className="flex h-full flex-col gap-3">
          <div className="">
            {previewUrl ? (
              <img
                src={previewUrl}
                alt={`${file.name} 预览`}
                className="h-auto max-w-full object-contain"
              />
            ) : null}
          </div>
          <div className="flex items-center justify-between gap-2">
            <span
              className="min-w-0 truncate text-[11px] text-black/45"
              title={file.name}
            >
              {file.name} · {formatBytes(file.size)}
            </span>
            <Button
              danger
              onClick={() => setFile(null)}
              aria-label={`删除 ${file.name}`}
            >
              删除图片
            </Button>
          </div>
          <div className="text-[11px]">
            {isUploading ? (
              <span className="text-black/45">正在上传到 RunPod…</span>
            ) : runInfo ? (
              <span className="text-[#389e0d]" title={runInfo.run_dir}>
                已上传 · run_id: {runInfo.run_id}
              </span>
            ) : uploadError ? (
              <span className="text-[#cf1322]">
                上传失败：{uploadError}
                <Button
                  type="link"
                  size="small"
                  className="px-1 text-[11px]"
                  onClick={() => void uploadOrigin()}
                >
                  重试
                </Button>
              </span>
            ) : null}
          </div>
        </div>
      ) : (
        <Dragger {...draggerProps} className="h-full">
          <p className="ant-upload-drag-icon">
            <span className="text-2xl text-[#1677ff]">+</span>
          </p>
          <p className="ant-upload-text">点击选择，或拖放图片到这里</p>
          <p className="ant-upload-hint">
            支持 PNG、JPEG、WebP、GIF，仅限一张且不超过 25 MB
          </p>
        </Dragger>
      )}
      <Modal
        title="恢复历史执行"
        open={restoreOpen}
        onOk={() => void doRestore()}
        onCancel={() => setRestoreOpen(false)}
        confirmLoading={restoring}
        okText="恢复"
        cancelText="取消"
      >
        <div className="mb-2 text-[12px] text-black/45">
          输入 run_id（形如 20260729_060002_818989），将从 RunPod
          网盘读取该目录并恢复各步骤结果（第 6 步分析icon除外）。
        </div>
        <Input
          value={restoreId}
          onChange={(e) => setRestoreId(e.target.value)}
          placeholder="run_id"
          spellCheck={false}
          onPressEnter={() => void doRestore()}
        />
        {runHistory.length > 0 ? (
          <div className="mt-3">
            <div className="mb-1 text-[12px] text-black/45">历史记录（点击填入）</div>
            <div className="flex flex-wrap gap-y-1">
              {runHistory.map((id) => (
                <Tag
                  key={id}
                  className="cursor-pointer"
                  color={id === restoreId.trim() ? 'blue' : undefined}
                  onClick={() => setRestoreId(id)}
                >
                  {id}
                </Tag>
              ))}
            </div>
          </div>
        ) : null}
      </Modal>
    </Card>
  )
}
