import { App, Button, Card, Input, InputNumber, Popover, Spin, Upload } from 'antd'
import { SettingOutlined, UploadOutlined } from '@ant-design/icons'
import {
  DEFAULT_TEXT_BACK_PROMPT,
  useDetectionStore,
} from '../stores/useDetectionStore'

const { TextArea } = Input

function SettingsPopover() {
  const { textBackPrompt, textBackSeed, textBackSteps, setField } =
    useDetectionStore()

  return (
    <Popover
      trigger="hover"
      placement="bottomRight"
      content={
        <div className="flex w-[340px] flex-col gap-3">
          <div>
            <div className="mb-1 text-[13px] font-bold">提示词</div>
            <TextArea
              value={textBackPrompt}
              onChange={(e) => setField('textBackPrompt', e.target.value)}
              placeholder={DEFAULT_TEXT_BACK_PROMPT}
              rows={6}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="mb-1 text-[13px] font-bold">Step</div>
              <InputNumber
                className="w-full"
                value={textBackSteps}
                onChange={(v) => setField('textBackSteps', v ?? 20)}
                min={1}
                max={50}
                precision={0}
              />
            </div>
            <div>
              <div className="mb-1 text-[13px] font-bold">SEED</div>
              <InputNumber
                className="w-full"
                value={textBackSeed}
                onChange={(v) => setField('textBackSeed', v ?? 5)}
                min={0}
                precision={0}
              />
            </div>
          </div>
        </div>
      }
    >
      <SettingOutlined
        className="cursor-pointer text-[16px] text-black/45 transition-colors hover:text-[#1677ff]"
        aria-label="去文字参数设置"
      />
    </Popover>
  )
}

export default function TextBackPanel() {
  const { message } = App.useApp()
  const {
    runInfo,
    textBackStatus,
    textBackImageUrl,
    textBackError,
    textBackLocalUrl,
    setTextBackLocalFile,
    runTextBack,
  } = useDetectionStore()

  const acceptLocalImage = (candidate: File): boolean => {
    if (!candidate.type.startsWith('image/')) {
      message.error(`“${candidate.name}”不是图片文件。`)
      return false
    }
    setTextBackLocalFile(candidate)
    return true
  }

  // 整张卡片是隐形拖放区:拖图片进来即作为本地去字图,无任何可见的拖拽样式
  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) acceptLocalImage(dropped)
  }

  return (
    <div
      className="h-full w-full"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
    >
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              2
            </span>
            去文字
          </span>
          <div className="flex items-center gap-2">
            {textBackStatus === 'done' ? (
              <Button size="small" onClick={() => void runTextBack()}>
                重新生成
              </Button>
            ) : null}
            <Upload
              accept="image/*"
              showUploadList={false}
              beforeUpload={(candidate) => {
                acceptLocalImage(candidate)
                return false
              }}
            >
              <Button size="small" icon={<UploadOutlined />}>
                上传
              </Button>
            </Upload>
            <SettingsPopover />
          </div>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {textBackLocalUrl ? (
        <div className="mb-3 rounded border border-[#91caff] bg-[#e6f4ff] p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[12px]">
              已使用本地去字图（仅供第 3 步检测，不上传网盘）
            </span>
            <Button size="small" type="link" onClick={() => setTextBackLocalFile(null)}>
              清除
            </Button>
          </div>
          <img
            src={textBackLocalUrl}
            alt="本地去字图"
            className="mt-2 h-auto max-w-full object-contain"
          />
        </div>
      ) : null}
      {textBackStatus === 'running' ? (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          <Spin />
          <span className="text-[12px] text-black/45">
            正在去文字…（GPU 队列串行执行，请稍候）
          </span>
        </div>
      ) : textBackStatus === 'done' ? (
        <img
          src={textBackImageUrl}
          alt="去文字结果"
          className="h-auto max-w-full object-contain"
        />
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          {textBackStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              去文字失败：{textBackError}
            </div>
          ) : null}
          <Button
            type="primary"
            disabled={!runInfo}
            onClick={() => void runTextBack()}
          >
            {textBackStatus === 'error' ? '重试' : '去文字'}
          </Button>
          {!runInfo ? (
            <span className="text-[12px] text-black/45">
              请先在第 1 步上传图片
            </span>
          ) : null}
        </div>
      )}
    </Card>
    </div>
  )
}
