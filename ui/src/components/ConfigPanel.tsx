import { useState } from 'react'
import { Button, Card, Input, Select } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { REASONING_OPTIONS, SPEED_OPTIONS } from '../lib/detection'
import YoloViewerModal from './YoloViewerModal'

const { TextArea } = Input

function FieldLabel({ text, hint }: { text: string; hint?: string }) {
  return (
    <div className="mb-2 flex items-center justify-between">
      <span className="text-[13px] font-bold">{text}</span>
      {hint ? <span className="text-[11px] text-black/45">{hint}</span> : null}
    </div>
  )
}

export default function ConfigPanel() {
  const {
    apiKey,
    reasoningEffort,
    speedMode,
    systemPrompt,
    userPrompt,
    yoloResult,
    isYoloRunning,
    yoloError,
    runInfo,
    isSending,
    setField,
    submit,
    cancel,
    runYolo,
  } = useDetectionStore()

  const [yoloViewerOpen, setYoloViewerOpen] = useState(false)

  return (
    <Card
      title={
        <span className="text-[15px] font-bold">
          <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
            3
          </span>
          配置请求
        </span>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      <div className="flex flex-col gap-[18px]">
        <div>
          <FieldLabel text="OpenRouter API Key" hint="本机浏览器自动保存" />
          <Input.Password
            value={apiKey}
            onChange={(e) => setField('apiKey', e.target.value)}
            placeholder="sk-or-v1-..."
            autoComplete="off"
            spellCheck={false}
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <FieldLabel text="推理强度" />
            <Select
              className="w-full"
              value={reasoningEffort}
              onChange={(v) => setField('reasoningEffort', v)}
              options={REASONING_OPTIONS}
            />
          </div>
          <div>
            <FieldLabel text="速度模式" />
            <Select
              className="w-full"
              value={speedMode}
              onChange={(v) => setField('speedMode', v)}
              options={SPEED_OPTIONS}
            />
          </div>
        </div>
        <div>
          <FieldLabel text="系统提示词" />
          <TextArea
            value={systemPrompt}
            onChange={(e) => setField('systemPrompt', e.target.value)}
            placeholder="例如：你是一名专业的视觉质检员。请严格按照指定规则分析图片，并输出清晰、可核验的结论。"
            rows={10}
            showCount
          />
        </div>

        <div>
          <FieldLabel text="用户提示词" />
          <TextArea
            value={userPrompt}
            onChange={(e) => setField('userPrompt', e.target.value)}
            placeholder="例如：检查图片中的文字、布局和异常元素，并按问题、位置、建议三个字段返回。"
            rows={10}
            showCount
          />
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between">
            <span className="text-[13px] font-bold">YOLO结果</span>
            <div className="flex gap-2">
              <Button
                size="small"
                disabled={!yoloResult.trim()}
                onClick={() => setYoloViewerOpen(true)}
              >
                查看
              </Button>
              <Button
                size="small"
                loading={isYoloRunning}
                disabled={!runInfo}
                onClick={() => void runYolo()}
              >
                生成Yolo结果
              </Button>
            </div>
          </div>
          <TextArea
            value={yoloResult}
            onChange={(e) => setField('yoloResult', e.target.value)}
            placeholder="粘贴 YOLO 检测结果，或点击右上角生成；发送时会拼接在用户提示词后面。"
            rows={10}
          />
          {yoloError ? (
            <div className="mt-1 text-[11px] text-[#cf1322]">
              YOLO 检测失败：{yoloError}
            </div>
          ) : null}
        </div>

        <div className="flex gap-[10px]">
          <Button
            type="primary"
            block
            loading={isSending}
            onClick={() => void submit()}
          >
            {isSending ? '正在生成…' : '发送给 GPT-5.6 Sol →'}
          </Button>
          {isSending ? (
            <Button danger onClick={cancel}>
              取消
            </Button>
          ) : null}
        </div>
      </div>

      <YoloViewerModal
        open={yoloViewerOpen}
        onClose={() => setYoloViewerOpen(false)}
      />
    </Card>
  )
}
