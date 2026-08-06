import { useState } from 'react'
import { Button, Card, Checkbox, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'
import ZoomableCanvas from './ZoomableCanvas'

/**
 * 第 6 步:文字层 = 原图与去字图的差值还原(text_front.png,生成后静默上传 pod)。
 * 手动点"生成"才计算;checkbox 勾选后叠加结构化检测结果里的 text 框。
 */
export default function TextFrontPanel() {
  const {
    runInfo,
    structuredResult,
    textBackStatus,
    textFrontStatus,
    textFrontImageUrl,
    textFrontError,
    runTextFront,
  } = useDetectionStore()
  const [showTextBoxes, setShowTextBoxes] = useState(false)

  const canRun = Boolean(runInfo) && textBackStatus === 'done'
  const textBoxes = showTextBoxes
    ? pickDetections(structuredResult, 'text').map((item) => item.bbox)
    : undefined

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              6
            </span>
            文字层
          </span>
          <div className="flex items-center gap-2">
            <Checkbox
              checked={showTextBoxes}
              onChange={(e) => setShowTextBoxes(e.target.checked)}
            >
              <span className="text-[12px]">bbox</span>
            </Checkbox>
            <Button
              size="small"
              type="primary"
              ghost
              loading={textFrontStatus === 'running'}
              disabled={!canRun}
              onClick={() => void runTextFront()}
            >
              生成
            </Button>
          </div>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {textFrontStatus === 'running' ? (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          <Spin />
          <span className="text-[12px] text-black/45">
            正在还原文字层并上传…
          </span>
        </div>
      ) : textFrontStatus === 'done' ? (
        <div className="flex flex-col gap-2">
          <ZoomableCanvas
            src={textFrontImageUrl}
            alt="文字层"
            boxes={textBoxes}
          />
          <p className="m-0 text-center text-[10px] text-black/45">
            滚轮缩放 · 拖动平移 · 双击复位
          </p>
        </div>
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
          {textFrontStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-[12px] text-[#cf1322]">
              生成失败：{textFrontError}
            </div>
          ) : null}
          <span className="px-2 text-[12px] text-black/45">
            {canRun
              ? '点击右上角"生成"：原图与去字图取差值还原文字层，完成后自动上传 pod'
              : '需要第 2 步去文字完成'}
          </span>
        </div>
      )}
    </Card>
  )
}
