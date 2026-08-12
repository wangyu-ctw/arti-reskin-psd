import { useState } from 'react'
import { Button, Card, Checkbox, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'
import ZoomableCanvas from './ZoomableCanvas'

/**
 * 第 6 步:文字层 = SAM2 按结构化检测的 text 框从原图抠字(text_front_sam.png,
 * 配置随第 8 步"小图标"档),再用第 9 步同款 flux_fill 修补抠字后的洞
 * (text_back_sam.png,展示在下方)。
 */
export default function TextFrontPanel() {
  const {
    runInfo,
    structuredResult,
    textFrontStatus,
    textFrontImageUrl,
    textFrontError,
    textBackSamImageUrl,
    runTextFront,
  } = useDetectionStore()
  const [showTextBoxes, setShowTextBoxes] = useState(false)

  const textCount = pickDetections(structuredResult, 'text').length
  const canRun = Boolean(runInfo) && textCount > 0
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
            SAM2 抠字 → fill 补底,两步串行执行中…
          </span>
        </div>
      ) : textFrontStatus === 'done' ? (
        <div className="flex flex-col gap-2">
          <span className="text-[12px] font-bold">text_front_sam(抠出的文字层)</span>
          <ZoomableCanvas
            src={textFrontImageUrl}
            alt="文字层"
            boxes={textBoxes}
          />
          {textBackSamImageUrl ? (
            <>
              <span className="mt-1 text-[12px] font-bold">
                text_back_sam(抠字后 fill 补底)
              </span>
              <img
                src={textBackSamImageUrl}
                alt="抠字补底图"
                className="h-auto max-w-full rounded border border-neutral-200 object-contain"
              />
            </>
          ) : null}
          <p className="m-0 text-center text-[10px] text-black/45">
            上图滚轮缩放 · 拖动平移 · 双击复位
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
              ? '点击右上角"生成"：SAM2 按 text 框抠字(第 8 步小图标档配置)→ 第 9 步同款 fill 修补'
              : '需要第 5 步检测结果中有 text'}
          </span>
        </div>
      )}
    </Card>
  )
}
