import { useState } from 'react'
import { Button, Card, Checkbox, InputNumber, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import { pickDetections } from '../lib/detection'
import ZoomableCanvas from './ZoomableCanvas'

/**
 * 第 13 步:中景层破洞图 = 第 9 步结果图挖掉 assets/bar/button 的 mask。
 * 手动点"生成"才计算,不自动生成。
 */
export default function MidHolePanel() {
  const {
    structuredResult,
    iconBackStatus,
    midStatus,
    midHoleStatus,
    midHoleImageUrl,
    midHoleError,
    midHoleGrow,
    runMidHole,
  } = useDetectionStore()
  const setField = useDetectionStore((s) => s.setField)
  const [showPanelBoxes, setShowPanelBoxes] = useState(false)

  const anyMidDone = Object.values(midStatus).some((s) => s === 'done')
  const canRun = iconBackStatus === 'done' && anyMidDone
  const panelBoxes = showPanelBoxes
    ? pickDetections(structuredResult, 'panel').map((item) => item.bbox)
    : undefined

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              13
            </span>
            中景层破洞图
          </span>
          <div className="flex items-center gap-2">
            <span className="text-[12px] text-black/60">腐蚀px</span>
            <InputNumber
              size="small"
              className="w-14"
              value={midHoleGrow}
              onChange={(v) => setField('midHoleGrow', v ?? 0)}
              disabled={midHoleStatus === 'running'}
              min={0}
              max={50}
              precision={0}
            />
            <Checkbox
              checked={showPanelBoxes}
              onChange={(e) => setShowPanelBoxes(e.target.checked)}
            >
              <span className="text-[12px]">panel</span>
            </Checkbox>
            <Button
              size="small"
              type="primary"
              ghost
              loading={midHoleStatus === 'running'}
              disabled={!canRun}
              onClick={() => void runMidHole()}
            >
              生成
            </Button>
          </div>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {midHoleStatus === 'running' ? (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          <Spin />
          <span className="text-[12px] text-black/45">正在生成破洞图…</span>
        </div>
      ) : midHoleStatus === 'done' ? (
        <div className="flex flex-col gap-2">
          <ZoomableCanvas src={midHoleImageUrl} alt="中景层破洞图" boxes={panelBoxes} />
          <p className="m-0 text-center text-[10px] text-black/45">
            滚轮缩放 · 拖动平移 · 双击复位
          </p>
        </div>
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
          {midHoleStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-[12px] text-[#cf1322]">
              生成失败：{midHoleError}
            </div>
          ) : null}
          <span className="px-2 text-[12px] text-black/45">
            {canRun
              ? '点击右上角"生成"：第 9 步结果图减去已提取的 assets/bar/button 区域'
              : '需要第 9 步完成，且 10~12 步至少提取过一层'}
          </span>
        </div>
      )}
    </Card>
  )
}
