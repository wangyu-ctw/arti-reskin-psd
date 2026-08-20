import { useCallback, useRef, useState } from 'react'
import { Button, Card, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'

/** 拖动分界线的双图对比:左侧露原图,右侧露新图,拖柄居中可拖。
 *  (被 /pipeline2 第 ④ 步复用,改动需两边兼容) */
export function ImageCompareSlider({
  leftSrc,
  rightSrc,
}: {
  leftSrc: string
  rightSrc: string
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [pos, setPos] = useState(50) // 分界线位置百分比
  const draggingRef = useRef(false)

  const updateFromClientX = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return
    const pct = ((clientX - rect.left) / rect.width) * 100
    setPos(Math.min(100, Math.max(0, pct)))
  }, [])

  return (
    <div
      ref={containerRef}
      className="relative w-full cursor-ew-resize touch-none select-none overflow-hidden rounded"
      onPointerDown={(e) => {
        draggingRef.current = true
        e.currentTarget.setPointerCapture(e.pointerId)
        updateFromClientX(e.clientX)
      }}
      onPointerMove={(e) => {
        if (draggingRef.current) updateFromClientX(e.clientX)
      }}
      onPointerUp={(e) => {
        draggingRef.current = false
        if (e.currentTarget.hasPointerCapture(e.pointerId)) {
          e.currentTarget.releasePointerCapture(e.pointerId)
        }
      }}
      onPointerCancel={() => {
        draggingRef.current = false
      }}
    >
      <img src={leftSrc} alt="原图" className="block h-auto w-full" draggable={false} />
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 0 0 ${pos}%)` }}
      >
        <img
          src={rightSrc}
          alt="新图"
          className="block h-auto w-full"
          draggable={false}
        />
      </div>
      {/* 分界线与拖柄 */}
      <div
        className="pointer-events-none absolute inset-y-0 w-[2px] bg-white/90 shadow"
        style={{ left: `${pos}%` }}
      />
      <div
        className="pointer-events-none absolute top-1/2 grid size-10 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border-2 border-white/90 bg-black/30 text-[16px] text-white"
        style={{ left: `${pos}%` }}
      >
        {'<>'}
      </div>
      <span className="pointer-events-none absolute bottom-2 left-2 rounded bg-black/60 px-2 py-1 text-[13px] text-white">
        原图
      </span>
      <span className="pointer-events-none absolute bottom-2 right-2 rounded bg-black/60 px-2 py-1 text-[13px] text-white">
        新图
      </span>
    </div>
  )
}

/**
 * 第 15+ 步:前中景对比 = 第 6/8/11/12/13/15 步产物按图层序本地拼合(不上传),
 * 与原图做拖动分界线对比。
 */
export default function CompareSliderPanel() {
  const {
    runInfo,
    compareStatus,
    compareImageUrl,
    compareError,
    compareMissing,
    runCompare,
  } = useDetectionStore()

  const originUrl = runInfo
    ? `/api/runs/${runInfo.run_id}/files/origin.png`
    : ''

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              15+
            </span>
            前中景对比
          </span>
          <Button
            size="small"
            type="primary"
            ghost
            loading={compareStatus === 'running'}
            disabled={!runInfo}
            onClick={() => void runCompare()}
          >
            生成
          </Button>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {compareStatus === 'running' ? (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          <Spin />
          <span className="text-[12px] text-black/45">正在拼合图层…</span>
        </div>
      ) : compareStatus === 'done' && compareImageUrl && originUrl ? (
        <div className="flex flex-col gap-2">
          <ImageCompareSlider leftSrc={originUrl} rightSrc={compareImageUrl} />
          {compareMissing ? (
            <p className="m-0 text-center text-[11px] text-[#d46b08]">
              {compareMissing}
            </p>
          ) : null}
          <p className="m-0 text-center text-[10px] text-black/45">
            拖动分界线对比 · 左原图 / 右拼合新图
          </p>
        </div>
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
          {compareStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-[12px] text-[#cf1322]">
              拼合失败：{compareError}
            </div>
          ) : null}
          <span className="px-2 text-[12px] text-black/45">
            点击右上角"生成"：把第 6/8/10/11/12/14 步的图层拼成新图（仅存本地，不上传），
            与原图拖动对比。缺哪层就跳过哪层。
          </span>
        </div>
      )}
    </Card>
  )
}
