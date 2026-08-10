import { useEffect, useState } from 'react'
import { Button, Card, Checkbox, Divider, Input, InputNumber, Select } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'
import stepDefaults from '../config/stepDefaults'

const { TextArea } = Input

/**
 * 第 8+ 步"素材化":先"分组"(VL 识别相同图标并命名/slug),
 * 再"生成素材"——每组 icon 出一张高清透明素材(组内共用)。
 * 素材链路在 pod 上执行:上下文裁块 + SAM2 抠图合成纯色底 → Qwen-Image-Edit
 * 双图参照重绘 → 边界泛洪去底 → icon_assets/<slug>.png + manifest + 拼回预览。
 */
/** 生成期间逐个轮询素材文件;出图后按该组每个成员的 bbox 定位到 UI 图上 */
function AssetOverlayItem({
  runId,
  slug,
  name,
  bboxes,
}: {
  runId: string
  slug: string
  name: string
  bboxes: number[][]
}) {
  const [loadedUrl, setLoadedUrl] = useState('')
  useEffect(() => {
    let stopped = false
    let timer = 0
    const tryLoad = () => {
      const url = `/api/runs/${runId}/files/icon_assets/${slug}.png?t=${Date.now()}`
      const img = new Image()
      img.onload = () => { if (!stopped) setLoadedUrl(url) }
      img.onerror = () => { if (!stopped) timer = window.setTimeout(tryLoad, 2500) }
      img.src = url
    }
    tryLoad()
    return () => { stopped = true; window.clearTimeout(timer) }
  }, [runId, slug])
  if (!loadedUrl) return null
  return (
    <>
      {bboxes.map((b, mi) => (
        <img
          key={mi}
          src={loadedUrl}
          alt={slug}
          title={`${name}（${slug} #${mi}）`}
          className="absolute"
          style={{
            left: `${(b[0] - b[2] / 2) * 100}%`,
            top: `${(b[1] - b[3] / 2) * 100}%`,
            width: `${b[2] * 100}%`,
            height: `${b[3] * 100}%`,
            objectFit: 'contain',
          }}
        />
      ))}
    </>
  )
}

function GroupConfigSection() {
  const iconGroupModel = useDetectionStore((s) => s.iconGroupModel)
  const iconGroupTemperature = useDetectionStore((s) => s.iconGroupTemperature)
  const iconGroupSystemPrompt = useDetectionStore((s) => s.iconGroupSystemPrompt)
  const iconGroupUserPrompt = useDetectionStore((s) => s.iconGroupUserPrompt)
  const iconGroupStatus = useDetectionStore((s) => s.iconGroupStatus)
  const setField = useDetectionStore((s) => s.setField)
  const disabled = iconGroupStatus === 'running'

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-[13px] font-bold">分组模型</div>
          <Select
            className="w-full"
            value={iconGroupModel}
            onChange={(v) => setField('iconGroupModel', v)}
            disabled={disabled}
            options={stepDefaults.iconGroups.models.map((id: string) => ({
              value: id,
              label: id.split('/').pop(),
            }))}
          />
        </div>
        <div>
          <div className="text-[13px] font-bold">temperature</div>
          <InputNumber
            className="w-full"
            value={iconGroupTemperature}
            onChange={(v) => setField('iconGroupTemperature', v ?? 0)}
            disabled={disabled}
            step={0.1}
            min={0}
            max={2}
          />
        </div>
      </div>
      <div>
        <div className="text-[13px] font-bold">分组系统提示词</div>
        <TextArea
          value={iconGroupSystemPrompt}
          onChange={(e) => setField('iconGroupSystemPrompt', e.target.value)}
          disabled={disabled}
          rows={4}
        />
      </div>
      <div>
        <div className="text-[13px] font-bold">分组用户提示词</div>
        <TextArea
          value={iconGroupUserPrompt}
          onChange={(e) => setField('iconGroupUserPrompt', e.target.value)}
          disabled={disabled}
          rows={2}
        />
      </div>
    </div>
  )
}

export default function IconAssetPanel() {
  const {
    runInfo,
    textBackStatus,
    textBackImageUrl,
    iconGroupStatus,
    iconGroupError,
    iconGroups,
    iconStatus,
    iconAssetStatus,
    iconAssetError,
    iconAssetSummary,
    iconAssetItems,
    iconAssetSourceSize,
    iconAssetUseRef,
    runAnalyzeGroups,
    runIconAsset,
  } = useDetectionStore()

  const groupCount = iconGroups?.length ?? 0
  const canGroup = Boolean(runInfo) && textBackStatus === 'done'
  const canRun = Boolean(runInfo) && groupCount > 0 && iconStatus === 'done'

  return (
    <Card
      title={
        <div className="flex items-center justify-between">
          <span className="text-[15px] font-bold">
            <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
              8+
            </span>
            素材化
          </span>
          <div className="flex items-center gap-2">
            <Button
              size="small"
              loading={iconGroupStatus === 'running'}
              disabled={!canGroup}
              onClick={() => void runAnalyzeGroups()}
            >
              {iconGroupStatus === 'done' ? '重新分组' : '分组'}
            </Button>
            {iconAssetStatus === 'done' ? (
              <Button size="small" disabled={!canRun} onClick={() => void runIconAsset()}>
                重新生成
              </Button>
            ) : null}
          </div>
        </div>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      <GroupConfigSection />

      {iconGroupStatus === 'error' ? (
        <div className="mt-2 break-all text-[12px] text-[#cf1322]">
          分组失败：{iconGroupError}
        </div>
      ) : null}

      {iconGroups?.length ? (
        <div className="mt-3 flex flex-col gap-1">
          <span className="text-[13px] font-bold">分组结果（{iconGroups.length} 组）</span>
          {iconGroups.map((group, gi) => (
            <div key={gi} className="rounded border border-neutral-200 px-3 py-1.5">
              <div className="flex items-center justify-between gap-2">
                <span className="min-w-0 truncate text-[13px]" title={group.name}>
                  {group.name}
                  <span className="ml-1 text-[11px] text-black/40">{group.slug}</span>
                </span>
                <span className="shrink-0 text-[11px] text-black/45">
                  ×{group.bbox.length}（#{group.indices.join(' #')}）
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : null}

      <Divider className="my-4" />

      {iconAssetStatus === 'running' ? (
        <div className="flex flex-col gap-2 py-2">
          <span className="text-center text-[12px] text-black/45">
            正在生成素材…出一张落位一张(直裁通道,Qwen 重绘已停用)
          </span>
          <div className="relative w-full overflow-hidden rounded border border-neutral-200">
            <img
              src={textBackImageUrl}
              alt="定位底图"
              className="w-full opacity-20"
            />
            {(iconGroups ?? []).map((g) => (
              <AssetOverlayItem
                key={g.slug}
                runId={runInfo?.run_id ?? ''}
                slug={g.slug}
                name={g.name}
                bboxes={g.bbox}
              />
            ))}
          </div>
        </div>
      ) : iconAssetStatus === 'done' ? (
        <div className="flex flex-col gap-3">
          <div className="text-[13px]">
            <span className="font-bold">完成:</span> {iconAssetSummary}
          </div>
          {iconAssetItems.length && iconAssetSourceSize ? (
            <div>
              <div className="mb-1 text-[12px] text-black/45">
                叠放预览（各素材按回贴矩形绝对定位,非拼合图片）
              </div>
              <div
                className="relative w-full overflow-hidden rounded border border-neutral-200"
                style={{
                  aspectRatio: `${iconAssetSourceSize[0]} / ${iconAssetSourceSize[1]}`,
                  background:
                    'conic-gradient(#eee 25%, #fff 0 50%, #eee 0 75%, #fff 0) 0 0 / 16px 16px',
                }}
              >
                {iconAssetItems.flatMap((it) =>
                  it.members.map((m) => (
                    <img
                      key={`${it.slug}-${m.member}`}
                      src={`/api/runs/${runInfo?.run_id}/files/icon_assets/${it.file}`}
                      alt={it.slug}
                      title={`${it.slug} #${m.member}`}
                      className="absolute"
                      style={{
                        left: `${(m.paste_x / iconAssetSourceSize[0]) * 100}%`,
                        top: `${(m.paste_y / iconAssetSourceSize[1]) * 100}%`,
                        width: `${(m.paste_w / iconAssetSourceSize[0]) * 100}%`,
                        height: `${(m.paste_h / iconAssetSourceSize[1]) * 100}%`,
                      }}
                    />
                  )),
                )}
              </div>
            </div>
          ) : null}
          <div className="text-[11px] leading-relaxed text-black/45">
            素材在 run 目录 icon_assets/ 下按 slug 命名;manifest.csv 记录每个成员的
            回贴矩形(叠放预览即按它定位);_debug/ 里是去底前的重绘原图。
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 py-4">
          {iconAssetStatus === 'error' ? (
            <div className="max-w-full break-all px-2 text-center text-[12px] text-[#cf1322]">
              素材化失败：{iconAssetError}
            </div>
          ) : null}
          <Checkbox checked={iconAssetUseRef} disabled>
            <span className="text-[12px]">
              带原图参照（Qwen 重绘通道停用中,当前直接从抠图层裁块落库）
            </span>
          </Checkbox>
          <Button type="primary" disabled={!canRun} onClick={() => void runIconAsset()}>
            {iconAssetStatus === 'error' ? '重试' : '生成素材'}
          </Button>
          <span className="px-2 text-center text-[12px] text-black/45">
            {!runInfo
              ? '请先上传图片'
              : groupCount === 0
                ? '请先点右上角「分组」（VL 识别相同图标并命名）'
                : iconStatus !== 'done'
                  ? '请先完成第 8 步提icon（需要 icons.png 抠图层）'
                  : `将为 ${groupCount} 组 icon 各生成一张高清透明素材`}
          </span>
        </div>
      )}
    </Card>
  )
}
