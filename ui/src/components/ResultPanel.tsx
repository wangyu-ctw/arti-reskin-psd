import { App, Button, Card, Spin } from 'antd'
import { useDetectionStore } from '../stores/useDetectionStore'

export default function ResultPanel() {
  const { message } = App.useApp()
  const {
    outputText,
    outputIsError,
    statusType,
    usageModel,
    usageTokens,
    usageVisible,
    importCachedResult,
  } = useDetectionStore()

  const handleImport = () => {
    if (importCachedResult()) {
      message.success('已导入上次生成的结果')
    } else {
      message.warning('没有可导入的缓存结果')
    }
  }

  const body = () => {
    if (statusType === 'working') {
      return (
        <div className="flex h-full flex-col items-center justify-center gap-3">
          <Spin />
          <span className="text-[12px] text-black/45">正在生成结构化检测结果…</span>
        </div>
      )
    }
    if (outputText && outputIsError) {
      return (
        <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
          <strong className="text-sm text-[#cf1322]">请求失败</strong>
          <pre className="m-0 max-w-full font-mono text-[12px] leading-relaxed break-words whitespace-pre-wrap text-[#cf1322]">
            {outputText}
          </pre>
        </div>
      )
    }
    if (outputText) {
      return (
        <pre className="m-0 font-mono text-[13px] leading-relaxed break-words whitespace-pre-wrap">
          {outputText}
        </pre>
      )
    }
    return (
      <div className="flex h-full flex-col items-center justify-center text-center text-black/45">
        <div className="mb-4 size-[74px] rounded-full border border-neutral-200" />
        <strong className="text-sm text-black/65">尚无检测结果</strong>
        <span className="mt-2 max-w-[280px] text-xs leading-relaxed">
          在左侧配置请求并上传图片，发送后这里会展示结构化 JSON
          结果；也可以点击右上角“导入缓存”复用上次的结果。
        </span>
      </div>
    )
  }

  return (
    <Card
      title={
        <span className="text-[15px] font-bold">
          <span className="mr-2 inline-grid size-[26px] place-items-center rounded-full bg-[#e6f4ff] text-xs font-extrabold text-[#1677ff]">
            4
          </span>
          结构化检测结果
        </span>
      }
      extra={
        <Button size="small" onClick={handleImport}>
          导入缓存
        </Button>
      }
      className="flex h-full w-full flex-col shadow-sm"
      styles={{ body: { overflow: 'auto', flex: 1 } }}
    >
      {body()}

      {usageVisible && outputText && !outputIsError ? (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-neutral-200 pt-3 text-[10px] text-black/45">
          <span>{usageModel}</span>
          <span>{usageTokens}</span>
        </div>
      ) : null}
    </Card>
  )
}
