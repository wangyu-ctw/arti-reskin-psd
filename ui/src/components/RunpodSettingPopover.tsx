import { Input, Popover } from 'antd'
import { SettingOutlined } from '@ant-design/icons'
import { useDetectionStore } from '../stores/useDetectionStore'

/**
 * RunPod 地址设置:hover 设置 icon 弹出输入框。
 * 值存在 detection store + localStorage(和 OpenRouter API Key 同一份存储),
 * 并自动同步给 Vite dev server,/api 代理实时转发到新地址。
 */
export default function RunpodSettingPopover() {
  const { runpodTarget, setRunpodTarget } = useDetectionStore()

  return (
    <Popover
      trigger="hover"
      placement="bottomRight"
      content={
        <div className="w-[320px]">
          <div className="mb-2 text-[13px] font-bold">runpod地址</div>
          <Input
            value={runpodTarget}
            onChange={(e) => setRunpodTarget(e.target.value)}
            placeholder="https://<pod-id>-8888.proxy.runpod.net"
            autoComplete="off"
            spellCheck={false}
          />
          <div className="mt-2 text-[11px] text-black/45">
            本机浏览器自动保存；pod 重启后换了地址在这里改，立即生效
          </div>
        </div>
      }
    >
      <SettingOutlined
        className="cursor-pointer text-[16px] text-black/45 transition-colors hover:text-[#1677ff]"
        aria-label="RunPod 服务地址设置"
      />
    </Popover>
  )
}
