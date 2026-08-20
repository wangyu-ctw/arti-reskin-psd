import UploadPanel from './components/UploadPanel'
import ConfigPanel from './components/ConfigPanel'
import ResultPanel from './components/ResultPanel'
import BboxViewer from './components/BboxViewer'
import TextBackPanel from './components/TextBackPanel'
import IconAnalysisPanel from './components/IconAnalysisPanel'
import TextFrontPanel from './components/TextFrontPanel'
import IconExtractPanel from './components/IconExtractPanel'
import IconAssetPanel from './components/IconAssetPanel'
import PanelFExtractPanel from './components/PanelFExtractPanel'
import IconBackPanel from './components/IconBackPanel'
import MidExtractPanel from './components/MidExtractPanel'
import BarDecomposePanel from './components/BarDecomposePanel'
import MidHolePanel from './components/MidHolePanel'
import MidFillPanel from './components/MidFillPanel'
import CompareSliderPanel from './components/CompareSliderPanel'
// 旧 16/17 步(panel修正 + 分层提取)暂由 Qwen 一步分层替换试验,代码保留:
// import PanelAuditPanel from './components/PanelAuditPanel'
// import PanelExtractPanel from './components/PanelExtractPanel'
import QwenLayerPanel from './components/QwenLayerPanel'
import SeedSeekPage from './pages/SeedSeekPage'
import Pipeline2Page from './pages/Pipeline2Page'
import TrainDataViewerPage, {
  panelZConfig,
  panelZPredConfig,
  sixSlotConfig,
  sixSlotPredConfig,
} from './pages/TrainDataViewer'

function App() {
  // 独立工具路由:/seedseek 找 seed 工具(vite dev 的 SPA fallback 会兜底到 index.html)
  if (window.location.pathname.startsWith('/seedseek')) {
    return <SeedSeekPage />
  }
  // /pipeline2 新管线(layered+YOLO,独立 store/config,不动旧管线)
  if (window.location.pathname.startsWith('/pipeline2')) {
    return <Pipeline2Page />
  }
  // 训练数据浏览页(读本机数据,不经 RunPod):/sixslot 六槽整图,/panelz panel 分层,
  // /sixslot-pred 六槽推理结果(前缀重叠,必须先于 /sixslot 判断)
  if (window.location.pathname.startsWith('/sixslot-pred')) {
    return <TrainDataViewerPage config={sixSlotPredConfig} />
  }
  if (window.location.pathname.startsWith('/sixslot')) {
    return <TrainDataViewerPage config={sixSlotConfig} />
  }
  if (window.location.pathname.startsWith('/panelz-pred')) {
    return <TrainDataViewerPage config={panelZPredConfig} />
  }
  if (window.location.pathname.startsWith('/panelz')) {
    return <TrainDataViewerPage config={panelZConfig} />
  }
  return (
    <main className="h-screen overflow-x-auto bg-neutral-100">
      <div className="flex h-full min-w-max gap-6 px-6 py-4">
        <div className="h-full w-110 shrink-0">
          <UploadPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <TextBackPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <ConfigPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <ResultPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <BboxViewer />
        </div>
        <div className="h-full w-110 shrink-0">
          <TextFrontPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <IconAnalysisPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <IconExtractPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <IconAssetPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <PanelFExtractPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <IconBackPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <MidExtractPanel category="assets" stepNo={11} title="提assets" />
        </div>
        <div className="h-full w-110 shrink-0">
          <MidExtractPanel category="button" stepNo={12} title="提button" />
        </div>
        <div className="h-full w-110 shrink-0">
          <MidExtractPanel category="bar" stepNo={13} title="提bar" />
        </div>
        <div className="h-full w-110 shrink-0">
          <BarDecomposePanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <MidHolePanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <MidFillPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <CompareSliderPanel />
        </div>
        {/* 旧 16/17 步保留,试验期由 Qwen 一步分层替换:
        <div className="h-full w-110 shrink-0">
          <PanelAuditPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <PanelExtractPanel />
        </div>
        */}
        <div className="h-full w-110 shrink-0">
          <QwenLayerPanel />
        </div>
      </div>
    </main>
  )
}

export default App
