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
import PanelExtractPanel from './components/PanelExtractPanel'
import SeedSeekPage from './pages/SeedSeekPage'

function App() {
  // 独立工具路由:/seedseek 找 seed 工具(vite dev 的 SPA fallback 会兜底到 index.html)
  if (window.location.pathname.startsWith('/seedseek')) {
    return <SeedSeekPage />
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
        <div className="h-full w-110 shrink-0">
          <PanelExtractPanel />
        </div>
      </div>
    </main>
  )
}

export default App
