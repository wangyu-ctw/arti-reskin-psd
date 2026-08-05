import UploadPanel from './components/UploadPanel'
import ConfigPanel from './components/ConfigPanel'
import ResultPanel from './components/ResultPanel'
import BboxViewer from './components/BboxViewer'
import TextBackPanel from './components/TextBackPanel'
import IconAnalysisPanel from './components/IconAnalysisPanel'
import IconExtractPanel from './components/IconExtractPanel'
import IconBackPanel from './components/IconBackPanel'
import MidExtractPanel from './components/MidExtractPanel'
import MidHolePanel from './components/MidHolePanel'
import MidFillPanel from './components/MidFillPanel'

function App() {
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
          <IconAnalysisPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <IconExtractPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <IconBackPanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <MidExtractPanel category="assets" stepNo={9} title="提assets" />
        </div>
        <div className="h-full w-110 shrink-0">
          <MidExtractPanel category="bar" stepNo={10} title="提bar" />
        </div>
        <div className="h-full w-110 shrink-0">
          <MidExtractPanel category="button" stepNo={11} title="提button" />
        </div>
        <div className="h-full w-110 shrink-0">
          <MidHolePanel />
        </div>
        <div className="h-full w-110 shrink-0">
          <MidFillPanel />
        </div>
      </div>
    </main>
  )
}

export default App
