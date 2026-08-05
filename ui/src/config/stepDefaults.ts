// 各步骤默认值的统一出口。
// 长提示词单独放文本文件里维护(可读、免转义),这里负责读取和拼装;
// 其余默认值仍在 stepDefaults.json。消费方一律从本模块导入,不要直接 import json。
import raw from './stepDefaults.json'
import detectionSystemPrompt from './detectionSystemPrompt.md?raw'

const stepDefaults = {
  ...raw,
  detection: {
    ...raw.detection,
    systemPrompt: detectionSystemPrompt.trim(),
  },
}

export default stepDefaults
