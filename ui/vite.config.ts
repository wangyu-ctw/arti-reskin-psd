import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import http from 'node:http'
import https from 'node:https'
import type { IncomingMessage, ServerResponse } from 'node:http'

// RunPod 每次启动地址不同,代理 target 支持在页面上动态修改:
//   GET  /__runpod-target          读当前 target
//   POST /__runpod-target {target} 改 target(持久化到 ui/.runpod-target,重启 dev server 不丢)
//   /api/*                         转发到当前 target(去掉 /api 前缀)
// 持久化文件放 node_modules/.cache:放项目根目录会被 vite/tailwind 的 watcher
// 扫到,每次写入都触发页面 full-reload,和页面加载时的同步请求形成无限刷新循环
const targetFile = path.resolve(__dirname, 'node_modules/.cache/runpod-target.txt')
const legacyTargetFile = path.resolve(__dirname, '.runpod-target')
let runpodTarget = 'https://your-pod-id-8888.proxy.runpod.net' // 占位,首次使用请在页面设置里填当天地址
try {
  runpodTarget = fs.readFileSync(targetFile, 'utf-8').trim() || runpodTarget
} catch {
  // 兼容旧位置的文件,读完就删,避免它留在根目录继续触发 reload
  try {
    runpodTarget = fs.readFileSync(legacyTargetFile, 'utf-8').trim() || runpodTarget
  } catch {
    /* 都没有,用默认值 */
  }
}
try {
  fs.rmSync(legacyTargetFile, { force: true })
} catch {
  /* 忽略 */
}

function persistTarget(t: string): void {
  try {
    if (fs.existsSync(targetFile) && fs.readFileSync(targetFile, 'utf-8').trim() === t) {
      return // 内容没变不落盘,不惊动任何 watcher
    }
    fs.mkdirSync(path.dirname(targetFile), { recursive: true })
    fs.writeFileSync(targetFile, t)
  } catch {
    /* 持久化失败不影响本次代理生效 */
  }
}

function runpodProxyPlugin(): Plugin {
  return {
    name: 'runpod-dynamic-proxy',
    configureServer(server) {
      server.middlewares.use('/__runpod-target', (req: IncomingMessage, res: ServerResponse) => {
        res.setHeader('Content-Type', 'application/json')
        if (req.method === 'GET') {
          res.end(JSON.stringify({ target: runpodTarget }))
          return
        }
        if (req.method === 'POST') {
          let body = ''
          req.on('data', (c) => (body += c))
          req.on('end', () => {
            try {
              const t = String(JSON.parse(body).target ?? '').trim().replace(/\/+$/, '')
              if (!/^https?:\/\//.test(t)) throw new Error('target 必须以 http:// 或 https:// 开头')
              new URL(t) // 校验是合法 URL
              runpodTarget = t
              persistTarget(t)
              res.end(JSON.stringify({ target: runpodTarget }))
            } catch (e) {
              res.statusCode = 400
              res.end(JSON.stringify({ error: e instanceof Error ? e.message : String(e) }))
            }
          })
          return
        }
        res.statusCode = 405
        res.end(JSON.stringify({ error: 'method not allowed' }))
      })

      // connect 挂载 '/api' 后 req.url 已去掉前缀,直接拼到 target 后面
      server.middlewares.use('/api', (req: IncomingMessage, res: ServerResponse) => {
        const t = new URL(runpodTarget)
        const isHttps = t.protocol === 'https:'
        const proxyReq = (isHttps ? https : http).request(
          {
            hostname: t.hostname,
            port: t.port || (isHttps ? 443 : 80),
            path: req.url,
            method: req.method,
            headers: { ...req.headers, host: t.host },
          },
          (proxyRes) => {
            res.writeHead(proxyRes.statusCode ?? 502, proxyRes.headers)
            proxyRes.pipe(res)
          },
        )
        proxyReq.on('error', (e) => {
          res.statusCode = 502
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ error: `代理到 ${runpodTarget} 失败: ${e.message}` }))
        })
        req.pipe(proxyReq)
      })
    },
  }
}

// 训练数据浏览(/sixslot、/panelz 页):dev server 直接读本机 ~/Desktop/训练数据,
// 不经 RunPod。TRAIN_DATA_DIR 环境变量可换根目录。
//   GET /__train-data/<数据集>/list                 样本 stem 清单(train/ 下有 meta.json 的目录)
//   GET /__train-data/<数据集>/train/<stem>/<file>  静态文件(png/json)
const trainDataRoot =
  process.env.TRAIN_DATA_DIR ?? path.join(os.homedir(), 'Desktop/训练数据')

function trainDataPlugin(): Plugin {
  const rootNorm = path.normalize(trainDataRoot)
  return {
    name: 'train-data-local',
    configureServer(server) {
      server.middlewares.use('/__train-data', (req: IncomingMessage, res: ServerResponse) => {
        res.setHeader('Content-Type', 'application/json')
        const url = decodeURIComponent((req.url ?? '/').split('?')[0])
        const listMatch = url.match(/^\/([^/]+)\/list$/)
        if (listMatch) {
          try {
            const dsDir = path.normalize(path.join(trainDataRoot, listMatch[1]))
            if (!dsDir.startsWith(rootNorm + path.sep)) throw new Error('非法数据集名')
            // val 在前:验收时最先看
            const samples: { stem: string; split: string }[] = []
            for (const split of ['val', 'train']) {
              const dir = path.join(dsDir, split)
              if (!fs.existsSync(dir)) continue
              for (const d of fs.readdirSync(dir).sort()) {
                if (fs.existsSync(path.join(dir, d, 'meta.json'))) {
                  samples.push({ stem: d, split })
                }
              }
            }
            res.end(JSON.stringify({ samples }))
          } catch (e) {
            res.statusCode = 500
            res.end(JSON.stringify({ error: e instanceof Error ? e.message : String(e) }))
          }
          return
        }
        const fp = path.normalize(path.join(trainDataRoot, url))
        if (!fp.startsWith(rootNorm + path.sep)) {
          res.statusCode = 403
          res.end(JSON.stringify({ error: 'forbidden' }))
          return
        }
        if (!fs.existsSync(fp) || !fs.statSync(fp).isFile()) {
          res.statusCode = 404
          res.end(JSON.stringify({ error: 'not found' }))
          return
        }
        res.setHeader(
          'Content-Type',
          path.extname(fp) === '.png' ? 'image/png' : 'application/json',
        )
        fs.createReadStream(fp).pipe(res)
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), runpodProxyPlugin(), trainDataPlugin()],
})
