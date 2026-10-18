<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { AgentApi, ApiError, type ApiSettings } from '@/services/api'
import type {
  AgentRouteResult,
  BatchJudgeResult,
  ChatResult,
  IntentResult,
  JudgeResult,
  MemoryResult,
  RagRetrieveResult,
} from '@/types/api'

type Page = 'chat' | 'tools' | 'evaluation'
type Tool = 'intent' | 'retrieve' | 'router' | 'memory'
type ChatMessage = {
  id: number
  role: 'user' | 'assistant'
  content: string
  time: string
  result?: ChatResult
}

const STORAGE_KEY = 'smart-customer-agent-settings'
const savedSettings = localStorage.getItem(STORAGE_KEY)
let parsedSettings: Partial<ApiSettings> = {}
if (savedSettings) {
  try {
    parsedSettings = JSON.parse(savedSettings) as Partial<ApiSettings>
  } catch {
    parsedSettings = {}
  }
}

const settings = ref<ApiSettings>({
  baseUrl:
    parsedSettings.baseUrl ??
    import.meta.env.VITE_API_BASE_URL ??
    (import.meta.env.DEV ? 'http://127.0.0.1:8000/api/v1' : ''),
  token: parsedSettings.token ?? '',
  userId: parsedSettings.userId ?? 'u-001',
  tenantId: parsedSettings.tenantId ?? 'default',
})
const draftSettings = ref<ApiSettings>({ ...settings.value })
const page = ref<Page>('chat')
const selectedTool = ref<Tool>('intent')
const settingsOpen = ref(false)
const loading = ref(false)
const errorMessage = ref('')
const lastRequestId = ref('')
const query = ref('')
const scrollContainer = ref<HTMLElement>()
const sessionId = ref(crypto.randomUUID())
const messages = ref<ChatMessage[]>([
  {
    id: 1,
    role: 'assistant',
    content: '你好，我是智能客服 Agent。你可以向我咨询业务流程、订单信息或服务规则。',
    time: '现在',
  },
])

const toolQuery = ref('')
const routeIntent = ref('订单咨询')
const routeEntity = ref('{"orderNo":"OD20261016001"}')
const routeQuery = ref('我的订单什么时候发货？')
const memorySessionId = ref(sessionId.value)
const toolResult = ref<IntentResult | RagRetrieveResult | AgentRouteResult | MemoryResult>()
const evaluationQuery = ref('')
const evaluationAnswer = ref('')
const evaluationReference = ref('')
const batchJson = ref('[\n  {\n    "sampleId": "case-001",\n    "query": "订单什么时候发货？",\n    "answer": "通常会在下单后24小时内发货。",\n    "reference": "订单下单后24小时内发货。"\n  }\n]')
const judgeResult = ref<JudgeResult>()
const batchResult = ref<BatchJudgeResult>()

const api = computed(() => new AgentApi(settings.value))
const pageTitle = computed(() => {
  if (page.value === 'chat') return '智能客服工作台'
  if (page.value === 'tools') return '能力调试'
  return '质量评测'
})
const currentTime = new Intl.DateTimeFormat('zh-CN', {
  month: 'long',
  day: 'numeric',
  weekday: 'long',
})
const dateLabel = currentTime.format(new Date())
const connectionLabel = computed(() => (settings.value.token ? '已配置鉴权' : '待配置 Token'))

function clearError() {
  errorMessage.value = ''
  lastRequestId.value = ''
}

function showError(error: unknown) {
  if (error instanceof ApiError) {
    errorMessage.value = error.message
    lastRequestId.value = error.requestId ?? ''
    return
  }
  errorMessage.value = error instanceof Error ? error.message : '操作失败，请稍后重试。'
}

function saveSettings() {
  settings.value = { ...draftSettings.value }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings.value))
  settingsOpen.value = false
  clearError()
}

function createNewSession() {
  sessionId.value = crypto.randomUUID()
  memorySessionId.value = sessionId.value
  messages.value = [
    {
      id: Date.now(),
      role: 'assistant',
      content: '新会话已开始。你好，有什么可以帮你？',
      time: '现在',
    },
  ]
  clearError()
}

async function scrollToLatest() {
  await nextTick()
  if (scrollContainer.value) {
    scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight
  }
}

watch(messages, scrollToLatest, { deep: true })

async function sendMessage() {
  const content = query.value.trim()
  if (!content || loading.value) return
  clearError()
  messages.value.push({
    id: Date.now(),
    role: 'user',
    content,
    time: new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit' }).format(
      new Date(),
    ),
  })
  query.value = ''
  loading.value = true
  try {
    const result = await api.value.chat({
      sessionId: sessionId.value,
      query: content,
      timeout: 30000,
    })
    messages.value.push({
      id: Date.now() + 1,
      role: 'assistant',
      content: result.answer,
      time: new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit' }).format(
        new Date(),
      ),
      result,
    })
  } catch (error) {
    showError(error)
  } finally {
    loading.value = false
  }
}

function onComposerKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void sendMessage()
  }
}

async function runTool() {
  if (loading.value) return
  clearError()
  toolResult.value = undefined
  loading.value = true
  try {
    if (selectedTool.value === 'intent') {
      toolResult.value = await api.value.classifyIntent({
        query: toolQuery.value.trim(),
        sessionId: memorySessionId.value || undefined,
      })
    } else if (selectedTool.value === 'retrieve') {
      toolResult.value = await api.value.retrieve({
        query: toolQuery.value.trim(),
        topK: 3,
        enableQueryRewrite: true,
      })
    } else if (selectedTool.value === 'router') {
      let entity: Record<string, unknown>
      try {
        entity = JSON.parse(routeEntity.value) as Record<string, unknown>
      } catch {
        throw new Error('实体内容必须是有效的 JSON 对象。')
      }
      toolResult.value = await api.value.route({
        intentLabel: routeIntent.value.trim(),
        entity,
        query: routeQuery.value.trim(),
      })
    } else {
      toolResult.value = await api.value.getMemory(memorySessionId.value.trim())
    }
  } catch (error) {
    showError(error)
  } finally {
    loading.value = false
  }
}

async function runJudge() {
  if (loading.value) return
  clearError()
  judgeResult.value = undefined
  batchResult.value = undefined
  loading.value = true
  try {
    judgeResult.value = await api.value.judge({
      query: evaluationQuery.value.trim(),
      answer: evaluationAnswer.value.trim(),
      reference: evaluationReference.value.trim() || undefined,
    })
  } catch (error) {
    showError(error)
  } finally {
    loading.value = false
  }
}

async function runBatchJudge() {
  if (loading.value) return
  clearError()
  judgeResult.value = undefined
  batchResult.value = undefined
  let samples
  try {
    const parsed: unknown = JSON.parse(batchJson.value)
    if (
      !Array.isArray(parsed) ||
      parsed.length < 1 ||
      parsed.length > 100 ||
      !parsed.every(
        (sample: unknown) =>
          typeof sample === 'object' &&
          sample !== null &&
          'query' in sample &&
          'answer' in sample,
      )
    ) {
      throw new Error('请提供 1 至 100 条包含 query 与 answer 的 JSON 样本。')
    }
    samples = parsed as Array<{
      sampleId?: string
      query: string
      answer: string
      reference?: string
    }>
  } catch (error) {
    showError(error)
    return
  }
  loading.value = true
  try {
    batchResult.value = await api.value.judgeBatch({ samples })
  } catch (error) {
    showError(error)
  } finally {
    loading.value = false
  }
}

async function copyText(value: string) {
  try {
    await navigator.clipboard.writeText(value)
  } catch {
    errorMessage.value = '复制失败，请检查浏览器剪贴板权限。'
  }
}

onMounted(() => {
  void scrollToLatest()
})
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <a class="brand" href="#" aria-label="智能客服 Agent 首页" @click.prevent="page = 'chat'">
        <span class="brand-mark">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <path d="M8 8.5h16a3.5 3.5 0 0 1 3.5 3.5v8a3.5 3.5 0 0 1-3.5 3.5h-7l-5.5 3v-3H8A3.5 3.5 0 0 1 4.5 20v-8A3.5 3.5 0 0 1 8 8.5Z" />
            <circle cx="11" cy="16" r="1.25" />
            <circle cx="16" cy="16" r="1.25" />
            <circle cx="21" cy="16" r="1.25" />
          </svg>
        </span>
        <span class="brand-copy">
          <strong>知答</strong>
          <span>AGENT PLATFORM</span>
        </span>
      </a>

      <div class="workspace-label">工作空间</div>
      <nav class="main-nav" aria-label="主导航">
        <button :class="{ active: page === 'chat' }" @click="page = 'chat'">
          <span class="nav-icon">◉</span><span>智能客服</span><span v-if="page === 'chat'" class="nav-current" />
        </button>
        <button :class="{ active: page === 'tools' }" @click="page = 'tools'">
          <span class="nav-icon">⌘</span><span>能力调试</span>
        </button>
        <button :class="{ active: page === 'evaluation' }" @click="page = 'evaluation'">
          <span class="nav-icon">▥</span><span>质量评测</span>
        </button>
      </nav>

      <div class="sidebar-divider" />
      <div class="workspace-label">当前会话</div>
      <button class="session-link" @click="createNewSession">
        <span class="session-dot" />
        <span class="session-link-copy"><strong>客服咨询</strong><small>刚刚 · Web 渠道</small></span>
        <span class="session-arrow">↗</span>
      </button>

      <div class="sidebar-bottom">
        <div class="help-card">
          <div class="help-orb">✳</div>
          <strong>开始使用</strong>
          <p>配置 API 地址与访问令牌，连接你的智能客服 Agent。</p>
          <button @click="draftSettings = { ...settings }; settingsOpen = true">
            配置连接 <span>→</span>
          </button>
        </div>
        <button class="profile-button" @click="draftSettings = { ...settings }; settingsOpen = true">
          <span class="avatar">知</span>
          <span class="profile-copy"><strong>{{ settings.userId || 'u-001' }}</strong><small>租户 {{ settings.tenantId }}</small></span>
          <span class="profile-menu">•••</span>
        </button>
      </div>
    </aside>

    <main class="main-panel">
      <header class="topbar">
        <div class="breadcrumbs"><span>Agent 工作空间</span><span class="crumb-slash">/</span><strong>{{ pageTitle }}</strong></div>
        <div class="topbar-actions">
          <span class="today-label">{{ dateLabel }}</span>
          <span class="topbar-divider" />
          <button class="connection-status" @click="draftSettings = { ...settings }; settingsOpen = true">
            <span class="status-light" :class="{ configured: settings.token }" />
            {{ connectionLabel }}
            <span class="chevron-down">⌄</span>
          </button>
          <button class="icon-button" title="接口文档" aria-label="查看接口文档" @click="page = 'tools'">⌘</button>
          <button class="top-avatar" title="连接设置" @click="draftSettings = { ...settings }; settingsOpen = true">知</button>
        </div>
      </header>

      <section v-if="page === 'chat'" class="page-content chat-page">
        <div class="page-heading chat-heading">
          <div>
            <div class="eyebrow"><span class="eyebrow-line" /> CUSTOMER SUPPORT · AI AGENT</div>
            <h1>让每个问题，都有<span>温度</span>。</h1>
            <p class="heading-subtitle">你好，{{ settings.userId }}。今天想为用户解决什么问题？</p>
          </div>
          <div class="heading-actions">
            <button class="button button-quiet" @click="createNewSession"><span class="plus-mark">＋</span> 新建会话</button>
            <button class="button button-primary" @click="draftSettings = { ...settings }; settingsOpen = true"><span class="button-spark">✳</span> 连接设置</button>
          </div>
        </div>

        <div class="chat-layout">
          <section class="conversation-card">
            <div class="conversation-topline">
              <div class="conversation-title">
                <span class="conversation-icon">✳</span>
                <div><strong>智能客服助手</strong><small><i /> Agent 在线 · {{ settings.tenantId }}</small></div>
              </div>
              <div class="conversation-tools">
                <span class="session-id">会话 {{ sessionId.slice(0, 8) }}</span>
                <button class="mini-icon" title="新建会话" @click="createNewSession">↗</button>
                <button class="mini-icon" title="查看会话记忆" @click="selectedTool = 'memory'; page = 'tools'; memorySessionId = sessionId">◷</button>
              </div>
            </div>

            <div ref="scrollContainer" class="message-list">
              <div class="day-separator"><span />今天<span /></div>
              <article
                v-for="message in messages"
                :key="message.id"
                class="message-row"
                :class="message.role"
              >
                <div v-if="message.role === 'assistant'" class="message-avatar assistant-avatar">✳</div>
                <div class="message-body">
                  <div class="message-meta">
                    <strong>{{ message.role === 'assistant' ? '知答助手' : '我' }}</strong>
                    <time>{{ message.time }}</time>
                  </div>
                  <div class="message-bubble" :class="{ 'user-bubble': message.role === 'user' }">
                    <p>{{ message.content }}</p>
                    <div v-if="message.result" class="answer-details">
                      <div class="answer-meta">
                        <span class="agent-chip"><i />{{ message.result.agentName }}</span>
                        <span class="intent-chip">{{ message.result.intentInfo.intentLabel }} · {{ Math.round(message.result.intentInfo.confidence * 100) }}%</span>
                        <span v-if="message.result.transferHuman" class="handoff-chip">建议转人工</span>
                      </div>
                      <details v-if="message.result.referenceDocs.length" class="reference-details">
                        <summary>参考资料 <span>{{ message.result.referenceDocs.length }}</span><b>⌄</b></summary>
                        <div v-for="document in message.result.referenceDocs" :key="document.docId" class="reference-item">
                          <strong>{{ document.docId }}</strong>
                          <p>{{ document.content }}</p>
                        </div>
                      </details>
                    </div>
                  </div>
                </div>
                <div v-if="message.role === 'user'" class="message-avatar user-avatar">知</div>
              </article>
              <div v-if="loading && page === 'chat'" class="message-row assistant">
                <div class="message-avatar assistant-avatar">✳</div>
                <div class="message-body">
                  <div class="message-meta"><strong>知答助手</strong><time>正在思考</time></div>
                  <div class="message-bubble typing-bubble"><span /><span /><span /></div>
                </div>
              </div>
            </div>

            <div v-if="errorMessage" class="inline-error" role="alert">
              <span>!</span><div><strong>请求没有完成</strong><p>{{ errorMessage }}<small v-if="lastRequestId"> 请求编号：{{ lastRequestId }}</small></p></div>
              <button aria-label="关闭错误提示" @click="clearError">×</button>
            </div>

            <div class="composer-wrap">
              <div class="composer-context"><span class="context-mark">✳</span> Agent 已启用<span class="context-dot">·</span> RAG 知识检索<span class="context-dot">·</span> 会话记忆</div>
              <div class="composer">
                <textarea
                  v-model="query"
                  rows="1"
                  maxlength="10000"
                  placeholder="输入用户的问题，按 Enter 发送…"
                  aria-label="输入客服问题"
                  :disabled="loading"
                  @keydown="onComposerKeydown"
                />
                <div class="composer-bottom">
                  <div class="composer-hints"><span class="shortcut-key">↵</span> 发送<span class="hint-gap" /><span class="shortcut-key">⇧ ↵</span> 换行</div>
                  <div class="composer-send">
                    <span>{{ query.length }} / 10000</span>
                    <button class="send-button" :disabled="!query.trim() || loading" aria-label="发送消息" @click="sendMessage">
                      <span v-if="loading" class="spinner" /><span v-else>↑</span>
                    </button>
                  </div>
                </div>
              </div>
              <p class="disclaimer">AI 生成的内容仅供参考，请核对重要业务信息。</p>
            </div>
          </section>

          <aside class="insights-column">
            <section class="insight-card session-overview">
              <div class="card-label">会话概览 <button class="more-button" @click="page = 'tools'; selectedTool = 'memory'; memorySessionId = sessionId">•••</button></div>
              <div class="overview-stat"><strong>{{ messages.filter((message) => message.role === 'user').length }}</strong><span>轮对话</span></div>
              <div class="overview-foot"><span class="live-dot" /> 当前会话进行中</div>
              <div class="overview-line"><span>会话 ID</span><button @click="copyText(sessionId)">{{ sessionId.slice(0, 12) }}… <b>▢</b></button></div>
            </section>

            <section class="insight-card route-card">
              <div class="card-label">Agent 路由 <span class="tiny-label">LIVE</span></div>
              <div class="route-visual">
                <div class="route-node primary-node"><span>✳</span><div><strong>主 Agent</strong><small>意图识别 · 全局调度</small></div></div>
                <div class="route-connector"><i /><i /><i /></div>
                <div class="route-node secondary-node"><span class="route-node-dot">◉</span><div><strong>专属 Agent</strong><small>按业务场景智能分发</small></div></div>
              </div>
              <button class="text-link" @click="page = 'tools'; selectedTool = 'router'">调试路由规则 <span>↗</span></button>
            </section>

            <section class="insight-card memory-card">
              <div class="card-label">三级记忆 <span class="memory-badge">● 已启用</span></div>
              <div class="memory-row"><span class="memory-symbol work-symbol">◷</span><div><strong>工作记忆</strong><small>Redis · 最近 N 轮上下文</small></div><i class="memory-check">✓</i></div>
              <div class="memory-row"><span class="memory-symbol episode-symbol">⌁</span><div><strong>情景记忆</strong><small>Chroma · 会话摘要</small></div><i class="memory-check">✓</i></div>
              <div class="memory-row"><span class="memory-symbol profile-symbol">◎</span><div><strong>用户画像</strong><small>Chroma · 偏好与特征</small></div><i class="memory-check">✓</i></div>
              <button class="text-link" @click="page = 'tools'; selectedTool = 'memory'; memorySessionId = sessionId">查看当前记忆 <span>↗</span></button>
            </section>

            <div class="tip-card"><span>✳</span><p><strong>小贴士</strong><br />试试在问题中加入订单号，Agent 会优先将请求分发至订单专属服务。</p></div>
          </aside>
        </div>
      </section>

      <section v-else-if="page === 'tools'" class="page-content secondary-page">
        <div class="page-heading secondary-heading">
          <div><div class="eyebrow"><span class="eyebrow-line" /> AGENT TOOLKIT</div><h1>能力调试</h1><p class="heading-subtitle">单独验证智能客服各模块接口，快速定位路由、意图与检索问题。</p></div>
          <button class="button button-quiet" @click="draftSettings = { ...settings }; settingsOpen = true">⚙ 连接设置</button>
        </div>
        <div class="toolkit-layout">
          <nav class="tool-nav" aria-label="调试工具">
            <span class="tool-nav-title">模块</span>
            <button :class="{ active: selectedTool === 'intent' }" @click="selectedTool = 'intent'; toolResult = undefined"><span>◎</span>意图识别<small>/intent/classify</small></button>
            <button :class="{ active: selectedTool === 'retrieve' }" @click="selectedTool = 'retrieve'; toolResult = undefined"><span>⌕</span>知识检索<small>/rag/retrieve</small></button>
            <button :class="{ active: selectedTool === 'router' }" @click="selectedTool = 'router'; toolResult = undefined"><span>⌘</span>Agent 路由<small>/agent/router</small></button>
            <button :class="{ active: selectedTool === 'memory' }" @click="selectedTool = 'memory'; toolResult = undefined"><span>◷</span>三级记忆<small>/memory/get</small></button>
            <div class="tool-nav-note"><span>ⓘ</span>调试请求使用当前租户 <strong>{{ settings.tenantId }}</strong>。</div>
          </nav>
          <section class="tool-workspace">
            <div class="tool-workspace-head">
              <div><span class="tool-workspace-icon">{{ selectedTool === 'intent' ? '◎' : selectedTool === 'retrieve' ? '⌕' : selectedTool === 'router' ? '⌘' : '◷' }}</span><div><h2>{{ selectedTool === 'intent' ? '意图识别' : selectedTool === 'retrieve' ? '知识库检索' : selectedTool === 'router' ? 'Agent 路由' : '会话记忆' }}</h2><p>{{ selectedTool === 'intent' ? '分析用户问题，获取融合意图、置信度与确认建议。' : selectedTool === 'retrieve' ? '运行 Query 改写、混合召回与文档重排。' : selectedTool === 'router' ? '根据意图、实体与关键词测试目标 Agent。' : '读取当前用户的工作记忆、历史摘要和用户画像。' }}</p></div></div>
              <span class="method-pill">POST <code>{{ selectedTool === 'intent' ? '/intent/classify' : selectedTool === 'retrieve' ? '/rag/retrieve' : selectedTool === 'router' ? '/agent/router' : '/memory/get' }}</code></span>
            </div>

            <div v-if="selectedTool === 'intent' || selectedTool === 'retrieve'" class="tool-form">
              <label class="field-label" for="tool-query">用户问题 <span>必填</span></label>
              <textarea id="tool-query" v-model="toolQuery" class="field-control textarea-control" rows="4" maxlength="10000" placeholder="例如：我的订单什么时候发货？" />
              <div v-if="selectedTool === 'retrieve'" class="field-inline-note"><span>✳</span> 将使用当前租户知识库进行检索，并自动启用 Query 改写。</div>
            </div>
            <div v-else-if="selectedTool === 'router'" class="tool-form field-grid">
              <div><label class="field-label" for="route-intent">意图标签</label><input id="route-intent" v-model="routeIntent" class="field-control" placeholder="订单咨询" /></div>
              <div><label class="field-label" for="route-query">用户问题（用于关键词路由）</label><input id="route-query" v-model="routeQuery" class="field-control" placeholder="我的订单什么时候发货？" /></div>
              <div class="field-wide"><label class="field-label" for="route-entity">业务实体 <span>JSON</span></label><textarea id="route-entity" v-model="routeEntity" class="field-control textarea-control code-control" rows="3" /></div>
            </div>
            <div v-else class="tool-form field-grid">
              <div class="field-wide"><label class="field-label" for="memory-session">会话 ID</label><input id="memory-session" v-model="memorySessionId" class="field-control" placeholder="sess-123456" /></div>
              <div class="field-wide memory-key-note">用户 ID：<strong>{{ settings.userId }}</strong><span>·</span> 租户：<strong>{{ settings.tenantId }}</strong></div>
            </div>

            <div v-if="errorMessage" class="inline-error tool-error" role="alert"><span>!</span><div><strong>请求没有完成</strong><p>{{ errorMessage }}<small v-if="lastRequestId"> 请求编号：{{ lastRequestId }}</small></p></div><button @click="clearError">×</button></div>
            <div class="tool-submit-row"><span>请求将发送至 <strong>{{ settings.baseUrl }}</strong></span><button class="button button-primary" :disabled="loading || (selectedTool !== 'router' && selectedTool !== 'memory' && !toolQuery.trim())" @click="runTool"><span v-if="loading" class="spinner spinner-light" /><span v-else>运行调试 <b>→</b></span></button></div>

            <section v-if="toolResult" class="result-panel">
              <div class="result-heading"><div><span class="result-success">✓</span><strong>请求成功</strong><small>HTTP 200</small></div><button class="copy-button" @click="copyText(JSON.stringify(toolResult, null, 2))">▢ 复制 JSON</button></div>
              <div v-if="selectedTool === 'intent'" class="intent-result">
                <div class="intent-score"><span>意图分类</span><strong>{{ (toolResult as IntentResult).intentLabel }}</strong><small>置信度 {{ Math.round((toolResult as IntentResult).confidence * 100) }}%</small></div>
                <div class="score-bars" v-if="(toolResult as IntentResult).scoreDetail">
                  <div v-for="item in [{ label: 'LLM Few-shot', value: (toolResult as IntentResult).scoreDetail?.llmFewShotScore }, { label: 'Embedding', value: (toolResult as IntentResult).scoreDetail?.embeddingScore }, { label: '关键词规则', value: (toolResult as IntentResult).scoreDetail?.regexKeywordScore }]" :key="item.label" class="score-row"><span>{{ item.label }}</span><div><i :style="{ width: `${Math.max(0, Math.min(100, (item.value ?? 0) * 100))}%` }" /></div><b>{{ Math.round((item.value ?? 0) * 100) }}%</b></div>
                </div>
                <span v-if="(toolResult as IntentResult).needConfirm" class="notice-chip">需要追问确认</span>
              </div>
              <div v-else-if="selectedTool === 'retrieve'" class="retrieve-result">
                <div class="rewrite-row"><span>改写 Query</span><div><b v-for="item in (toolResult as RagRetrieveResult).rewriteQueries" :key="item">{{ item }}</b></div></div>
                <div class="result-doc" v-for="document in (toolResult as RagRetrieveResult).rerankDocs" :key="document.docId"><div><strong>{{ document.docId }}</strong><span v-if="document.score !== undefined">相关度 {{ Math.round(document.score * 100) }}%</span></div><p>{{ document.content }}</p></div>
                <div v-if="!(toolResult as RagRetrieveResult).rerankDocs.length" class="empty-result">没有召回文档。请确认知识库已索引当前租户的文档。</div>
                <div class="mrr-note">MRR：{{ (toolResult as RagRetrieveResult).mrr ?? '暂无标注评测数据' }}</div>
              </div>
              <div v-else-if="selectedTool === 'router'" class="route-result"><span class="route-result-icon">⌘</span><div><small>目标 Agent</small><strong>{{ (toolResult as AgentRouteResult).targetAgent }}</strong><p>{{ (toolResult as AgentRouteResult).routeReason }}</p></div><span class="fallback-pill">{{ (toolResult as AgentRouteResult).fallback ? '通用兜底' : '已匹配' }}</span></div>
              <div v-else class="memory-result">
                <div class="memory-result-block"><strong>工作记忆 <span>{{ (toolResult as MemoryResult).workingMemory.length }}</span></strong><pre>{{ JSON.stringify((toolResult as MemoryResult).workingMemory, null, 2) }}</pre></div>
                <div class="memory-result-block"><strong>情景记忆 <span>{{ (toolResult as MemoryResult).episodicMemory.length }}</span></strong><pre>{{ JSON.stringify((toolResult as MemoryResult).episodicMemory, null, 2) }}</pre></div>
                <div class="memory-result-block"><strong>用户画像</strong><pre>{{ JSON.stringify((toolResult as MemoryResult).userProfile, null, 2) }}</pre></div>
              </div>
            </section>
            <div v-else class="result-placeholder"><span>⌁</span><strong>运行一次调试</strong><p>接口响应会显示在这里，你可以检查结果并复制 JSON。</p></div>
          </section>
        </div>
      </section>

      <section v-else class="page-content secondary-page evaluation-page">
        <div class="page-heading secondary-heading">
          <div><div class="eyebrow"><span class="eyebrow-line" /> QUALITY & REGRESSION</div><h1>质量评测</h1><p class="heading-subtitle">使用 LLM-as-Judge 评估回答的相关性、准确性、完整性与有用性。</p></div>
          <span class="evaluation-badge"><span /> LLM-as-Judge</span>
        </div>
        <div class="evaluation-layout">
          <section class="evaluation-card">
            <div class="evaluation-card-head"><div><span class="eval-icon">✳</span><div><h2>单条问答评测</h2><p>提交一个问题及 Agent 回答，即时查看四维评分。</p></div></div><span class="method-pill">POST <code>/eval/llmjudge</code></span></div>
            <div class="evaluation-fields">
              <label class="field-label" for="eval-query">用户问题 <span>必填</span></label><textarea id="eval-query" v-model="evaluationQuery" class="field-control textarea-control" rows="2" placeholder="输入用户提出的问题…" />
              <label class="field-label" for="eval-answer">Agent 回答 <span>必填</span></label><textarea id="eval-answer" v-model="evaluationAnswer" class="field-control textarea-control" rows="3" placeholder="输入需要评估的客服回复…" />
              <label class="field-label" for="eval-reference">参考答案 / 知识库资料 <span>选填</span></label><textarea id="eval-reference" v-model="evaluationReference" class="field-control textarea-control" rows="2" placeholder="提供参考文本可帮助 Judge 评估回答准确性…" />
            </div>
            <div v-if="page === 'evaluation' && errorMessage" class="inline-error tool-error"><span>!</span><div><strong>请求没有完成</strong><p>{{ errorMessage }}</p></div><button @click="clearError">×</button></div>
            <div class="evaluation-submit"><span>评分范围 0—5 分 · 测评结果会被持久化</span><button class="button button-primary" :disabled="loading || !evaluationQuery.trim() || !evaluationAnswer.trim()" @click="runJudge"><span v-if="loading" class="spinner spinner-light" /><span v-else>开始评测 <b>→</b></span></button></div>
            <div v-if="judgeResult" class="judge-result">
              <div class="judge-summary"><div><span>综合评分</span><strong>{{ judgeResult.avgScore.toFixed(2) }}<small> / 5</small></strong></div><span class="quality-tag" :class="judgeResult.qualityTag === '优质' ? 'quality-good' : judgeResult.qualityTag === '一般' ? 'quality-average' : 'quality-bad'">{{ judgeResult.qualityTag }}</span></div>
              <div class="judge-dimensions">
                <div v-for="item in [{ key: 'relevance', label: '相关性' }, { key: 'accuracy', label: '准确性' }, { key: 'completeness', label: '完整性' }, { key: 'usefulness', label: '有用性' }]" :key="item.key"><span>{{ item.label }}</span><strong>{{ judgeResult.score[item.key as keyof JudgeResult['score']].toFixed(1) }}<small>/5</small></strong><div class="dimension-bar"><i :style="{ width: `${judgeResult.score[item.key as keyof JudgeResult['score']] / 5 * 100}%` }" /></div></div>
              </div>
              <p class="judge-reason"><strong>评测理由</strong>{{ judgeResult.judgeReason }}</p>
            </div>
          </section>

          <section class="evaluation-card batch-card">
            <div class="evaluation-card-head"><div><span class="eval-icon batch-icon">▤</span><div><h2>批量回归评测</h2><p>提交 JSON 样本列表，单次最多支持 100 条。</p></div></div><span class="method-pill">POST <code>/eval/batch</code></span></div>
            <div class="batch-format"><span>JSON 样本格式</span><code>{ "query": "问题", "answer": "回答", "reference": "参考答案" }</code><small>也可附带 sampleId；reference 为选填项。</small></div>
            <label class="field-label" for="batch-json">评测样本 <span>1—100 条</span></label><textarea id="batch-json" v-model="batchJson" class="field-control textarea-control code-control batch-textarea" spellcheck="false" rows="11" />
            <div class="evaluation-submit"><span>失败样本会记录错误并继续评测</span><button class="button button-primary" :disabled="loading" @click="runBatchJudge"><span v-if="loading" class="spinner spinner-light" /><span v-else>运行批量评测 <b>→</b></span></button></div>
            <div v-if="batchResult" class="batch-summary">
              <div><strong>{{ batchResult.completed }} / {{ batchResult.total }}</strong><span>完成</span></div><div><strong>{{ batchResult.failed }}</strong><span>失败</span></div><div><strong>{{ batchResult.avgScore?.toFixed(2) ?? '—' }}</strong><span>平均分</span></div>
              <div v-for="item in batchResult.results" :key="item.sampleId" class="batch-row"><span>{{ item.sampleId }}</span><strong v-if="item.avgScore !== undefined">{{ item.avgScore.toFixed(2) }} · {{ item.qualityTag }}</strong><strong v-else class="batch-failed">{{ item.error }}</strong></div>
            </div>
          </section>
        </div>
      </section>

      <footer class="app-footer"><span>知答智能客服平台</span><span>AI Agent · RAG · Memory</span><span>前端接口遵循后端 v1 API</span></footer>
    </main>

    <div v-if="settingsOpen" class="modal-backdrop" @click.self="settingsOpen = false">
      <section class="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <div class="modal-heading"><div><span class="modal-icon">⚙</span><h2 id="settings-title">连接设置</h2><p>配置客服 API 地址、JWT 鉴权及当前用户上下文。</p></div><button class="modal-close" aria-label="关闭" @click="settingsOpen = false">×</button></div>
        <label class="field-label" for="api-url">后端 API Base URL</label><input id="api-url" v-model="draftSettings.baseUrl" class="field-control" placeholder="http://127.0.0.1:8000/api/v1" /><div class="field-inline-note"><span>ⓘ</span><span>Pages 演示请填写本机后端 HTTPS 隧道地址，并以 /api/v1 结尾。</span></div>
        <label class="field-label" for="api-token">JWT Bearer Token <span>主问答必填</span></label><textarea id="api-token" v-model="draftSettings.token" class="field-control textarea-control token-control" rows="3" placeholder="粘贴由认证服务签发的 JWT（不会发送到浏览器存储以外的第三方）" />
        <div class="field-grid settings-grid"><div><label class="field-label" for="user-id">用户 ID</label><input id="user-id" v-model="draftSettings.userId" class="field-control" /></div><div><label class="field-label" for="tenant-id">租户 ID</label><input id="tenant-id" v-model="draftSettings.tenantId" class="field-control" /></div></div>
        <div class="settings-note"><span>ⓘ</span><p>JWT 的 userId、user_id 或 sub 声明必须与这里填写的用户 ID 一致。其他工具接口按当前租户范围调试。</p></div>
        <div class="modal-actions"><button class="button button-quiet" @click="settingsOpen = false">取消</button><button class="button button-primary" @click="saveSettings">保存连接</button></div>
      </section>
    </div>
  </div>
</template>

<style>
@import './assets/main.css';
</style>
