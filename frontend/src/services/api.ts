import type {
  AgentRouteRequest,
  AgentRouteResult,
  ApiResponse,
  BatchJudgeRequest,
  BatchJudgeResult,
  ChatRequest,
  ChatResult,
  IntentRequest,
  IntentResult,
  JudgeRequest,
  JudgeResult,
  MemoryKeyRequest,
  MemoryResult,
  MemorySaveRequest,
  MemorySaveResult,
  RagRetrieveRequest,
  RagRetrieveResult,
} from '@/types/api'

export interface ApiSettings {
  baseUrl: string
  token: string
  userId: string
  tenantId: string
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly requestId?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export class AgentApi {
  constructor(private readonly settings: ApiSettings) {}

  private async post<T>(path: string, payload: unknown, authenticated = false): Promise<T> {
    const baseUrl = this.settings.baseUrl.trim().replace(/\/+$/, '')
    if (!baseUrl) throw new ApiError('请先配置后端 API 地址。', 0)

    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (authenticated) {
      if (!this.settings.token.trim()) {
        throw new ApiError('主问答接口需要 JWT，请在右上角「连接设置」中填写 Token。', 401)
      }
      headers.Authorization = `Bearer ${this.settings.token.trim()}`
    }

    let response: Response
    try {
      response = await fetch(`${baseUrl}${path}`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      })
    } catch {
      throw new ApiError('无法连接后端服务，请检查 API 地址和网络连接。', 0)
    }

    let envelope: ApiResponse<T> | undefined
    try {
      envelope = (await response.json()) as ApiResponse<T>
    } catch {
      throw new ApiError(`后端返回了无法解析的响应（HTTP ${response.status}）。`, response.status)
    }

    if (!response.ok || envelope.code !== 200) {
      throw new ApiError(
        envelope.message || `请求失败（HTTP ${response.status}）。`,
        response.status,
        envelope.requestId,
      )
    }
    return envelope.data
  }

  chat(request: Omit<ChatRequest, 'userId' | 'tenantId'>): Promise<ChatResult> {
    return this.post(
      '/agent/chat',
      {
        ...request,
        userId: this.settings.userId,
        tenantId: this.settings.tenantId,
        channel: request.channel ?? 'web',
      } satisfies ChatRequest,
      true,
    )
  }

  classifyIntent(request: Pick<IntentRequest, 'query' | 'sessionId'>): Promise<IntentResult> {
    return this.post('/intent/classify', {
      ...request,
      tenantId: this.settings.tenantId,
    } satisfies IntentRequest)
  }

  retrieve(request: Omit<RagRetrieveRequest, 'tenantId'>): Promise<RagRetrieveResult> {
    return this.post('/rag/retrieve', {
      ...request,
      tenantId: this.settings.tenantId,
    } satisfies RagRetrieveRequest)
  }

  route(request: Omit<AgentRouteRequest, 'tenantId'>): Promise<AgentRouteResult> {
    return this.post('/agent/router', {
      ...request,
      tenantId: this.settings.tenantId,
    } satisfies AgentRouteRequest)
  }

  getMemory(sessionId: string): Promise<MemoryResult> {
    return this.post('/memory/get', {
      sessionId,
      userId: this.settings.userId,
      tenantId: this.settings.tenantId,
    } satisfies MemoryKeyRequest)
  }

  saveMemory(request: Omit<MemorySaveRequest, 'userId' | 'tenantId'>): Promise<MemorySaveResult> {
    return this.post('/memory/save', {
      ...request,
      userId: this.settings.userId,
      tenantId: this.settings.tenantId,
    } satisfies MemorySaveRequest)
  }

  judge(request: JudgeRequest): Promise<JudgeResult> {
    return this.post('/eval/llmjudge', request)
  }

  judgeBatch(request: BatchJudgeRequest): Promise<BatchJudgeResult> {
    return this.post('/eval/batch', request)
  }
}
