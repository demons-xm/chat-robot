export interface ApiResponse<T> {
  code: number
  message: string
  data: T
  requestId: string
}

export interface ChatRequest {
  sessionId: string
  userId: string
  query: string
  tenantId: string
  channel?: 'web' | 'app' | 'mini-program' | string
  extraEntity?: Record<string, unknown>
  timeout?: number
}

export interface ReferenceDocument {
  docId: string
  content: string
  score?: number
}

export interface IntentInfo {
  intentLabel: string
  confidence: number
  subIntent: string
  hitKeywords: string[]
  scoreDetail: {
    llmFewShotScore: number
    embeddingScore: number
    regexKeywordScore: number
  }
  needConfirm: boolean
  transferHuman: boolean
  queryType?: 'general_knowledge' | 'personal_order_status' | 'other'
  rewriteConsistent?: boolean
}

export interface QueryUnderstanding {
  originalQuery: string
  rewrittenQuery: string
  rewriteConsistent: boolean
  queryType: 'general_knowledge' | 'personal_order_status' | 'other'
  requiredSlots: string[]
  missingSlots: string[]
}

export interface ChatResult {
  answer: string
  intentInfo: IntentInfo
  agentName: string
  referenceDocs: ReferenceDocument[]
  queryUnderstanding: QueryUnderstanding
  transferHuman: boolean
  transferReason: string
  sessionSummary: string
}

export interface IntentRequest {
  query: string
  tenantId: string
  sessionId?: string
}

export interface IntentResult extends Partial<IntentInfo> {
  intentLabel: string
  confidence: number
}

export interface RagRetrieveRequest {
  query: string
  tenantId: string
  topK?: number
  enableQueryRewrite?: boolean
}

export interface RagRetrieveResult {
  rewriteQueries: string[]
  rawRetrieveList: ReferenceDocument[]
  rerankDocs: ReferenceDocument[]
  mrr: number | null
}

export interface AgentRouteRequest {
  intentLabel: string
  entity?: Record<string, unknown>
  tenantId: string
  query?: string
}

export interface AgentRouteResult {
  targetAgent: string
  routeReason: string
  fallback: boolean
}

export interface MemoryKeyRequest {
  sessionId: string
  userId: string
  tenantId: string
}

export interface WorkingMemoryItem {
  role: 'user' | 'assistant' | 'system' | string
  content: string
}

export interface MemoryResult {
  workingMemory: WorkingMemoryItem[]
  episodicMemory: string[]
  userProfile: Record<string, unknown>
}

export interface MemorySaveRequest extends MemoryKeyRequest {
  query: string
  answer: string
  sessionSummary?: string
  userFeatures?: Record<string, unknown> | string[]
}

export interface MemorySaveResult {
  queued: boolean
}

export interface JudgeRequest {
  query: string
  answer: string
  reference?: string
}

export interface JudgeResult {
  evaluationId?: string
  score: {
    relevance: number
    accuracy: number
    completeness: number
    usefulness: number
  }
  avgScore: number
  judgeReason: string
  qualityTag: '优质' | '一般' | '不合格' | string
}

export interface BatchJudgeSample extends JudgeRequest {
  sampleId?: string
}

export interface BatchJudgeRequest {
  samples: BatchJudgeSample[]
}

export interface BatchJudgeResult {
  total: number
  completed: number
  failed: number
  avgScore: number | null
  results: Array<JudgeResult & { sampleId?: string; error?: string }>
}
