# 智能客服 Agent（LLM 问答平台）后端

基于 FastAPI 的智能客服后端。主问答接口负责协调意图识别、会话记忆、Agent 路由、知识库检索和大模型生成，并将记忆保存及质量评测放入后台执行。

> 本文以当前后端实现为准。接口在线 Schema、可执行请求以及响应模型可在服务启动后通过 Swagger UI 查看。

## 目录

- [功能概览](#功能概览)
- [技术组件和数据流](#技术组件和数据流)
- [环境要求](#环境要求)
- [安装和运行](#安装和运行)
- [配置说明](#配置说明)
- [API 使用](#api-使用)
- [知识库管理](#知识库管理)
- [记忆、评测与限流](#记忆评测与限流)
- [日志、数据和备份](#日志数据和备份)
- [测试与开发](#测试与开发)
- [部署和安全建议](#部署和安全建议)
- [故障排查](#故障排查)
- [项目结构](#项目结构)

## 功能概览

| 能力 | 说明 |
| --- | --- |
| 主问答 | 单一入口组织识别、路由、记忆、检索及回答，支持多轮对话 |
| 意图识别 | 结合关键词规则、向量语义相似度及可用时的大模型分类 |
| Agent 路由 | 根据意图、订单/产品等实体和问题关键词选择目标 Agent |
| RAG | Chroma 向量检索、BM25 词法召回、结果融合、可选 Query 改写和重排 |
| 上下文查询理解 | 用近期对话补全省略查询；改写经单独语义校验后才用于回答和联合检索 |
| 租户隔离 | 知识文档、会话工作记忆及长期记忆均使用租户标识进行范围过滤 |
| 三级记忆 | Redis 工作记忆、Chroma 情景记忆和用户画像 |
| 质量评测 | 单条 LLM-as-Judge 和批量回归评测；主问答结束后也会异步评测 |
| API 约束 | 统一 JSON 响应、JWT 主问答校验、请求限流、CORS 来源控制 |

本项目目前不包含用户注册/登录、JWT 签发、真实人工坐席转接、知识文档上传管理页面或管理后台。转人工字段是业务建议，不代表已呼叫人工坐席。

### 前端工作台预览

以下截图展示项目配套的 Vue 3 客服工作台及其调试、评测页面。图片存放在 `backend/data/pictures/`，本文档使用相对路径引用，克隆仓库后可直接查看。

#### 智能客服对话

主工作区用于多轮客服问答，展示 Agent 路由、参考资料和会话记忆状态。

![智能客服对话工作台](data/pictures/智能客服.png)

#### 能力调试

能力调试页可分别测试意图识别、知识检索、Agent 路由和三级记忆接口。

![能力调试页面](data/pictures/能力调试.png)

#### 质量评测

质量评测页支持单条 LLM-as-Judge 评测及批量回归评测。

![质量评测页面](data/pictures/质量评测.png)

## 技术组件和数据流

### 主要组件

- **Web/API**：FastAPI、Uvicorn、Pydantic Settings。
- **聊天模型**：默认 DeepSeek；也支持通过配置切换到 OpenAI 兼容的模型服务。
- **Embedding**：默认 `BAAI/bge-small-zh-v1.5`，用于意图语义表示和向量检索。
- **向量存储**：本地持久化 Chroma。
- **词法召回**：BM25。
- **可选重排**：CrossEncoder；默认关闭。
- **短期记忆**：Redis；Redis 不可用时退化为进程内缓存。
- **长期记忆**：Chroma，与向量库共用配置的持久化目录。
- **评测结果**：本地 SQLite 文件。

### 主问答处理顺序

1. 校验 Bearer JWT，并检查 JWT 身份与请求 `userId` 一致。
2. 清理问题文本，提取手机号、订单号、商品号等常见实体，并合并调用方提供的 `extraEntity`。
3. 将近期工作记忆与会话摘要交给对话理解器，确定当前意图、查询类型、必要信息，并在需要时将省略式追问改写为独立查询。
4. 对发生变化的改写，另行调用 LLM 做语义一致性校验；校验结果不一致、模型不可用或改写丢失否定/实体时，回退至原始问题。
5. 根据意图和提取实体选择目标 Agent；“支持哪些快递”等配送方式问题会优先识别为配送咨询。
6. 对一般知识问答，以原始问题和通过校验的改写作为联合检索输入，再基于检索资料及对话历史生成回答。
7. 目前没有订单/物流实时查询 API。个人订单状态问题会明确说明能力限制并建议从下单平台查询，不会反复要求用户提供系统当前无法用于查询的订单号，也不会编造物流状态。
8. 仅在仍需澄清且问题不是个人订单实时状态查询时进行澄清；只有用户明确要求人工时才给出转人工建议。
9. 返回答案、意图、Agent、引用文档、查询理解元数据、转人工建议和摘要。
10. 异步提交记忆保存和问答质量评测；后台任务失败不会改写已经返回的答案。

以上流程中的模型调用、知识库和 Redis 可用性会影响部分功能；具体降级行为见[记忆、评测与限流](#记忆评测与限流)。

## 环境要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 包管理器
- 当前所选聊天模型服务的 API Key
- Redis（可选；未运行时工作记忆仅保存在当前进程短期缓存）
- 网络连接（首次下载 Hugging Face embedding/重排模型时需要，或预先准备本地模型）

安装依赖时，`uv sync` 会按照 `pyproject.toml` 和 `uv.lock` 创建/同步虚拟环境。

## 安装和运行

以下示例针对 Windows PowerShell。请在后端目录执行命令：

```powershell
cd D:\agent_pro\chat-robot\backend
uv sync
uv run uvicorn main:app --reload
```

首次运行可能耗时较长：应用启动时会尝试初始化评测存储和知识向量库，也可能加载或下载 embedding 模型。确认终端出现 `Application startup complete` 后再测试 API。

服务默认监听 `127.0.0.1:8000`：

| 地址 | 用途 |
| --- | --- |
| <http://127.0.0.1:8000/> | 健康检查 |
| <http://127.0.0.1:8000/docs> | Swagger UI |
| <http://127.0.0.1:8000/redoc> | ReDoc |
| <http://127.0.0.1:8000/openapi.json> | OpenAPI JSON |

健康检查响应示例：

```json
{
  "code": 200,
  "message": "智能客服Agent服务运行中",
  "data": null,
  "requestId": "..."
}
```

`--reload` 是开发热重载模式，不用于生产部署。需要修改默认配置时，编辑 `app/config.py` 后重启服务。添加/修改知识文档后也建议重启，以重新加载启动时构建的索引。

### 指定监听地址和端口

本地开发可通过 Uvicorn 命令参数指定监听地址和端口，例如：

```powershell
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

监听 `0.0.0.0` 会使服务可从其他网络接口访问。只在确有需要且网络访问已受控时使用；这不会自动配置防火墙、HTTPS 或身份认证。

## 配置说明

所有配置类型和默认值集中在 `app/config.py` 的 `Settings` 类中。**非敏感普通参数应在该文件中查看和修改；`backend/.env` 仅用于敏感值。**

配置基于 Pydantic Settings：进程环境变量可能覆盖代码默认值；`.env` 文件由当前配置加载，并在环境变量未覆盖时提供对应值。实际部署时如通过系统环境变量或容器 Secret 注入配置，请记录生效值来源，避免排查时只检查 `.env`。

### `.env` 中的敏感值

仅填写当前使用到的密钥，不要将真实值写入文档、代码提交、截图、问题描述或日志：

```dotenv
# 使用默认 DeepSeek 模型时需要
DEEPSEEK_API_KEY=替换为你的模型API密钥

# 只有选择 OpenAI 兼容模型时才需要
MS_API_KEY=替换为兼容服务API密钥

# 主问答 JWT 验签密钥；必须配置
JWT_SECRET_KEY=替换为高强度随机字符串
```

说明：

- 默认 `CHOOSE_CHAT_MODEL` 为 `"deepseek"`，此时聊天模型使用 `DEEPSEEK_API_KEY`、`CHAT_MODEL_NAME`、`DEEPSEEK_BASE_URL`。
- 当 `CHOOSE_CHAT_MODEL` 设为 `"openai"` 时，聊天模型使用 `MS_API_KEY`、`MS_MODEL_NAME`、`MS_BASE_URL`。这里的 `"openai"` 指 OpenAI 兼容接口，不限定某个模型供应商。
- `JWT_SECRET_KEY` 为空时，主问答接口会拒绝请求。它只用于后端验签，不应发送给浏览器；浏览器只提交用它签名的 JWT。
- embedding 模型当前使用本地 Hugging Face 模型，不通过 `EMBEDDING_API_KEY` 调用远程 Embedding API；设置一个未被代码读取的密钥不会自动启用对应服务。
- 当前代码未读取 Tavily、代理 Key 等配置；如需启用这些功能，需先实现相应服务逻辑。
- 不要将 `.env` 复制到前端目录。前端只需要后端地址、JWT、用户 ID 和租户 ID。

### 本地生成测试 JWT

后端没有登录或 Token 签发 API。以下示例是在本地使用后端密钥生成测试令牌，适用于 `userId=u-001`。先确认 `.env` 的 `JWT_SECRET_KEY` 已设置，再从 `backend` 目录运行：

```powershell
uv run python
```

在 Python 交互提示符中执行：

```python
from datetime import datetime, timedelta, timezone
import jwt
from app.config import config

token = jwt.encode(
    {
        "sub": "u-001",
        "exp": datetime.now(timezone.utc)
        + timedelta(hours=config.JWT_EXPIRE_HOURS),
    },
    config.JWT_SECRET_KEY,
    algorithm=config.JWT_ALGORITHM,
)
print(token)
```

复制输出的完整 JWT 到前端连接设置的 Token 字段，并设置用户 ID 为 `u-001`。Token 过期时间默认为 24 小时。生产系统必须由受信任的认证服务签发 Token；不要把固定测试身份或 JWT 密钥用于生产。

### 普通运行参数清单

下表默认值与当前 `app/config.py` 对应。改变这些默认值应编辑 `Settings` 类，而非把普通参数添加到 `.env`。

#### 模型、鉴权和浏览器来源

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `CHOOSE_CHAT_MODEL` | `"deepseek"` | 当前聊天模型提供方：`"deepseek"` 或 `"openai"` |
| `CHAT_MODEL_NAME` | `"deepseek-chat"` | DeepSeek 聊天模型名 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek API 根地址 |
| `MS_MODEL_NAME` | `"deepseek-chat"` | OpenAI 兼容聊天模型名 |
| `MS_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容 API 根地址 |
| `HF_EMBEDDING_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | 本地 Hugging Face Embedding 模型 |
| `JWT_EXPIRE_HOURS` | `24` | README 本地测试 Token 示例使用的期限 |
| `JWT_ALGORITHM` | `"HS256"` | JWT 验签算法 |
| `CORS_ALLOWED_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | 允许的浏览器来源，逗号分隔；需包含协议和端口 |

#### 存储、记忆和请求控制

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis 工作记忆连接地址 |
| `MEMORY_TTL_SECONDS` | `1800` | 工作记忆和 Redis 会话索引过期时间（秒） |
| `MEMORY_WINDOW_TURNS` | `10` | 每个会话工作记忆保留的对话轮数 |
| `DEFAULT_TENANT_ID` | `"default"` | 知识库根目录文档默认归属的租户 |
| `RATE_LIMIT_PER_MINUTE` | `120` | 进程内限流键每分钟允许的请求数 |
| `DEFAULT_CHAT_TIMEOUT_MS` | `8000` | 主问答默认超时时间（毫秒） |

#### 意图识别参数

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `INTENT_HIGH_THRESHOLD` | `0.75` | 高置信度阈值 |
| `INTENT_LOW_THRESHOLD` | `0.4` | 置信度澄清区间的下界；低于该值不会仅因低置信度触发转人工 |
| `INTENT_KEYWORDS_JSON` | `""` | 额外/覆盖的意图关键词规则 JSON |
| `INTENT_EXAMPLES_JSON` | `""` | 额外的意图 few-shot 描述 JSON |

`INTENT_KEYWORDS_JSON` 必须是 JSON 对象，值为字符串数组；数组中的字符串作为正则表达式匹配。例如：

```json
{"订单咨询":["订单号","物流","发货"]}
```

`INTENT_EXAMPLES_JSON` 必须是 JSON 对象，键和值均为字符串。例如：

```json
{"订单咨询":"查询订单、物流或发货进度"}
```

两项都应保持为有效 JSON 字符串。应用进程启动后首次解析并缓存配置；修改后需要重启进程。

#### RAG、切分和重排

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_CONTEXT` | `4000` | 预留的上下文长度配置；当前服务流程未直接读取 |
| `CHUNK_SIZE` | `300` | 文档切块长度 |
| `CHUNK_OVERLAP` | `30` | 相邻文档块重叠长度 |
| `SEPARATORS` | `["\n\n","。",".","?","？","!"," ",""]` | 文本切分分隔符，按列表顺序应用 |
| `TOP_K` | `10` | 向量与 BM25 底层召回数量 |
| `USE_BM25` | `true` | 是否构建和使用 BM25 词法检索 |
| `RERANK` | `false` | 是否启用 CrossEncoder 重排 |
| `RERANK_TOP_K` | `5` | 向量库辅助重排方法的默认参数；主问答/RAG API 会显式传入本次 `topK`，因此请求值优先 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | CrossEncoder 重排模型 |

单次 `/rag/retrieve` 可通过 `topK` 指定最终请求数量（1–20，默认 3）；`TOP_K` 是底层召回的默认数量，两者用途不同。当前实现启用 Query 改写时会在聊天模型可用的情况下尝试改写；有候选文档且聊天模型可用时会尝试使用聊天模型重新排序，失败时保留融合排序。`RERANK=true` 额外启用本地 CrossEncoder 重排；因此 `RERANK=false` 只关闭 CrossEncoder，不代表关闭聊天模型排序。

#### 数据路径

以下路径均在 `app/config.py` 中计算或设置，默认相对于 `backend/data`：

| 配置项 | 默认路径 | 用途 |
| --- | --- | --- |
| `KNOWLEDGE_PATH` | `data/knowledge` | PDF/Markdown/TXT 知识文档 |
| `CHROMA_DB_PATH` | `data/chroma_db` | Chroma 向量数据和长期记忆 |
| `MD5_PATH` | `data/md5_bge_small_zh_v1_5.txt` | 已索引文件的 MD5 记录 |
| `HISTORY_PATH` | `data/history` | 创建的历史数据目录；当前核心问答流程未直接写入该目录 |
| `LOG_PATH` | `data/logger` | 应用日志目录 |

## API 使用

### 统一协议

- API 前缀：`/api/v1`。
- 除健康检查外，业务请求均为 JSON `POST`。
- 请求及响应字段使用 camelCase；未知的请求字段会被拒绝。
- 统一响应形状：

```json
{
  "code": 200,
  "message": "请求成功",
  "data": {},
  "requestId": "..."
}
```

`code=200` 代表业务成功。错误响应也使用 `code`、`message`、`data`、`requestId` 字段；具体 HTTP 状态码可用于区分错误类型。

### 接口目录

| 方法 | 路径 | 鉴权 | 用途 |
| --- | --- | --- | --- |
| `GET` | `/` | 无 | 服务健康检查 |
| `POST` | `/api/v1/agent/chat` | Bearer JWT | 智能客服主问答 |
| `POST` | `/api/v1/intent/classify` | 无 | 独立意图识别 |
| `POST` | `/api/v1/rag/retrieve` | 无 | 租户范围知识检索 |
| `POST` | `/api/v1/agent/router` | 无 | 独立路由判断 |
| `POST` | `/api/v1/memory/get` | 无 | 查询三级记忆 |
| `POST` | `/api/v1/memory/save` | 无 | 提交记忆写入任务 |
| `POST` | `/api/v1/eval/llmjudge` | 无 | 单条回答评测 |
| `POST` | `/api/v1/eval/batch` | 无 | 批量回答回归评测 |

**重要：**除 `/agent/chat` 外的 API 当前没有 JWT 校验。租户 ID 用于数据筛选，不等同于认证或授权；在公网或多租户生产环境中，不应未经访问控制直接暴露这些接口。

启动服务后，也可打开 <http://127.0.0.1:8000/docs> 查看 Swagger UI 中的在线接口说明。下图展示当前 API 分组和主要端点：

![Swagger API 文档中的接口列表](data/pictures/docs.png)

### 常见 HTTP 状态码

| HTTP 状态 | 含义 | 建议处理 |
| --- | --- | --- |
| `200` | 请求处理成功 | 检查响应体 `code` 和 `data` |
| `400` | 请求字段缺失、格式错误或不符合范围 | 根据 Swagger Schema 修正 JSON |
| `401` | 主问答缺少/无效 Token，或 Token 用户与 `userId` 不匹配 | 检查密钥、签名算法、到期时间和用户 ID |
| `404` | 路径不存在 | 检查 API 前缀 `/api/v1` 及路由名 |
| `429` | 来源 IP 或租户/IP 限流 | 退避后重试，勿高频自动重试 |
| `500` | 未处理的服务异常 | 使用 `requestId` 关联后端日志 |
| `503` | 聊天模型未配置或不可用 | 检查模型 Key 和服务配置 |
| `504` | 主问答超时 | 检查模型/检索时延并适当调大请求超时和网关超时 |

### 1. 主问答：`POST /api/v1/agent/chat`

请求头：

```http
Authorization: Bearer <JWT>
Content-Type: application/json
```

请求示例：

```json
{
  "sessionId": "session-001",
  "userId": "u-001",
  "tenantId": "default",
  "query": "我的订单什么时候发货？",
  "channel": "web",
  "extraEntity": {
    "orderNo": "OD20261016001"
  },
  "timeout": 30000
}
```

字段约束：

| 字段 | 类型 | 必需 | 约束/默认值 |
| --- | --- | --- | --- |
| `sessionId` | string | 是 | 1–128 字符；多轮对话保持相同值 |
| `userId` | string | 是 | 1–128 字符；必须与 JWT 声明的身份一致 |
| `tenantId` | string | 是 | 1–128 字符；用于知识和记忆范围 |
| `query` | string | 是 | 1–10000 字符；不能全为空白 |
| `channel` | string | 否 | 最多 32 字符 |
| `extraEntity` | object | 否 | 补充订单号、产品号等结构化实体 |
| `timeout` | integer | 否 | 毫秒；100–120000；默认 `DEFAULT_CHAT_TIMEOUT_MS` |

JWT 声明中 `userId`、`user_id` 或 `sub` 至少需要一个能提供用户标识，且与请求 `userId` 字符串值一致。用户端切换账号时也要切换 Token，不能复用另一个用户的会话身份。

响应 `data` 示例：

```json
{
  "answer": "目前我无法直接连接订单或物流系统查询实时状态，您可以在下单平台的“我的订单”中查看物流。",
  "intentInfo": {
    "intentLabel": "订单咨询",
    "confidence": 0.91,
    "subIntent": "发货",
    "hitKeywords": ["订单", "发货"],
    "scoreDetail": {
      "llmFewShotScore": 0.9,
      "embeddingScore": 0.82,
      "regexKeywordScore": 0.85
    },
    "needConfirm": false,
    "transferHuman": false
  },
  "agentName": "OrderConsultAgent",
  "referenceDocs": [],
  "queryUnderstanding": {
    "originalQuery": "物流状态",
    "rewrittenQuery": "物流状态",
    "rewriteConsistent": true,
    "queryType": "personal_order_status",
    "requiredSlots": ["orderNo"],
    "missingSlots": ["orderNo"]
  },
  "transferHuman": false,
  "transferReason": "",
  "sessionSummary": "用户咨询订单发货。"
}
```

`referenceDocs` 只包含本次被用于回答的检索结果；无命中或个人订单实时状态查询时可为空。`queryUnderstanding` 返回原始/改写查询、改写校验结果、查询类型及理解出的必要/缺失信息，便于前端调试或呈现处理过程。当前尚未接入订单/物流实时查询 API；个人状态咨询会明确说明系统能力边界。只有当前用户明确请求人工时才会返回转人工建议，且不意味着系统已实际联系人工客服。异步记忆保存和评测不在该 HTTP 响应中等待完成。

PowerShell 快速调用示例（把 JWT 替换为有效值）：

```powershell
$body = @{
  sessionId = "session-001"
  userId = "u-001"
  tenantId = "default"
  query = "我的订单什么时候发货？"
  channel = "web"
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/agent/chat" `
  -Method Post `
  -Headers @{ Authorization = "Bearer <JWT>" } `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

### 2. 意图识别：`POST /api/v1/intent/classify`

请求：

```json
{
  "query": "我的快递到哪里了？",
  "tenantId": "default",
  "sessionId": "session-001"
}
```

`sessionId` 可选；提供时后端会尝试读取该租户和会话的近期工作记忆作为上下文。`tenantId` 和 `query` 必须提供。

`data` 包含 `intentLabel`、`confidence`、`subIntent`、`hitKeywords`、`scoreDetail`、`needConfirm` 和 `transferHuman`。默认标签包括业务咨询、订单咨询、配送咨询、身份咨询、投诉、建议、故障报修、转人工和闲聊。“支持哪些快递/快递公司/承运商”等明确问题会优先进入配送咨询；“你是谁/你叫什么/你能做什么”等身份问题会优先进入身份咨询。这些确定性规则无需等待大模型分类。

其他问题会融合关键词、Embedding 和可用时的大模型分类结果。主问答还会结合近期上下文生成独立查询，并通过单独的 LLM 语义审核；改写验证失败或服务不可用时采用原始问题。RAG 使用原始问题及通过审核的改写共同召回，避免单一改写改变用户原意。个人订单物流状态暂不支持实时查询，会如实说明限制，不反复追问配送方式或无用订单号。低置信度本身不等同于转人工；明确的转人工请求由独立规则识别。大模型不可用时记录告警并退化到规则/向量分类及原始查询检索；语义 Embedding 模型不可用时也会记录告警。

### 3. 知识检索：`POST /api/v1/rag/retrieve`

请求：

```json
{
  "query": "员工年假如何申请？",
  "tenantId": "default",
  "topK": 3,
  "enableQueryRewrite": true
}
```

| 字段 | 类型 | 必需 | 约束/默认值 |
| --- | --- | --- | --- |
| `query` | string | 是 | 1–10000 字符 |
| `tenantId` | string | 是 | 用于隔离知识文档 |
| `topK` | integer | 否 | 1–20，默认 3 |
| `enableQueryRewrite` | boolean | 否 | 默认 `true` |

`data` 字段：

| 字段 | 说明 |
| --- | --- |
| `rewriteQueries` | 检索实际使用的改写查询 |
| `rawRetrieveList` | 融合/重排前的候选文档 |
| `rerankDocs` | 最终排序结果 |
| `mrr` | 检索 MRR；当前没有标注评测集时为 `null` |

每个文档含 `docId`、`content`，可能包含 `score`。`score` 是检索/排序分数，不应未经解释就当成概率或 MRR。

### 4. Agent 路由：`POST /api/v1/agent/router`

请求：

```json
{
  "intentLabel": "订单咨询",
  "entity": {
    "orderNo": "OD123456"
  },
  "tenantId": "default",
  "query": "查一下这个订单"
}
```

响应 `data` 含 `targetAgent`、`routeReason`、`fallback`。已实现的意图包括订单咨询、业务咨询、投诉、建议、转人工、闲聊和故障报修；可根据意图、实体及问题中的关键词选择目标。此接口用于测试决策，不会调用目标 Agent 或执行转接。

### 5. 读取记忆：`POST /api/v1/memory/get`

请求：

```json
{
  "sessionId": "session-001",
  "userId": "u-001",
  "tenantId": "default"
}
```

`data` 结构：

```json
{
  "workingMemory": [
    {"role": "user", "content": "……"},
    {"role": "assistant", "content": "……"}
  ],
  "episodicMemory": ["历史会话摘要……"],
  "userProfile": {}
}
```

工作记忆按租户、用户和会话定位；情景记忆和画像按租户及用户过滤。

### 6. 保存记忆：`POST /api/v1/memory/save`

请求：

```json
{
  "sessionId": "session-001",
  "userId": "u-001",
  "tenantId": "default",
  "query": "用户问题",
  "answer": "客服回答",
  "sessionSummary": "当前会话摘要",
  "userFeatures": {
    "preferredLanguage": "zh-CN"
  }
}
```

`sessionSummary` 可选且最多 4000 字符；`userFeatures` 可为对象或字符串数组。响应中的 `data.queued=true` 仅表示写入任务已提交线程池，并不保证 Redis/Chroma 持久化已完成。主问答已经自动保存，不应在前端为同一轮重复触发该接口。

### 7. 单条评测：`POST /api/v1/eval/llmjudge`

请求：

```json
{
  "query": "用户问题",
  "answer": "待评测的客服回答",
  "reference": "可选的标准答案或参考知识"
}
```

`reference` 可省略或为 `null`。评测使用聊天模型，对相关性、准确性、完整性和有用性分别给出 0–5 分。响应 `data` 示例：

```json
{
  "evaluationId": "uuid",
  "score": {
    "relevance": 4,
    "accuracy": 4,
    "completeness": 3,
    "usefulness": 4
  },
  "avgScore": 3.75,
  "judgeReason": "回答相关且基本准确。",
  "qualityTag": "一般"
}
```

评测成功后会将记录写入本地 SQLite。模型未配置或调用失败时接口返回错误，不能把错误兜底为成功评分。

### 8. 批量评测：`POST /api/v1/eval/batch`

请求：

```json
{
  "samples": [
    {
      "sampleId": "case-001",
      "query": "订单什么时候发货？",
      "answer": "通常会在下单后24小时内发货。",
      "reference": "订单下单后24小时内发货。"
    },
    {
      "sampleId": "case-002",
      "query": "如何申请年假？",
      "answer": "请在系统中提交年假申请。",
      "reference": "员工通过人事系统提交年假申请。"
    }
  ]
}
```

`samples` 必须包含 1–100 条样本；每条字段约束与单条评测一致，`sampleId` 可选。响应包含 `total`、`completed`、`failed`、`avgScore` 和 `results`。单个样本失败时相应结果包含错误信息，其他样本仍会继续处理；即使全部样本失败，批量 HTTP 请求仍可能以整体成功响应返回，此时以 `failed` 和逐项结果为准。

## 知识库管理

### 目录与租户

服务启动时递归扫描 `KNOWLEDGE_PATH` 下的 `.pdf`、`.md`、`.markdown` 和 `.txt` 文件。根目录中的文件归属 `DEFAULT_TENANT_ID`；子目录的第一层名称作为租户 ID：

```text
backend/
└── data/
    └── knowledge/
        ├── 员工手册.pdf                 # 租户 default
        ├── default/
        │   └── FAQ.md                   # 租户 default
        ├── tenant-a/
        │   ├── 产品说明.txt             # 租户 tenant-a
        │   └── policies/
        │       └── 退换货政策.pdf       # 仍归属 tenant-a
        └── tenant-b/
            └── service.md               # 租户 tenant-b
```

文件夹命名必须与 API 请求使用的 `tenantId` 一致。每次启动应用都会遍历文件并构建进程内 BM25 数据；Chroma 保存向量索引。当前没有公开的 HTTP 上传接口，文档通过服务端文件系统管理。

### 索引行为与注意事项

- 文档被解析、切块后进入词法索引；向量数据进入 Chroma。
- MD5 记录用于识别已处理文件，避免无必要重复写入向量库，并辅助检查已有向量数据的租户元数据。
- 新增文档后重启应用，确保进程内 BM25 索引包含新文件。
- 用不同内容替换原文件、删除文档或调整切块策略时，应检查现有 Chroma 数据和 MD5 记录是否仍与文件状态匹配；必要时按维护流程清理/重建对应索引。删除向量库或 MD5 文件会影响所有使用该存储目录的租户，请先备份并确认操作范围。
- 文件解析失败会记录日志并跳过该文件；检查启动日志中的文件名和解析错误。
- 根目录文档属于默认租户，不会自动共享给所有租户。
- Chroma 向量检索不可用时，服务会记录警告并尝试使用 BM25 结果；长期记忆仍依赖 Chroma，能力会受影响。
- 有候选文档且聊天模型已配置时，服务会尝试使用聊天模型对文档重排；失败时保留融合排序。`RERANK=false` 只表示不启用额外的 CrossEncoder 重排。
- `RERANK=true` 时需确保重排模型文件可用且机器资源足够；重排异常时会记录告警并回退到融合排序。

## 记忆、评测与限流

### 记忆层次

| 记忆层 | 存储 | 内容/生命周期 |
| --- | --- | --- |
| 工作记忆 | Redis | 指定用户、租户、会话的最近若干轮对话；默认 TTL 30 分钟，窗口 10 轮 |
| 情景记忆 | Chroma | 会话摘要，按租户、用户和 memory type 检索 |
| 用户画像 | Chroma | 偏好和用户特征，按租户、用户过滤 |

Redis 访问失败时，工作记忆回退到当前 Python 进程的内存缓存；该缓存不会跨进程或重启持久化。Chroma 长期记忆失败时会记录告警，并返回空情景记忆/画像以尽量保持问答流程。降级意味着数据持久性或检索能力下降，不是等价替代。

### 评测

- 主问答在后台提交 LLM-as-Judge 评测，评测失败不会影响已经返回的对话结果。
- `/eval/llmjudge` 与 `/eval/batch` 为同步 HTTP 请求中的模型调用，响应时间受模型服务影响。
- SQLite 评测数据库在 `data/agent_evaluations.sqlite3`，应用启动时会确保表存在。
- 批量评测逐条处理样本；个别失败会记录日志并返回逐项错误摘要。
- MRR 需要有人工标注的检索评测数据；当前没有相应标注时响应为 `null`。

### 限流

中间件对 `/api/` 下的请求按来源 IP 计数；部分涉及租户的接口还执行租户与来源 IP 组合的第二层限制。限制保存在进程内，按分钟滑动窗口清理；多 worker 部署时各进程的计数并不共享。超限返回 HTTP `429`。生产环境若需要全实例统一限流，应在网关或共享存储中实施。

## 日志、数据和备份

应用日志输出到控制台及 `LOG_PATH`。日志格式含时间、级别、模块/文件、行号和消息；问答及调试接口会记录部分操作、耗时、租户、会话或检索统计信息。不要在日志中写入 JWT、API Key 或完整敏感对话。

需纳入备份计划的数据包括：

- 知识源文件：`data/knowledge`。
- 向量数据和长期记忆：`data/chroma_db`。
- 文件摘要记录：由 `MD5_PATH` 指定的文件。
- 评测结果：`data/agent_evaluations.sqlite3`。
- 运行日志：`LOG_PATH`。
- Redis 持久化数据（如启用 Redis 持久化）。

备份前应停止服务或采用存储引擎认可的一致性备份方式。不要将 `.env` 与普通业务数据备份到公开或未加密的位置；备份恢复时注意 Chroma 数据、知识文件及 MD5 记录的一致性。

## 测试与开发

### 当前仓库的验证情况

`backend` 目录当前没有项目自有的自动化测试套件。可以先执行语法编译检查：

```powershell
cd D:\agent_pro\chat-robot\backend
uv run python -m compileall -q main.py app
```

意图识别和路由的边界测试位于 `tests/test_intent_routing.py`，可使用 Python 标准库运行，无需额外安装测试依赖：

```powershell
uv run python -m unittest discover -s tests -v
```

启动服务后通过 `/docs` 验证请求 Schema，并用健康检查确认服务状态。调用模型的端到端问答和评测需要有效模型 Key 及可用网络。

### 不依赖 Token 的连通性检查

PowerShell:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/"
```

使用符合接口 Schema 的路由请求进行 API 检查（该接口不要求 JWT）：

```powershell
$body = @{
  intentLabel = "订单咨询"
  entity = @{}
  tenantId = "default"
  query = "我的订单什么时候发货？"
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/agent/router" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

该请求只验证 HTTP、Schema、租户限流和路由返回，不验证 JWT、模型调用或知识检索。

### 前端联调

前端默认使用 `http://127.0.0.1:8000/api/v1`。后端默认允许 `http://127.0.0.1:5173` 和 `http://localhost:5173`。若 Vite 端口变化或通过其他主机访问，编辑 `CORS_ALLOWED_ORIGINS`（完整来源，含 scheme/port）并重启后端。

前端连接设置需配置：

| 字段 | 本地示例 |
| --- | --- |
| 后端 API Base URL | `http://127.0.0.1:8000/api/v1` |
| JWT | 后端 `JWT_SECRET_KEY` 签发的测试 JWT |
| 用户 ID | 与 JWT `sub` 相同，如 `u-001` |
| 租户 ID | `default` 或对应知识目录租户 |

CORS 仅解决浏览器跨源访问，不会启动后端，也不会替代 JWT 或 API 授权。

## 部署和安全建议

当前项目面向开发及业务逻辑验证。对外部署前至少完成以下工作：

1. **访问控制**：为目前不要求 JWT 的接口增加认证、授权或可信网关限制；不能信任客户端自行提交的 `tenantId`。
2. **密钥管理**：通过 Secret 管理系统注入 API Key 和强随机 JWT Secret；轮换已泄露的凭证。不要在客户端保存模型 Key。
3. **JWT 生命周期**：接入正式认证/签发流程，校验签名、过期和身份；实现登出/吊销等需求时选择适合的 Token 策略。
4. **HTTPS 与网络**：使用 HTTPS 和反向代理，限制后端直接暴露；配置代理超时、请求体大小和可信代理 IP。
5. **CORS**：只允许实际前端来源，避免宽泛或不受信任的 Origin。
6. **限流**：当前计数为进程内状态；多 worker 或多副本部署需使用共享限流存储或网关策略。
7. **数据安全**：保护知识文件、用户对话、Redis、Chroma 和 SQLite 数据，配置访问权限、备份、保留期限及个人信息治理。
8. **运行模式**：开发重载模式不适用于生产；由进程管理器/容器确保重启、健康探针和资源限制。
9. **模型依赖**：评估模型供应商的数据处理政策、网络超时、可用性、成本和返回格式，并监控降级与错误。
10. **隐私日志**：日志与错误信息不应包含密钥、Token 或不必要的完整用户内容。

以上为部署检查方向，不代表当前代码已实现全部生产级控制。

## 故障排查

| 现象 | 排查步骤 |
| --- | --- |
| 浏览器提示无法连接 | 确认 Uvicorn 仍在运行；浏览器打开 `/` 检查健康；核对 API Base URL 包含 `/api/v1`；确认后端 CORS Origin 与前端实际地址完全一致 |
| 浏览器 OPTIONS 通过但请求失败 | 查看随后的 POST 状态码和后端日志；OPTIONS 200 仅说明 CORS 预检通过，不代表业务请求通过 |
| `/agent/chat` 返回 401 | 检查 `.env` 中 JWT Secret 已配置；Token 用相同算法和密钥签发、未过期；Token 用户声明与请求 `userId` 完全一致 |
| 主问答返回 503 | 检查 `CHOOSE_CHAT_MODEL` 取值、对应 API Key、模型名、Base URL 和模型端服务状态 |
| 主问答或评测超时 | 查看模型响应时延、检索负载及前置代理超时；主问答 `timeout` 不超过 120000ms，并配置足够的代理超时 |
| 意图结果不稳定 | 检查 Embedding 模型是否成功初始化、关键词正则是否有效、JSON 配置格式和阈值设置 |
| 检索结果为空 | 确认知识文件类型受支持、启动日志显示加载成功、文件夹租户与请求 `tenantId` 一致 |
| Chroma 启动/写入警告 | 检查 `CHROMA_DB_PATH` 写权限和磁盘状态；知识向量与长期记忆均依赖该存储 |
| Redis 警告或记忆丢失 | 检查 `REDIS_URL`、Redis 是否可达及 TTL；Redis 故障时工作记忆只在当前进程短期保留 |
| 429 Too Many Requests | 等待限流窗口恢复，避免自动紧密重试；检查是否有多端共享出口 IP 或过低的限制阈值 |
| 400 请求参数错误 | 核对字段名大小写（camelCase）、必需字段、长度范围和额外字段；优先使用 `/docs` 请求示例 |
| 首次启动很慢 | 等待本地 Embedding 模型下载/初始化；检查 Hugging Face 网络连接、磁盘、内存和启动日志 |
| README 中配置修改看似无效 | 核对进程环境变量是否覆盖了代码默认值，并确认修改的是正在运行实例使用的 `backend/app/config.py` |

## 项目结构

```text
backend/
├── app/
│   ├── api/                   # API 路由、JWT 依赖与版本前缀
│   ├── agent/
│   │   ├── rag/               # 检索融合逻辑
│   │   └── vector_store/      # 文件解析、切分、Chroma、BM25、重排
│   ├── common/                # 统一响应和异常处理
│   ├── schemas/               # Pydantic 请求 Schema 和校验
│   ├── services/              # 问答、意图、记忆、评测及限流
│   ├── utils/                 # 日志工具
│   └── config.py              # 类型化配置、默认值和数据路径
├── data/
│   ├── knowledge/             # 知识源文件（需按租户组织）
│   ├── chroma_db/             # 向量索引和长期记忆数据
│   ├── history/               # 历史数据目录
│   └── logger/                # 应用日志
├── .env                       # 本机敏感配置，不要提交真实密钥
├── main.py                    # FastAPI 应用、生命周期、中间件和健康检查
├── pyproject.toml             # Python 版本与直接依赖
├── uv.lock                    # 锁定依赖版本
└── readme.md                  # 本文档
```

应用还会在 `data` 目录创建评测 SQLite 数据库和文件 MD5 记录文件。
