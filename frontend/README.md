# 智能客服 Agent 前端

基于 Vue 3、TypeScript 和 Vite 的智能客服工作台，包含多轮聊天、意图/知识检索/路由/记忆调试，以及回答质量评测功能。

## 本地开发

```powershell
cd D:\agent_pro\chat-robot\frontend
npm ci
npm run dev
```

前端默认请求 `http://127.0.0.1:8000/api/v1`。后端需要先启动，并配置模型服务和 JWT；主问答使用的 JWT 中 `sub`、`userId` 或 `user_id` 必须与连接设置中的用户 ID 一致。

构建、类型检查和 lint：

```powershell
npm run type-check
npm run build
npm run lint
```

## 临时演示：GitHub Pages 前端 + 本机 FastAPI 后端

GitHub Pages 只能托管静态前端；部署后的浏览器无法访问你电脑上的 `localhost`。要从 Pages 演示真实问答，需要在演示期间运行本机后端，并通过 HTTPS 隧道临时公开后端端口。

### 1. 发布前端

仓库包含 `.github/workflows/deploy-pages.yml`。将代码推送到 `main` 后，工作流会构建 `frontend` 并发布到：

<https://demons-xm.github.io/chat-robot/>

首次部署前，在 GitHub 仓库 **Settings → Pages → Build and deployment** 中把 **Source** 设为 **GitHub Actions**。之后可在 **Actions → Deploy frontend to GitHub Pages** 查看构建和发布状态，也可手动运行该工作流。

若后端隧道地址稳定，可在 **Settings → Secrets and variables → Actions → Variables** 添加仓库变量 `VITE_API_BASE_URL`，值为完整 API 前缀，例如 `https://your-tunnel.example/api/v1`，然后重新运行部署工作流。临时隧道域名经常变化时，不需要重建前端，可直接在工作台的“连接设置”里填写新地址。

### 2. 临时公开本机后端

安装 [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) 后，在两个 PowerShell 窗口分别运行：

窗口一，启动本机后端：

```powershell
cd D:\agent_pro\chat-robot\backend
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

窗口二，创建临时 HTTPS 隧道：

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

cloudflared 会打印一个随机 `https://....trycloudflare.com` 地址。隧道运行期间，外部用户可经该地址访问本机后端；关闭隧道进程后，公网访问即停止。

### 3. 配置工作台连接

打开 Pages 页面右上角的“连接设置”，填写：

| 字段 | 示例 |
| --- | --- |
| 后端 API Base URL | `https://随机隧道域名.trycloudflare.com/api/v1` |
| JWT | 由本机后端配置的 `JWT_SECRET_KEY` 对应签发的测试 JWT |
| 用户 ID | 与 JWT 的 `sub`、`userId` 或 `user_id` 声明一致 |
| 租户 ID | `default` |

保存后即可测试。浏览器会将连接设置保存在当前浏览器的 `localStorage`，包含 JWT；不要在公用电脑上保存真实凭据。重启 Quick Tunnel 后域名会改变，需要更新 API Base URL。

后端默认 CORS 允许 `https://demons-xm.github.io`。如果 Pages 使用不同的自定义域名，需将其精确加入后端 `CORS_ALLOWED_ORIGINS` 并重启后端。

### 临时演示的安全边界

- 隧道会把本机后端暴露给互联网访问。仅在演示时开启，结束后关闭 cloudflared 和后端。
- 当前后端只有 `/agent/chat` 要求 JWT；其他业务 API 尚未统一鉴权。CORS 不是认证或访问控制，非浏览器客户端仍可能访问公开隧道。
- 仅使用测试数据和短期测试 JWT；不要公开模型 API Key、`JWT_SECRET_KEY`、真实用户数据或生产 Token。不要把这些值设置为 Pages 构建变量，也不要提交到仓库。
- GitHub Pages 的前端代码对所有人公开。任何写入 `VITE_*` 的值都会打包到浏览器，不能放置密钥或 JWT。

## 常用命令

```powershell
npm ci
npm run dev
npm run type-check
npm run build
npm run lint
```
