# PI Agent - MemorySearch

一个基于 Python 构建的智能对话 Agent，集成了 MemorySearch 工具，支持多用户后台服务。

## 项目概述

PI Agent 是一个功能强大的 AI 对话代理系统，核心特性包括：

- **MemorySearch 工具**：智能记忆检索，支持语义搜索和关键词匹配
- **多用户支持**：完整的用户隔离机制，确保数据安全
- **历史对话管理**：持久化存储用户对话记录，支持会话恢复
- **后台服务架构**：可部署为独立的后台应用服务

## 系统架构

```
pypi-mono/
├── agent/                 # Agent 核心逻辑
│   ├── __init__.py
│   ├── core.py           # Agent 主类
│   └── tools/            # 工具集
│       ├── __init__.py
│       └── memory_search.py  # MemorySearch 工具实现
├── api/                   # API 接口层
│   ├── __init__.py
│   └── routes.py         # 路由定义
├── storage/              # 数据存储层
│   ├── __init__.py
│   ├── database.py       # 数据库连接管理
│   └── models.py         # 数据模型定义
├── services/             # 业务服务层
│   ├── __init__.py
│   ├── user_service.py   # 用户管理服务
│   └── chat_service.py   # 对话管理服务
├── config/               # 配置管理
│   ├── __init__.py
│   └── settings.py       # 应用配置
├── tests/                # 测试用例
├── requirements.txt      # 依赖列表
└── README.md
```

## 核心功能

### 1. MemorySearch 工具

MemorySearch 是一个智能记忆检索工具，具备以下能力：

- **语义搜索**：基于向量相似度的智能匹配
- **关键词搜索**：精确关键词匹配
- **时间范围过滤**：按日期范围检索历史记录
- **用户隔离**：确保搜索结果仅限当前用户数据

### 2. 多用户支持

- 独立的用户会话管理
- 用户级别的对话历史隔离
- 支持并发请求处理
- 安全的认证与授权机制

### 3. 历史对话管理

- 持久化存储所有对话记录
- 支持会话恢复与继续
- 对话历史导出功能
- 按时间、主题检索历史对话

## 快速开始

### 环境要求

- Python 3.10+
- SQLite / PostgreSQL（可选）

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd pypi-mono

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置

```bash
# 复制配置模板
cp config/settings.example.py config/settings.py

# 编辑配置文件，设置必要的参数
# - API Keys
# - 数据库连接信息
# - 日志级别等
```

### 运行

```bash
# 启动后台服务
python -m api.routes

# 或使用 uvicorn（推荐）
uvicorn api.routes:app --host 0.0.0.0 --port 8000
```

## API 接口

### 用户管理

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/users/register` | POST | 用户注册 |
| `/api/users/login` | POST | 用户登录 |
| `/api/users/logout` | POST | 用户登出 |

### 对话管理

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/chat/sessions` | GET | 获取用户所有会话 |
| `/api/chat/sessions/{id}` | GET | 获取指定会话详情 |
| `/api/chat/message` | POST | 发送消息 |
| `/api/chat/history` | GET | 获取对话历史 |

### MemorySearch

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/memory/search` | POST | 执行记忆搜索 |
| `/api/memory/index` | POST | 索引新记忆 |

## 技术栈

- **语言**: Python 3.10+
- **Web 框架**: FastAPI / Flask
- **数据库**: SQLite（开发）/ PostgreSQL（生产）
- **向量存储**: FAISS / ChromaDB
- **AI SDK**: OpenAI / Anthropic SDK

## 开发指南

### 代码风格

本项目遵循 PEP 8 规范，使用以下工具保证代码质量：

```bash
# 代码格式化
black .

# 代码检查
ruff check .

# 类型检查
mypy .
```

### 测试

```bash
# 运行所有测试
pytest tests/

# 带覆盖率报告
pytest tests/ --cov=. --cov-report=html
```

### 提交规范

使用 Conventional Commits 格式：

```
<type>: <description>

# type: feat, fix, refactor, docs, test, chore
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题，请提交 Issue 或联系维护者。
