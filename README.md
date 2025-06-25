# KnowFlowAIWeb - 智能知识库问答系统

KnowFlowAIWeb 是一个功能强大、易于部署的本地知识库问答Web应用。它基于 **检索增强生成 (RAG)** 架构，结合了先进的文本嵌入模型、向量数据库和大型语言模型（LLM），旨在为您提供一个私有、安全且高效的智能信息处理中心。

用户可以上传自己的文档（支持PDF, TXT, DOCX格式），系统会自动处理这些文档，构建一个可供检索的知识库。当用户提出问题时，应用会首先从知识库中检索最相关的信息片段，然后将这些信息与原始问题一同发送给大型语言模型，生成精准、可靠的回答。

---

## ✨ 主要功能

- **多格式文档支持**: 轻松上传和管理 `.txt`, `.pdf`, `.docx` 格式的文档，构建您的专属知识库。
- **先进的RAG架构**: 采用 `m3e-base` 嵌入模型和 `FAISS` 向量数据库，确保了信息检索的高效性和相关性。
- **流畅的对话体验**:
    - 对接主流大型语言模型（默认配置为智谱AI GLM系列），提供高质量的问答能力。
    - 支持流式响应，实现类似打字机的实时输出效果，提升用户交互体验。
- **联网搜索能力 (可选)**: 当本地知识库无法回答时，可启动联网搜索，获取最新信息。
- **完善的会话管理**:
    - 使用 SQLite 数据库持久化存储聊天记录。
    - 自动为新开启的对话生成摘要，方便历史追溯。
- **高性能与高可用性**:
    - 基于 FastAPI 构建，性能卓越。
    - 耗时的索引构建任务在后台异步执行，保证API接口的快速响应。
- **易于部署和配置**:
    - 通过配置文件和环境变量管理，轻松切换模型、API密钥等。
    - 提供健康检查接口，便于集成到容器化部署流程中。

---

## 🏗️ 技术架构

项目采用模块化设计，核心组件包括：

- **`app.py`**: 应用主文件，使用 **FastAPI** 框架构建，整合了所有模块并定义了API端点。
- **`config.py`**: 集中式配置管理，用于设置API密钥、模型路径、数据库路径等。推荐使用环境变量进行配置。
- **`DatabaseManager`**: 数据库管理类，封装了所有与 **SQLite** 相关的操作，如会话历史的增删改查。
- **`DocumentProcessor`**: 文档处理类，负责从不同格式的文件中提取纯文本内容，并将其分割成适合嵌入的文本块 (Chunk)。
- **`DocumentManager`**: 文档元数据管理类，负责跟踪用户上传的文档信息（如文件名、存储路径），并将其信息持久化到JSON文件中。
- **`VectorStore`**: 向量存储核心类，封装了 **FAISS** 索引、文本向量化（使用 `sentence-transformers`）以及相似性检索的全部逻辑。

---

## 🚀 快速开始

### 1. 环境准备

- Python 3.8+
- 一个支持 `pip` 的包管理工具

### 2. 克隆与安装

```bash
# 克隆项目到本地
git clone <your-repository-url>
cd KnowFlowAIWeb

# 创建并激活Python虚拟环境 (推荐)
python -m venv venv
# Windows
# venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装所有依赖项
pip install -r requirements.txt
```

### 3. 应用配置

应用的所有配置都通过环境变量进行管理，请在项目根目录下创建一个 `.env` 文件（或直接设置系统环境变量），并填入以下内容：

```env
# OpenAI / ZhipuAI 等大语言模型的API Key
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# 模型的API基地址
# 例如，智谱AI的地址为: https://open.bigmodel.cn/api/paas/v4/
BASE_URL="https://open.bigmodel.cn/api/paas/v4/"

# (可选) Google自定义搜索的API Key和搜索引擎ID，用于启用网络搜索
SEARCH_API_KEY="your_google_search_api_key"
SEARCH_ENGINE_ID="your_google_search_engine_id"
```

**注意**:
- `m3e-base` 嵌入模型在首次运行时会自动从Hugging Face下载并缓存到 `local_m3e_model` 目录。
- 首次运行后，系统会自动创建 `chat_history.db` (数据库文件) 和 `docs` (文档存储目录)。

### 4. 运行应用

```bash
# 在项目根目录下运行
uvicorn app:app --host 0.0.0.0 --port 8000
```

应用启动后，您可以访问 `http://127.0.0.1:8000/static/chat.html` 查看主聊天页面，或访问 `http://127.0.0.1:8000/docs` 查看自动生成的API文档。

---

## 📚 API 端点说明

应用提供了以下主要的API接口：

- **`POST /api/upload`**: 上传新文档。
- **`GET /api/documents`**: 获取所有已上传文档的列表。
- **`DELETE /api/documents/{doc_id}`**: 删除指定的文档。
- **`POST /api/stream`**: 发起聊天请求，获取流式响应。
- **`GET /api/chat/history`**: 获取所有聊天会话的摘要列表。
- **`GET /api/chat/session/{session_id}`**: 获取指定会话的完整消息历史。
- **`DELETE /api/chat/session/{session_id}`**: 删除指定的聊天会话。
- **`GET /health`**: 应用健康检查接口，返回 `{"status": "healthy"}`。

详细的请求/响应格式请参考 `http://127.0.0.1:8000/docs` 中的Swagger UI文档。

---

## 🤝 贡献

欢迎任何形式的贡献，包括但不限于：
- 提交问题报告 (Issues)
- 提出新功能建议
- 贡献代码 (Pull Requests)

感谢您的关注和使用！
