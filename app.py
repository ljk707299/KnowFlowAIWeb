# -*- coding: utf-8 -*-
"""
KnowFlowAIWeb - 一个基于RAG和LLM的智能问答Web应用

本应用使用FastAPI构建，集成了以下功能:
- **文档管理**: 支持上传、删除、和检索 .txt, .pdf, .docx 格式的文档。
- **RAG (检索增强生成)**:
  - 使用 sentence-transformers (m3e-base) 将文档内容编码为向量。
  - 使用 FAISS 进行高效的相似性搜索。
  - 根据用户问题检索相关文档片段作为上下文。
- **LLM 对话**:
  - 对接大语言模型 (默认为智谱AI的GLM系列) 生成回答。
  - 支持流式响应，提供流畅的打字机效果。
  - (可选) 集成网络搜索能力，以回答需要最新信息的问题。
- **聊天历史**:
  - 使用 SQLite 存储和管理多轮对话历史。
  - 自动为新会话生成摘要。

核心架构:
- `config.py`: 集中管理所有配置，如API密钥、模型路径等。
- `DatabaseManager`: 封装所有数据库操作。
- `DocumentProcessor`: 负责从不同格式文件中提取文本并进行分块。
- `DocumentManager`: 管理已上传文档的元数据。
- `VectorStore`: 封装FAISS索引、向量生成和检索逻辑。

API端点:
- `/static/*`: 提供静态文件服务 (前端页面)。
- `/api/upload`: 上传文档。
- `/api/documents`: 获取已上传文档列表。
- `/api/documents/{doc_id}`: 删除指定文档。
- `/api/stream`: (GET/POST) 核心聊天接口，支持流式响应。
- `/api/chat/history`: 获取聊天会话历史列表。
- `/api/chat/session/{session_id}`: 获取/删除特定聊天会话。
- `/health`: 健康检查接口。
"""
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os
import json
import uuid
from datetime import datetime
import asyncio
import sqlite3
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
import urllib.parse
import config
from pypdf import PdfReader
from docx import Document
import logging
import requests

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pydantic 模型定义 ---
# 用于规范API的输入和输出，提供数据校验和自动文档生成

class ChatStreamRequest(BaseModel):
    """聊天流式接口的请求体"""
    query: str = Field(..., description="用户的问题")
    session_id: Optional[str] = Field(None, description="会话ID，如果为空则创建新会话")
    web_search: bool = Field(False, description="是否启用网络搜索")

class DocumentInfo(BaseModel):
    """单个文档的信息"""
    id: str
    name: str

class DocumentListResponse(BaseModel):
    """文档列表的响应体"""
    documents: List[DocumentInfo]

class SessionInfo(BaseModel):
    """单个会话的摘要信息"""
    id: str
    summary: str
    updated_at: str

class ChatHistoryResponse(BaseModel):
    """聊天历史列表的响应体"""
    history: List[SessionInfo]

class Message(BaseModel):
    """单条消息的结构"""
    role: str
    content: str

class SessionMessagesResponse(BaseModel):
    """会话消息列表的响应体"""
    messages: List[Message]

class UploadSuccessResponse(BaseModel):
    """文件上传成功的响应体"""
    message: str
    doc_id: str

# --- 数据库管理 ---
class DatabaseManager:
    """
    处理所有与SQLite数据库相关的操作。
    这个类封装了数据库的连接、初始化、增删改查等所有原子操作，
    使得应用主逻辑无需关心数据库的具体实现。
    """
    def __init__(self, db_path: str):
        """
        初始化数据库管理器。
        :param db_path: SQLite数据库文件路径。
        """
        self.db_path = db_path
        # 在多线程环境下，为每个线程创建独立的连接可能更安全。
        # FastAPI的依赖注入系统可以更好地管理连接生命周期。
        # 为简化，此处使用单个长连接。
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def init_db(self):
        """初始化数据库和表"""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
        ''')
        self.conn.commit()
        logging.info("数据库初始化完成")

    def get_chat_history(self) -> List[Dict]:
        """获取所有聊天会话的摘要列表，按更新时间降序排列。"""
        self.cursor.execute("SELECT id, summary, updated_at FROM chat_sessions ORDER BY updated_at DESC")
        sessions = [{"id": row[0], "summary": row[1], "updated_at": row[2]} for row in self.cursor.fetchall()]
        return sessions

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """获取特定会话的所有消息，按时间升序排列。"""
        self.cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
        messages = [{"role": row[0], "content": row[1]} for row in self.cursor.fetchall()]
        return messages

    def delete_session(self, session_id: str):
        """删除一个会话及其所有消息"""
        self.cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        self.conn.commit()

    def create_chat_session(self, session_id: str, summary: str):
        """创建一个新的聊天会话"""
        self.cursor.execute(
            "INSERT INTO chat_sessions (id, summary) VALUES (?, ?)",
            (session_id, summary)
        )
        self.conn.commit()
    
    def add_message(self, session_id: str, role: str, content: str):
        """向会话中添加一条消息"""
        self.cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        self.cursor.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,)
        )
        self.conn.commit()

    def close(self):
        """关闭数据库连接。在应用关闭时调用。"""
        if self.conn:
            self.conn.close()

    def session_exists(self, session_id: str) -> bool:
        """检查指定的会话ID是否存在。"""
        self.cursor.execute("SELECT 1 FROM chat_sessions WHERE id = ?", (session_id,))
        return self.cursor.fetchone() is not None

# 全局数据库管理器实例
db_manager = DatabaseManager(config.DB_PATH)

# --- 文档处理 ---
class DocumentProcessor:
    """
    处理文档内容的提取和分块。
    支持多种文件格式，并提供一个统一的文本分块方法。
    """
    def read_document(self, file_path: str) -> str:
        """根据文件类型读取文档内容"""
        if file_path.endswith(".pdf"):
            return self._read_pdf(file_path)
        elif file_path.endswith(".docx"):
            return self._read_docx(file_path)
        elif file_path.endswith(".txt"):
            return self._read_txt(file_path)
        else:
            return ""

    def _read_pdf(self, file_path: str) -> str:
        """读取PDF文件内容"""
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages)
            return text
        except Exception as e:
            logging.error(f"读取PDF文件失败: {e}")
            return ""

    def _read_docx(self, file_path: str) -> str:
        """读取DOCX文件内容"""
        try:
            doc = Document(file_path)
            text = "".join(para.text for para in doc.paragraphs)
            return text
        except Exception as e:
            logging.error(f"读取DOCX文件失败: {e}")
            return ""

    def _read_txt(self, file_path: str) -> str:
        """读取TXT文件内容, 尝试多种编码"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    return f.read()
            except Exception as e:
                logging.error(f"读取TXT文件失败: {e}")
                return ""
        except Exception as e:
            logging.error(f"读取TXT文件失败: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        将长文本分割成带有重叠部分的块。
        重叠是为了保证语义的连续性，避免信息在块的边界被切断。
        :param text: 待分割的文本。
        :param chunk_size: 每个块的目标大小。
        :param overlap: 相邻块之间的重叠字数。
        :return: 文本块列表。
        """
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

# --- 向量存储 ---
class VectorStore:
    """
    管理FAISS索引、文本嵌入和文档映射关系。
    这是RAG核心组件，负责将文本转换为向量并提供高效的检索服务。
    """
    def __init__(self, model):
        self.model = model
        self.index = None
        self.document_to_chunks = {}
        self.chunks_to_document = {}
        self.all_chunks = []

    def build_index(self, documents: Dict[str, Dict], doc_processor: DocumentProcessor):
        """
        从文档构建或重建FAISS索引。
        这是一个开销较大的操作，包括读取文件、分块、生成嵌入和构建索引。
        在文档数量多或文档大时，建议作为后台任务执行。
        """
        # 重置数据
        self.document_to_chunks = {}
        self.chunks_to_document = {}
        self.all_chunks = []
        
        for doc_id, doc_data in documents.items():
            content = doc_processor.read_document(doc_data["path"])
            chunks = doc_processor.chunk_text(content)
            
            chunk_ids = []
            for chunk in chunks:
                chunk_id = len(self.all_chunks)
                self.all_chunks.append(chunk)
                self.chunks_to_document[chunk_id] = doc_id
                chunk_ids.append(chunk_id)
            self.document_to_chunks[doc_id] = chunk_ids
        
        if not self.all_chunks:
            self.index = None
            return

        chunk_embeddings = self._get_embeddings(self.all_chunks)
        dimension = chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(chunk_embeddings)

        self.save_store()

    def search(self, query: str, k: int = 3) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        在向量存储中搜索与查询最相关的k个文档块。
        :param query: 用户查询字符串。
        :param k: 要检索的块数量。
        :return: (相关文档ID列表, (文档ID, 块内容)元组列表)
        """
        if self.index is None or not self.all_chunks:
            return [], []
            
        query_embedding = self._get_embeddings([query])
        distances, chunk_indices = self.index.search(query_embedding, k)
        
        retrieved_doc_ids = set()
        retrieved_chunks = []
        
        for chunk_idx in chunk_indices[0]:
            if 0 <= chunk_idx < len(self.all_chunks):
                doc_id = self.chunks_to_document.get(int(chunk_idx))
                if doc_id:
                    retrieved_doc_ids.add(doc_id)
                    retrieved_chunks.append((doc_id, self.all_chunks[int(chunk_idx)]))
        
        return list(retrieved_doc_ids), retrieved_chunks

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        使用预加载的sentence-transformer模型为文本列表生成嵌入向量。
        `normalize_embeddings=True` 对于使用L2距离(IndexFlatL2)的FAISS很关键。
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)

    def save_store(self):
        """保存FAISS索引和映射数据"""
        if self.index:
            faiss.write_index(self.index, config.FAISS_INDEX_PATH)
        mapping_data = {
            'doc_to_chunks': self.document_to_chunks,
            'chunks_to_doc': self.chunks_to_document,
            'all_chunks': self.all_chunks
        }
        np.save(config.CHUNKS_MAPPING_PATH, mapping_data)

    def load_store(self):
        """加载FAISS索引和映射数据"""
        if os.path.exists(config.FAISS_INDEX_PATH):
            self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        if os.path.exists(config.CHUNKS_MAPPING_PATH):
            mapping_data = np.load(config.CHUNKS_MAPPING_PATH, allow_pickle=True).item()
            self.document_to_chunks = mapping_data.get('doc_to_chunks', {})
            self.chunks_to_document = mapping_data.get('chunks_to_doc', {})
            self.all_chunks = mapping_data.get('all_chunks', [])


# --- 文档管理器 ---
class DocumentManager:
    """
    管理上传的文档元数据，如文件名、存储路径等。
    同时负责将文档元数据持久化到JSON文件中。
    """
    def __init__(self, docs_dir: str, index_path: str):
        self.docs_dir = docs_dir
        self.index_path = index_path
        self.uploaded_documents: Dict[str, Dict] = {}
        os.makedirs(self.docs_dir, exist_ok=True)

    def add_document(self, file: UploadFile, docs_dir: str) -> Tuple[str, str]:
        """
        保存上传的文件到指定目录，并返回其唯一ID和路径。
        使用UUID确保文件名唯一，避免冲突。
        """
        doc_id = str(uuid.uuid4())
        file_path = os.path.join(docs_dir, f"{doc_id}_{file.filename}")
        
        try:
            with open(file_path, "wb") as f:
                f.write(file.file.read())
        except IOError as e:
            logging.error(f"保存文件失败: {e}")
            raise HTTPException(status_code=500, detail="文件保存失败")
            
        self.uploaded_documents[doc_id] = {
            "name": file.filename,
            "path": file_path
        }
        return doc_id, file_path
        
    def delete_document(self, doc_id: str):
        """删除文档"""
        if doc_id in self.uploaded_documents:
            doc_path = self.uploaded_documents[doc_id]["path"]
            if os.path.exists(doc_path):
                os.remove(doc_path)
            del self.uploaded_documents[doc_id]

    def save_to_json(self):
        """将文档索引保存到JSON文件"""
        serializable_docs = {
            doc_id: {"name": data["name"], "path": data["path"]}
            for doc_id, data in self.uploaded_documents.items()
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

    def load_from_json(self):
        """从JSON文件加载文档索引"""
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                serialized_docs = json.load(f)
            for doc_id, doc_data in serialized_docs.items():
                if os.path.exists(doc_data["path"]):
                    self.uploaded_documents[doc_id] = doc_data
        except Exception as e:
            logging.error(f"加载文档索引失败: {e}")

# 创建应用启动上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期事件管理器。
    在应用启动时执行初始化操作，在关闭时执行清理操作。
    """
    # 启动前执行
    logging.info("应用启动中...")
    init()
    db_manager.init_db()
    doc_manager.load_from_json()
    vector_store.load_store()
    # 如果没有加载到索引，或者文档列表和索引内容不一致，则重建索引
    if not vector_store.index or len(doc_manager.uploaded_documents) != len(vector_store.document_to_chunks):
        logging.info("未找到有效索引或文档已更新，开始重建索引...")
        vector_store.build_index(doc_manager.uploaded_documents, doc_processor)
        logging.info("索引重建完成。")
    logging.info("应用启动完成。")
    
    yield
    
    # 关闭时执行
    logging.info("应用关闭中...")
    doc_manager.save_to_json()
    # vector_store.save_store() # 在build_index时已保存，此处可省略
    db_manager.close()
    logging.info("应用已关闭。")

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan)
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# 添加CORS中间件允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model = None
client = None
doc_processor = DocumentProcessor()
vector_store = None # will be initialized in init()
doc_manager = DocumentManager(config.DOCS_DIR, config.DOCUMENTS_INDEX_PATH)

def init():
    """
    初始化应用所需的全局资源，如LLM客户端和嵌入模型。
    """
    global model, client, vector_store
    
    # 初始化LLM客户端
    logging.info("初始化LLM客户端...")
    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.BASE_URL
    )
    
    # 加载或下载嵌入模型
    logging.info("加载嵌入模型...")
    local_model_path = config.EMBEDDING_MODEL_PATH
    if os.path.exists(local_model_path):
        model = SentenceTransformer(local_model_path)
    else:
        logging.info(f"本地未找到模型，正在从Hugging Face下载 {config.EMBEDDING_MODEL_NAME}...")
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        model.save(local_model_path)
    logging.info("嵌入模型加载完成。")
    
    # 实例化向量存储
    vector_store = VectorStore(model)

def retrieve_docs(query, k=3) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    封装了向量存储的搜索功能，并格式化输出。
    :param query: 用户查询。
    :param k: 检索数量。
    :return: (文档名列表, (文档ID, 块内容)元组列表)
    """
    doc_ids, chunks = vector_store.search(query, k)
    # 根据检索到的文档ID，从文档管理器中获取原始文档名
    retrieved_docs = [f"文档: {doc_manager.uploaded_documents[doc_id]['name']}" for doc_id in doc_ids]
    return retrieved_docs, chunks

# 定义重建索引的后台任务
def background_rebuild_index():
    """
    封装索引重建逻辑，以便在后台任务中调用。
    """
    logging.info("后台任务：开始重建索引...")
    vector_store.build_index(doc_manager.uploaded_documents, doc_processor)
    logging.info("后台任务：索引重建完成。")

# 文档管理 API
@app.post("/api/upload", response_model=UploadSuccessResponse)
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    上传文档接口。
    支持 .txt, .pdf, .docx 格式。
    文件上传后，会触发一个后台任务来异步重建向量索引，避免阻塞API。
    """
    if not file.filename.endswith((".txt", ".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="仅支持.txt, .pdf, .docx文件")
    
    doc_id, _ = doc_manager.add_document(file, config.DOCS_DIR)
    
    # 将索引重建任务添加到后台执行
    background_tasks.add_task(background_rebuild_index)
    
    return {"message": "文件上传成功，正在后台处理...", "doc_id": doc_id}

@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents():
    """获取所有已上传文档的列表。"""
    docs = [{"id": doc_id, "name": data["name"]} for doc_id, data in doc_manager.uploaded_documents.items()]
    return {"documents": docs}

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    删除指定ID的文档。
    删除后同样会触发后台任务重建索引。
    """
    if doc_id not in doc_manager.uploaded_documents:
        raise HTTPException(status_code=404, detail="文档未找到")
        
    doc_manager.delete_document(doc_id)
    
    # 将索引重建任务添加到后台执行
    background_tasks.add_task(background_rebuild_index)
    
    return {"message": "文档删除成功，正在后台更新索引..."}

@app.post("/api/stream")
async def stream_post(request: ChatStreamRequest):
    """
    处理流式聊天请求 (POST)。
    使用Pydantic模型自动验证请求体。
    """
    try:
        return await process_stream_request(request.query, request.session_id, request.web_search)
    except Exception as e:
        error_msg = str(e)
        logging.error(f"聊天接口错误: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/stream")
async def stream_get(query: str = Query(None), session_id: str = Query(None), web_search: bool = Query(False)):
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Missing query parameter")
        return await process_stream_request(query, session_id, web_search)
    except Exception as e:
        error_msg = str(e)
        logging.error(f"聊天接口错误: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
 
# 执行网络搜索
async def perform_web_search(query: str):
    try:
        query = urllib.parse.quote(query)
        api_key = config.SEARCH_API_KEY
        search_engine_id = config.SEARCH_ENGINE_ID
        
        # 检查是否配置了有效的API密钥
        if api_key == "your_google_search_api_key" or search_engine_id == "your_google_search_engine_id":
            logging.warning("未配置网络搜索API Key或搜索引擎ID，将跳过网络搜索。")
            return "网络搜索功能未配置。如需使用此功能，请配置有效的Google Custom Search API密钥和搜索引擎ID。"
            
        search_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}&start=0"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 403:
            logging.error("Google Custom Search API返回403错误，可能是API密钥无效或权限不足")
            return "网络搜索暂时不可用，请检查API配置。"
        elif response.status_code != 200:
            logging.error(f"Google搜索失败，状态码: {response.status_code}")
            return f"网络搜索失败，状态码: {response.status_code}"
        
        data = response.json()
        logging.info(f"Google search response: {data}")
        
        # 解析搜索结果
        if 'items' in data and data['items']:
            results = []
            for item in data['items'][:3]:  # 只取前3个结果
                title = item.get('title', '无标题')
                snippet = item.get('snippet', '无摘要')
                results.append(f"标题: {title}\n摘要: {snippet}")
            return "\n\n".join(results)
        else:
            return "未找到相关搜索结果。"
            
    except Exception as e:
        logging.error(f"执行网络搜索时出错: {str(e)}")
        return f"网络搜索出错: {str(e)}"

async def process_stream_request(query: str, session_id: str = None, web_search: bool = False):
    logging.info(f"收到请求: query='{query}', session_id='{session_id}', web_search={web_search}")
    
    # 检查会话是否存在，如果不存在则生成新的session_id
    has_session = session_id and db_manager.session_exists(session_id)
    if not has_session:
        session_id = str(uuid.uuid4())
        logging.info(f"新会话，生成ID: {session_id}")
    
    # 1. 构建上下文 (Context)
    context_parts = []
    
    # 1a. (可选) 网络搜索
    if web_search:
        logging.info("执行网络搜索...")
        web_results = await perform_web_search(query)
        context_parts.append(f"网络搜索结果:\n{web_results}")
    
    # 1b. 文档检索
    if doc_manager.uploaded_documents:
        logging.info("执行文档检索...")
        retrieved_docs, retrieved_chunks = retrieve_docs(query, k=3)
        if retrieved_docs:
            # 添加相关文档名列表
            context_parts.append("从您的文档中找到以下相关信息:\n" + "\n".join(retrieved_docs))
            
            # 添加具体的文档内容片段
            chunk_context = "\n\n相关内容片段:\n"
            for i, (doc_id, chunk) in enumerate(retrieved_chunks):
                doc_name = doc_manager.uploaded_documents.get(doc_id, {}).get("name", "未知文档")
                chunk_context += f"- [片段 {i+1} 来自: {doc_name}]\n{chunk}\n"
            context_parts.append(chunk_context)
        else:
            context_parts.append("\n在您的文档中未找到直接相关的内容。")
    else:
        logging.info("知识库为空，跳过文档检索。")

    # 2. 组合最终的 Prompt
    context = "\n---\n".join(context_parts)
    # 根据是否有上下文，决定prompt模板
    if context:
        prompt = f"请参考以下信息来回答用户的问题。\n\n[参考信息]\n{context}\n\n[用户问题]: {query}"
    else:
        prompt = query # 如果没有任何上下文，直接使用用户问题

    logging.info(f"最终发送给LLM的Prompt:\n---\n{prompt}\n---")

    # 3. 创建流式响应生成器
    async def generate():
        nonlocal has_session
        full_response = ""
        
        system_message = "你是一个专业的问答助手。"
        
        if web_search:
            system_message += "你拥有联网搜索能力，可以提供最新的信息。"
        else:
            system_message += "请仅基于提供的上下文信息回答问题，不要添加任何未在上下文中提及的信息。"
            
        system_message += "如果没有相关信息，请明确告知用户无法回答该问题。"
        
        stream = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {json.dumps({'content': content, 'session_id': session_id})}\n\n"
                await asyncio.sleep(0.01)  # 添加小延迟确保流式输出
                
            if chunk.choices[0].finish_reason is not None:
                yield f"data: {json.dumps({'content': '', 'session_id': session_id, 'done': True})}\n\n"
                break
                
        # 响应完成后，将完整会话保存到数据库
        # 必须在生成器函数内部处理，以确保流式传输结束后执行
        logging.info("LLM响应完成，开始保存会话...")
        if has_session:
            await add_message_to_session(session_id, query, full_response)
            logging.info(f"消息已添加到现有会话: {session_id}")
        else:
            await create_new_chat_session(session_id, query, full_response)
            logging.info(f"新会话已创建并保存: {session_id}")
            
    return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )

# 创建新的聊天会话
async def create_new_chat_session(session_id, query, response):
    # 创建会话摘要
    try:
        summary_prompt = f"请为以下对话创建一个简洁的摘要，作为会话标题（10个字以内）:\n用户: {query}\n助手: {response}"
        summary_response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=30  # 限制摘要长度
        )
        summary = summary_response.choices[0].message.content.strip()
    except Exception as e:
        summary = query[:20]  # 如果生成摘要失败，使用问题的前20个字符
        logging.error(f"生成会话摘要失败: {e}")

    db_manager.create_chat_session(session_id, summary)
    db_manager.add_message(session_id, "user", query)
    db_manager.add_message(session_id, "assistant", response)

# 向现有会话添加消息
async def add_message_to_session(session_id, query, response):
    # 直接将会话ID、问题和回答添加到数据库
    db_manager.add_message(session_id, "user", query)
    db_manager.add_message(session_id, "assistant", response)
    


@app.get("/api/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history():
    """获取所有聊天会话的历史记录"""
    try:
        history_data = db_manager.get_chat_history()
        return {"history": history_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取聊天记录失败: {str(e)}")

@app.get("/api/chat/session/{session_id}", response_model=SessionMessagesResponse)
async def get_session(session_id: str):
    """获取特定会话的消息历史，并按前端格式要求转换。"""
    try:
        messages = db_manager.get_session_messages(session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="会话未找到")
        
        return {"messages": messages}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话失败: {str(e)}")

@app.delete("/api/chat/session/{session_id}")
async def delete_session(session_id: str):
    """删除一个聊天会话"""
    try:
        db_manager.delete_session(session_id)
        return {"status": "success", "message": "会话已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")

# 健康检查接口
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 运行服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)