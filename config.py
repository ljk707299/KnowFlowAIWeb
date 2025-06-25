import os

# OpenAI API
OPENAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "your_default_api_key")
BASE_URL = os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

# Embedding Model
EMBEDDING_MODEL_PATH = 'local_m3e_model'
EMBEDDING_MODEL_NAME = 'moka-ai/m3e-base'

# Vector Store
FAISS_INDEX_PATH = "m3e_faiss_index.bin"
CHUNKS_MAPPING_PATH = "chunks_mapping.npy"
DOCUMENTS_INDEX_PATH = "docs/documents_index.json"
DOCS_DIR = "docs"

# Database
DB_PATH = 'chat_history.db'

# Web Search
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "your_google_search_api_key")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "your_google_search_engine_id") 