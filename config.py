# config.py
import os

# Milvus settings
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "simple_rag"
EMBEDDING_DIM = 1024  

# Data directory
DATA_DIR = "./data"

# Local llama.cpp settings
LLAMA_MTMD_CLI_PATH = r"..\llama.cpp\build\bin\llama-mtmd-cli"  
LLAMA_CLI_PATH = r"..\llama.cpp\build\bin\llama-cli"  
LLAMA_EMB_PATH = r"..\llama.cpp\build\bin\llama-embedding"  
VOXTRAL_MODEL = r"..\llama.cpp\models\voxtral\Voxtral-Mini-3B-2507-Q4_K_M.gguf"
VOXTRAL_MMPROJ = r"..\llama.cpp\models\voxtral\mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf"

# Local embedding model
EMBEDDING_MODEL_PATH = r"..\llama.cpp\models\qwen\qwen3-embedding-0.6B.gguf" 
TEXT_LLM_MODEL_PATH = r"..\llama.cpp\models\qwen\Qwen3-VL-4B-Instruct-Q4_K_M.gguf"  