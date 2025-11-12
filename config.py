# config.py
import os

# Milvus settings
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "simple_rag"
EMBEDDING_DIM = 768  # Gemini embedding dimension

# Data directory
DATA_DIR = "./data"

# Model settings
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"