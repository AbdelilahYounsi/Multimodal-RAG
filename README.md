# Local Multimodal RAG

A local, privacy-first Retrieval-Augmented Generation (RAG) system powered by **llama.cpp**, **Qwen**, **Voxtral**, and **Milvus**.

## Features

 **Multimodal Input Support**
- 📄 PDF documents
- 📝 Text files
- 🎵 Audio files (MP3, WAV)

 **Local Processing**
- No API calls, complete privacy
- Runs entirely on your machine
- Uses open-source models

 **Audio Transcription**
- Voxtral-Mini-3B for accurate audio-to-text conversion (can use whisper or any other model)
- Support for multiple audio formats

 **Vector Search**
- Milvus vector database for semantic search
- Qwen 3 embedding model (0.6B) (any other model)
- COSINE similarity matching

✨ **Response Generation**
- Qwen3-VL-4B for intelligent answer generation
- Context-aware responses based on retrieved documents

## Prerequisites

- **Python 3.10+**
- **llama.cpp** (with build compiled)
- **Milvus** 

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/AbdelilahYounsi/Multimodal-RAG.git
cd Multimodal-RAG
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download models** (if not already done)
- Place models in `..\llama.cpp\models\` directory
- Required models: (or any other models)
  - `Voxtral-Mini-3B-2507-Q4_K_M.gguf`
  - `mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf`
  - `qwen3-embedding-0.6B.gguf`
  - `Qwen3-VL-4B-Instruct-Q4_K_M.gguf`

5. **Start Milvus**
```bash
docker compose up 
```

## Configuration

Update `config.py` with your system paths:

```python
# Local llama.cpp paths
LLAMA_MTMD_CLI_PATH = r"path\to\llama-mtmd-cli"
LLAMA_CLI_PATH = r"path\to\llama-cli"
LLAMA_EMB_PATH = r"path\to\llama-embedding"

# Model paths
VOXTRAL_MODEL = r"path\to\Voxtral-Mini-3B.gguf"
EMBEDDING_MODEL_PATH = r"path\to\qwen3-embedding-0.6B.gguf"
TEXT_LLM_MODEL_PATH = r"path\to\Qwen3-VL-4B-Instruct.gguf"
```

## Usage

1. **Add your documents**
   - Place files in `./data` folder
   - Supported: PDF, TXT, MP3, WAV

2. **Run the app**
```bash
streamlit run app.py
```

3. **Ingest data**
   - Click "🚀 Ingest Data" in the sidebar
   - Wait for processing to complete

4. **Ask questions**
   - Choose query type: Text or Audio
   - Enter your question
   - Click "🔍 Search"

## Project Structure

```
multimodal-rag/
├── app.py                 # Streamlit UI
├── flows.py              # CrewAI flows for ingestion & RAG
├── config.py             # Configuration settings
├── utils.py              # Helper functions for transcription, embedding and response generation
├── requirements.txt      # Python dependencies
├── data/                 # Your documents folder
├── ingested_files.txt    # Tracking ingested files
└── README.md
```

## How It Works

### Ingestion Pipeline
1. **Load Files** - Scans `./data` for supported file types
2. **Process Content** - Extracts text from PDFs, transcribes audio
3. **Chunk Data** - Splits documents into manageable chunks
4. **Generate Embeddings** - Creates vector representations using Qwen embedding model
5. **Store in Milvus** - Indexes vectors for semantic search

### RAG Pipeline
1. **Transcribe Query** - Converts audio queries to text 
2. **Search** - Finds semantically similar chunks from knowledge base
3. **Generate Response** - Uses Qwen3-VL-4B to generate context-aware answers


## Troubleshooting

**Models not found?**
- Ensure llama.cpp is built: `cd ../llama.cpp && cmake --build build`
- Verify model paths in `config.py`

**Milvus connection failed?**
- Check if Milvus is running: `docker ps`
- Restart Milvus: `docker restart milvus`

**Response is empty?**
- Check terminal logs for errors
- Ensure ingestion completed successfully
- Try re-ingesting with fresh data

## License

MIT License - See LICENSE file for details




