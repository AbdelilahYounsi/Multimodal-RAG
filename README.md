# 🤖 Multimodal RAG with Gemini + Whisper + Milvus

This project is a **multimodal retrieval-augmented generation (RAG)** system that integrates **Google Gemini**, **OpenAI Whisper**, and **Milvus** to enable **question answering over text, PDFs, and audio**.
It provides a simple **Streamlit** interface to ingest data and query it using either **text** or **audio input**.

---

## 🚀 Features

* **Multimodal ingestion:** Extracts and embeds text from `.pdf`, `.txt`, `.mp3`, and `.wav` files.
* **Vector database:** Uses **Milvus** for fast semantic search.
* **Gemini models:** Handles both embeddings and response generation.
* **Audio transcription:** Converts audio queries or documents into text with **Whisper**.
* **Streamlit UI:** Clean, interactive web interface for ingestion and querying.

---

## 📦 Project Structure

```
.
├── app.py                 # Streamlit UI
├── rag_pipeline.py        # Core RAG logic (Ingestion + Query Flows)
├── config.py              # Configuration (paths, model names, Milvus setup)
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Milvus setup
└── README.md              # Documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AbdelilahYounsi/Multimodal-RAG.git
cd Multimodal-RAG
```

### 2. Start Milvus with Docker Compose

Ensure Docker and Docker Compose are installed, then run:

```bash
docker-compose up -d
```

This will start:

* **Milvus standalone server**
* **etcd** and **minio** dependencies


You can verify it’s running:

```bash
docker ps
```

---

### 3. Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### 4. Configure environment

Edit the `config.py` file to match your setup:

```python
DATA_DIR = "data"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_collection"
EMBEDDING_DIM = 768
EMBEDDING_MODEL = "models/embedding-001"
GEMINI_MODEL = "gemini-2.0-flash"
```

---

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Then open:
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧠 How It Works

### Step 1 — Data Ingestion

1. Place your `.pdf`, `.txt`, `.mp3`, or `.wav` files in the directory defined in `config.DATA_DIR`.
2. In the sidebar, click **🚀 Ingest Data**.
3. The ingestion flow will:

   * Extract text from files
   * Split text into 1000-character chunks
   * Generate embeddings using Gemini
   * Store everything in Milvus

### Step 2 — Query

1. Enter your **Gemini API key** in the sidebar.
2. Choose between **Text** or **Audio** query.
3. Click **🔍 Search**.
4. The app will:

   * Transcribe audio (if applicable)
   * Generate query embeddings
   * Retrieve top-5 similar chunks from Milvus
   * Generate a synthesized response using Gemini

---

## 🧰 Technologies Used

| Component      | Purpose                   | Library               |
| -------------- | ------------------------- | --------------------- |
| **Streamlit**  | Web UI                    | `streamlit`           |
| **Gemini API** | Embeddings + generation   | `google-generativeai` |
| **Whisper**    | Audio transcription       | `openai-whisper`      |
| **Milvus**     | Vector storage            | `pymilvus`            |
| **CrewAI**     | Multi-agent orchestration | `crewai`              |
| **PyPDF2**     | PDF text extraction       | `PyPDF2`              |

---

## ⚠️ Notes

* The Whisper model (`small`) is loaded locally — the first load may take time.
* Re-ingesting data will recreate the Milvus collection (erasing old data).
* Supported formats: `.pdf`, `.txt`, `.mp3`, `.wav`.

---

## 🧹 Stop Services

When done, stop Milvus:

```bash
docker-compose down
```

---


