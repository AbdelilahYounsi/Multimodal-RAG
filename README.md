# 🤖 Multimodal RAG with Gemini + Whisper + Milvus

This project is a **multimodal retrieval-augmented generation (RAG)** system that integrates **Google Gemini**, **OpenAI Whisper**, and **Milvus** to enable **question answering over text, PDFs, and audio**.  
It provides a simple **Streamlit** interface to ingest data and query it using either **text** or **audio input**.

---

## 🚀 Features

- **Multimodal data ingestion:** Automatically extracts and embeds text from `.pdf`, `.txt`, `.mp3`, and `.wav` files.  
- **Vector storage with Milvus:** Uses Milvus as a scalable vector database to store and search embeddings.  
- **Gemini embeddings and generation:** Employs Gemini models for embedding and answer synthesis.  
- **Audio transcription:** Converts speech queries or audio documents into text using **Whisper**.  
- **Interactive Streamlit app:** Allows ingestion, search, and QA directly from the browser.

---

## 🧩 Architecture Overview

