# app.py
import streamlit as st
import tempfile
import os
from pathlib import Path

from flows import IngestionFlow, MultimodalRAGFlow
import config

st.set_page_config(page_title="Local Multimodal RAG", page_icon="🤖", layout="wide")

# Initialize session state
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False

st.title("🤖 Local Multimodal RAG (Powered by llama.cpp)")

# Sidebar for setup
with st.sidebar:
    st.header("⚙️ Setup")
    
    st.markdown("---")
    st.header("📁 Data Ingestion")
    st.info(f"Place your files in `{config.DATA_DIR}` folder")
    st.caption("Supported: PDF, MP3, WAV, TXT")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Ingest Data", disabled=st.session_state.ingestion_done):
            with st.spinner("Processing files..."):
                try:
                    flow = IngestionFlow()
                    state1 = flow.load_files()
                    state2 = flow.process_files(state1)
                    state3 = flow.setup_milvus(state2)
                    flow.embed_and_store(state3)
                    
                    st.session_state.ingestion_done = True
                    st.success(f"✅ Ingested {len(state2.chunks)} chunks!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if st.session_state.ingestion_done:
            if st.button("🔄 Re-ingest", help="Reset and ingest files again"):
                st.session_state.ingestion_done = False
                st.rerun()
    
    if st.session_state.ingestion_done:
        st.success("✅ System Ready!")

# Main query interface
if not st.session_state.ingestion_done:
    st.warning("👈 Please ingest data first using the sidebar")
else:
    st.header("💬 Ask Questions")
    
    # Query type selection
    query_type = st.radio("Query Type:", ["Text", "Audio"], horizontal=True)
    
    if query_type == "Text":
        text_query = st.text_input("Enter your question:", key="text_query")
        audio_query = None
    else:
        audio_query = st.audio_input("Record query", key = "audio_query")
        text_query = ""
    
    if st.button("🔍 Search", type="primary"):
        if text_query or audio_query:
            with st.spinner("Processing..."):
                try:
                    # Run query flow
                    flow = MultimodalRAGFlow()
                    state1 = flow.transcribe_audio_if_needed(text_query, audio_query)
                    state2 = flow.search_knowledge_base(state1)
                    state3 = flow.generate_response(state2)
                    
                    # Display results
                    st.success("✅ Done!")

                    if not state3.response:
                        st.warning("⚠️ Response is empty. Check logs for errors.")
                    else:
                        st.markdown("### 🤖 Answer")
                        st.markdown(state3.response)
                    
                    with st.expander("📚 Retrieved Context"):
                        st.text(state3.results)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question or upload an audio file")

# Footer
st.markdown("---")
st.caption("Local Multimodal RAG with llama.cpp (Qwen + Voxtral) + Milvus")