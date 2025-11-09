# app.py
import streamlit as st
import tempfile
import os
from pathlib import Path
import google.generativeai as genai

from rag_pipeline import IngestionFlow, MultimodalRAGFlow
import config

st.set_page_config(page_title="Simple RAG", page_icon="🤖", layout="wide")

# Initialize session state
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False

st.title("🤖 Simple Multimodal RAG")

# Sidebar for API key and setup
with st.sidebar:
    st.header("⚙️ Setup")
    
    api_key = st.text_input("Gemini API Key", type="password", key="api_key")
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API Key configured!")
        
        st.markdown("---")
        st.header("📁 Data Ingestion")
        st.info(f"Place your files in `{config.DATA_DIR}` folder")
        
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
        
        if st.session_state.ingestion_done:
            st.success("✅ System Ready!")

# Main query interface
if not api_key:
    st.warning("👈 Please enter your Gemini API key in the sidebar")
elif not st.session_state.ingestion_done:
    st.warning("👈 Please ingest data first using the sidebar")
else:
    st.header("💬 Ask Questions")
    
    # Query type selection
    query_type = st.radio("Query Type:", ["Text", "Audio"], horizontal=True)
    
    if query_type == "Text":
        query = st.text_input("Enter your question:", key="text_query")
        audio_file = None
    else:
        audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])
        query = ""
    
    if st.button("🔍 Search", type="primary"):
        if query or audio_file:
            with st.spinner("Processing..."):
                try:
                    # Save audio if uploaded
                    temp_audio = None
                    if audio_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                            f.write(audio_file.read())
                            temp_audio = f.name
                    
                    # Run query flow
                    flow = MultimodalRAGFlow(api_key)
                    state1 = flow.transcribe_audio_if_needed(query, temp_audio)
                    state2 = flow.search_knowledge_base(state1)
                    state3 = flow.generate_response(state2)
                    
                    # Display results
                    st.success("✅ Done!")
                    
                    if temp_audio:
                        st.info(f"**Transcribed Query:** {state3.query}")
                    
                    st.markdown("### 🤖 Answer")
                    st.markdown(state3.response)
                    
                    with st.expander("📚 Retrieved Context"):
                        st.text(state3.results)
                    
                    # Cleanup
                    if temp_audio:
                        try:
                            os.unlink(temp_audio)
                        except:
                            pass
                
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question or upload an audio file")

# Footer
st.markdown("---")
st.caption("Multimodal RAG with Gemini 2.5 Flash + Whisper + Milvus")