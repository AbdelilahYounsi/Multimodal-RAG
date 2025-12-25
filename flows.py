import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from PyPDF2 import PdfReader
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai.flow.flow import Flow, start, listen
import config
from utils import transcribe_audio, generate_embeddings, generate_response_llm, get_ingested_files, save_ingested_file
import tempfile 
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IngestionState:
    files: List[str] = None
    chunks: List[Dict] = None
    collection: Collection = None


@dataclass
class QueryState:
    query: str = ""
    results: str = ""
    response: str = ""


class IngestionFlow(Flow):
    """Simple data ingestion flow"""
    
    @start()
    def load_files(self):
        """Load only new files"""
        logger.info("Loading files...")
        data_dir = Path(config.DATA_DIR)
        ingested = get_ingested_files()
        
        files = []
        for ext in ["*.pdf", "*.mp3", "*.wav", "*.txt"]:
            for file_path in data_dir.glob(ext):
                if file_path.name not in ingested:
                    files.append(str(file_path))
        
        logger.info(f"Found {len(files)} new files")
        return IngestionState(files=files)
    
    @listen(lambda s: s.chunks)
    def process_files(self, state: IngestionState):
        """Extract text from files"""
        logger.info("Processing files...")
        chunks = []
        
        for file_path in state.files:
            file_path = Path(file_path)
            
            try:
                if file_path.suffix == '.pdf':
                    reader = PdfReader(file_path)
                    text = "\n".join(page.extract_text() for page in reader.pages)
                elif file_path.suffix in ['.mp3', '.wav']:
                    # Use local llama.cpp model for transcription
                    text = transcribe_audio(str(file_path))
                elif file_path.suffix == '.txt':
                    text = file_path.read_text()
                else:
                    continue
                
                # Simple chunking
                for i in range(0, len(text), 1000):
                    chunk = text[i:i+1000].strip()
                    if chunk:
                        chunks.append({"text": chunk, "source": file_path.name})
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
        
        logger.info(f"Created {len(chunks)} chunks")
        return IngestionState(files=state.files, chunks=chunks)
    
    @listen(lambda s: s.chunks)
    def setup_milvus(self, state: IngestionState):
        """Setup Milvus collection"""
        logger.info("Setting up Milvus...")
        connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        
        if not utility.has_collection(config.COLLECTION_NAME):
            fields = [
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("source", DataType.VARCHAR, max_length=500),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)
            ]
            schema = CollectionSchema(fields)
            collection = Collection(config.COLLECTION_NAME, schema)
            collection.create_index("embedding", {"index_type": "FLAT", "metric_type": "COSINE"})
        else:
            collection = Collection(config.COLLECTION_NAME)
        
        collection.load()
        logger.info("Milvus setup complete")
        return IngestionState(files=state.files, chunks=state.chunks, collection=collection)
        
    @listen(lambda s: s.collection)
    def embed_and_store(self, state: IngestionState):
        """Generate embeddings and store in Milvus"""
        logger.info("Generating embeddings with local LLM...")
        
        texts = [c["text"] for c in state.chunks]
        sources = [c["source"] for c in state.chunks]
        
        # Generate embeddings using local model
        if texts:
            embeddings = generate_embeddings(texts)
            
            for chunk in state.chunks:
                save_ingested_file(chunk["source"])
            
            # Insert into Milvus
            data = [texts, sources, embeddings]
            state.collection.insert(data)
            logger.info(f"Stored {len(texts)} chunks")
        else:
            logger.info("No new chunks to store")
        
        state.collection.flush()
        state.collection.load()
        
        return state


class MultimodalRAGFlow(Flow):
    """Multimodal RAG query flow with local llama.cpp models"""
    
    def __init__(self):
        super().__init__()
        connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        self.collection = Collection(config.COLLECTION_NAME)
        self.collection.load()
    
    @start()
    def transcribe_audio_if_needed(self, query: str, audio):
        """Transcribe audio if query is audio-based"""
        if audio:
            logger.info("Transcribing audio with local llama.cpp model...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav',dir='./') as tmp:
                tmp.write(audio.getvalue())
                audio_path = tmp.name

            try:
                transcribed_query = transcribe_audio(audio_path)
                logger.info(f"Transcribed: {transcribed_query}")
            finally:
                os.remove(audio_path)
        else:
            transcribed_query = query
            logger.info("Using text query")
        
        return QueryState(query=transcribed_query)
    
    @listen(lambda s: s.query)
    def search_knowledge_base(self, state: QueryState):
        """Search the vector database for relevant information"""
        logger.info("Searching knowledge base...")
        
        # Generate query embedding using local model
        embeddings = generate_embeddings([state.query])
        query_embedding = embeddings[0]
        
        # Search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=5,
            output_fields=["text", "source"]
        )
        
        # Format results
        formatted = []
        for hit in results[0]:
            relevance = hit.score * 100
            formatted.append(
                f"Source: {hit.entity.get('source')}\n"
                f"Relevance: {relevance:.1f}%\n"
                f"Content: {hit.entity.get('text')}\n"
                f"---"
            )
        
        results_text = "\n".join(formatted)
        logger.info("Search completed")
        
        return QueryState(query=state.query, results=results_text)
    
    @listen(lambda s: s.results)
    def generate_response(self, state: QueryState):
        """Generate final response using local llama.cpp LLM"""
        logger.info("Generating response with local LLM...")
        
        try:
            # Generate response using local LLM
            response_text = generate_response_llm(state.query, state.results)
            
            logger.info("Response generated")
            return QueryState(
                query=state.query,
                results=state.results,
                response=response_text
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return QueryState(
                query=state.query,
                results=state.results,
                response=f"Error: {str(e)}"
            )