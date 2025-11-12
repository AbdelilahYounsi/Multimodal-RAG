# rag_pipeline.py
import glob
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import google.generativeai as genai
import whisper
from PyPDF2 import PdfReader
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai.flow.flow import Flow, start, listen

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global whisper model
_whisper_model = None


def get_whisper_model():
    """Load whisper model once"""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("small")
    return _whisper_model

# Keep track of already ingested files
INGESTED_FILES = Path("ingested_files.txt")

def get_ingested_files() -> set:
    """Load list of already ingested files"""
    if INGESTED_FILES.exists():
        return set(INGESTED_FILES.read_text().splitlines())
    return set()

def save_ingested_file(filename: str):
    """Append filename to ingested list"""
    with open(INGESTED_FILES, "a") as f:
        f.write(f"{filename}\n")


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


class SearchKnowledgeBaseTool(BaseTool):
    """Tool for searching the knowledge base"""
    
    name: str = "search_knowledge_base"
    description: str = "Search the multimodal knowledge base for relevant information"
    collection: Collection = None
    
    def _run(self, query: str) -> str:
        """Search the knowledge base and return formatted results"""
        logger.info(f"Tool searching for: {query}")
        
        # Generate query embedding
        result = genai.embed_content(
            model=config.EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = result['embedding']
        
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
                f"Content: {hit.entity.get('text')[:500]}...\n"
            )
        
        return "\n---\n".join(formatted)


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
    
    @listen(lambda s: s.files)
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
                    model = get_whisper_model() 
                    result = model.transcribe(str(file_path))
                    text = result["text"]
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
        logger.info("Generating embeddings...")
        
        texts = [c["text"] for c in state.chunks]
        sources = [c["source"] for c in state.chunks]
        
        # Batch embeddings
        embeddings = []
        if texts:
            for i in range(0, len(texts), 10):
                batch = texts[i:i+10]
                result = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=batch,
                    task_type="retrieval_document"
                )
                embeddings.extend(result['embedding'])
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
    """Multimodal RAG query flow with Gemini 2.5 Flash"""
    
    def __init__(self, api_key: str):
        super().__init__()
        genai.configure(api_key=api_key)
        connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        self.collection = Collection(config.COLLECTION_NAME)
        self.collection.load()
    
    @start()
    def transcribe_audio_if_needed(self, query: str, audio_file: Optional[str] = None):
        """Transcribe audio if query is audio-based"""
        if audio_file:
            logger.info("Transcribing audio...")
            model = get_whisper_model()
            result = model.transcribe(audio_file)
            transcribed_query = result["text"]
            logger.info(f"Transcribed: {transcribed_query}")
        else:
            transcribed_query = query
            logger.info("Using text query")
        
        return QueryState(query=transcribed_query)
    
    @listen(lambda s: s.query)
    def search_knowledge_base(self, state: QueryState):
        """Search the vector database for relevant information"""
        logger.info("Searching knowledge base...")
        
        # Generate query embedding
        result = genai.embed_content(
            model=config.EMBEDDING_MODEL,
            content=state.query,
            task_type="retrieval_query"
        )
        query_embedding = result['embedding']
        
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
        """Generate final response using CrewAI with Gemini 2.5 Flash"""
        logger.info("Generating response with CrewAI + Gemini 2.5 Flash...")
        
        try:
            # Gemini LLM for CrewAI
            gemini_llm = LLM(
                model=f"gemini/{config.GEMINI_MODEL}",
                api_key=genai.get_api_key()
            )
            
            # Create search tool with collection
            search_tool = SearchKnowledgeBaseTool(collection=self.collection)
            
            # Create agents
            research_agent = Agent(
                role="Information Retrieval Specialist",
                goal="Find the most relevant information from the knowledge base to answer user queries",
                backstory="You are an expert at analyzing queries and searching through multimodal knowledge bases.",
                tools=[search_tool],
                llm=gemini_llm,
                verbose=True
            )
            
            response_agent = Agent(
                role="Response Generator",
                goal="Generate comprehensive, accurate responses based on retrieved information",
                backstory="You are an expert at synthesizing information and creating clear, helpful responses.",
                llm=gemini_llm,
                verbose=True
            )
            
            # Create tasks
            research_task = Task(
                description=f"Search for information relevant to: '{state.query}'. Use the search_knowledge_base tool.",
                agent=research_agent,
                expected_output="Detailed information from the knowledge base"
            )
            
            response_task = Task(
                description=f"Based on the research findings, generate a comprehensive response to: '{state.query}'.",
                agent=response_agent,
                expected_output="A well-structured, comprehensive response"
            )
            
            # Create and run crew
            crew = Crew(
                agents=[research_agent, response_agent],
                tasks=[research_task, response_task],
                verbose=True
            )
            
            result = crew.kickoff()
            logger.info("Response generated")
            
            return QueryState(
                query=state.query,
                results=state.results,
                response=str(result)
            )
            
        except Exception as e:
            logger.error(f"Error with CrewAI: {e}")
            # Fallback to direct Gemini call
            model = genai.GenerativeModel(config.GEMINI_MODEL)
            prompt = f"""Based on this context {state.results}, answer the question {state.query}."""
            
            response = model.generate_content(prompt)
            
            return QueryState(
                query=state.query,
                results=state.results,
                response=response.text
            )