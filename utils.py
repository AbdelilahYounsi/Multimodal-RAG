import logging
import subprocess
import json
from pathlib import Path
from typing import List
import os
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keep track of already ingested files
INGESTED_FILES = Path("ingested_files.txt")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using local Voxtral model via llama-mtmd-cli"""
    try:
        logger.info(f"Transcribing {audio_path} with llama.cpp Voxtral model...")
        
        cmd = [
            config.LLAMA_MTMD_CLI_PATH,
            "-m", config.VOXTRAL_MODEL,
            "--mmproj", config.VOXTRAL_MMPROJ,
            "--audio", audio_path,
            "-p", "transcribe the audio"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=os.environ)
        
        if result.returncode != 0:
            logger.error(f"Transcription error: {result.stderr}")
            raise Exception(f"Transcription failed: {result.stderr}")
        
        transcribed_text = result.stdout.strip()
        logger.info(f"Transcription complete: {transcribed_text[:100]}...")
        return transcribed_text
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using local embedding model via llama.cpp"""
    embeddings = []
    
    for text in texts:
        try:
            cmd = [
                    config.LLAMA_EMB_PATH,
                    "-m", config.EMBEDDING_MODEL_PATH,
                    "--embd-output-format", "array",
                    "-p", text
]            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=os.environ)
        
            if result.returncode != 0:
                logger.error(f"Embedding error: {result.stderr}")
                raise Exception(f"Embedding failed: {result.stderr}")
            
            embedding = json.loads(result.stdout.strip())[0]
            logger.info(f"Embedding complete:")
            embeddings.append(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            embeddings.append([0.0] * config.EMBEDDING_DIM)  
    return embeddings


def generate_response_llm(query: str, context: str) -> str:
    """Generate response using local LLM via llama.cpp"""
    try:
        logger.info("Generating response with local LLM...")
        
        prompt = f"""Based on this context: {context}, Answer the following question: {query}. Provide an accurate response."""
        
        cmd = [
            config.LLAMA_CLI_PATH,
            "-m", config.TEXT_LLM_MODEL_PATH,
            "-p", prompt, "--no-conversation"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=os.environ)
        if result.returncode != 0:
            logger.error(f"Generation error: {result.stderr}")
            raise Exception(f"Generation failed: {result.stderr}")
        
        text = result.stdout.strip().replace('[end of text]','').replace(prompt,'')
        logger.info(f"The answer is: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise



def get_ingested_files() -> set:
    """Load list of already ingested files"""
    if INGESTED_FILES.exists():
        return set(INGESTED_FILES.read_text().splitlines())
    return set()

def save_ingested_file(filename: str):
    """Append filename to ingested list"""
    with open(INGESTED_FILES, "a") as f:
        f.write(f"{filename}\n")