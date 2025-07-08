import logging
from sentence_transformers import SentenceTransformer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_embeddings(chunks):
    """
    Generate embeddings for document chunks using all-MiniLM-L6-v2.
    Returns embeddings (torch tensor), chunk texts, and embedder.
    """
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = embedder.encode(chunk_texts, convert_to_tensor=True, batch_size=8)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings, chunk_texts, embedder
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise