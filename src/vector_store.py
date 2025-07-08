import logging
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def store_embeddings(embeddings):
    """
    Store embeddings in a FAISS index.
    Returns the FAISS index.
    """
    try:
        embeddings_np = embeddings.cpu().numpy()
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        logger.info(f"Stored {index.ntotal} embeddings in FAISS index")
        return index
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        raise