import logging
import requests
import numpy as np
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_query(question, embedder, index, chunk_texts, api_key):
    """
    Process a query using FAISS for retrieval and Groq API (via requests) for answer generation.
    Returns the answer and retrieved context.
    """
    try:
        # Log the API key (first 10 and last 10 characters for security)
        logger.info(f"Using API key: {api_key[:10]}...{api_key[-10:]}")
        
        question_embedding = embedder.encode([question], convert_to_tensor=True)
        _, indices = index.search(question_embedding.cpu().numpy(), k=2)
        context = [chunk_texts[i] for i in indices[0]]
        prompt = f"Based solely on the provided context, answer the question accurately and concisely in 1-2 sentences, without adding external information.\nContext: {''.join(context)}\nQuestion: {question}\nAnswer:"
        logger.info(f"Prompt: {prompt}")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        answer = response.json()["choices"][0]["message"]["content"].strip()
        return answer, context
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise