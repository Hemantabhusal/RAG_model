import os
import logging
from dotenv import load_dotenv
from src.document_processor import load_document, split_documents
from src.embedding_generator import generate_embeddings
from src.vector_store import store_embeddings
from src.query_processor import process_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    file_path = "data/sample.txt"
    groq_api_key = os.getenv("GROQ-API-KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KE environment variable is not set")

    try:
        documents = load_document(file_path)
        chunks = split_documents(documents, chunk_size=400, chunk_overlap=20)
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk.page_content[:100]}...")
        embeddings, chunk_texts, embedder = generate_embeddings(chunks)
        index = store_embeddings(embeddings)

        while True:
            question = input("Enter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            answer, context = process_query(question, embedder, index, chunk_texts, groq_api_key)
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Context: {context}\n")
    except Exception as e:
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()