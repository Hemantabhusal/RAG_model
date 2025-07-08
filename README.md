# Modular RAG System with Groq API

A Retrieval-Augmented Generation (RAG) system that answers questions about documents using neural embeddings and the Groq API.

## Features
- Modular design with separate files for document loading, chunking, embedding generation, vector storage, and querying.
- Processes `sample.txt` (~1,500 characters) into ~8 chunks with `chunk_size=400`.
- Uses `all-MiniLM-L6-v2` for embeddings and `llama3-70b-8192` (via Groq API) for answers.
- Stores embeddings in FAISS for fast similarity search.
- Uses `requests` for reliable Groq API calls, bypassing `groq` library issues.
- Secure API key management with `.env` using `GROQ-API-KEY`.
- Dockerized for reproducibility.

## Setup
1. Create a `data/` directory and add `sample.txt`.
2. Create a `.env` file in the root directory with:
   ```text
   GROQ_API_KE=your-api-key