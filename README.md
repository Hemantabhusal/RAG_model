# 🧠 RAG-Powered Q&A System with FAISS and Groq

A Retrieval-Augmented Generation (RAG) powered system that intelligently answers questions using information from text or PDF documents. It combines HuggingFace transformer embeddings, FAISS vector search, and the Groq API (LLaMA3-70B) to deliver context-aware, accurate responses.

## 🎯 Main Objectives

- Generate Text Embeddings using `all-MiniLM-L6-v2` from HuggingFace.
- Store & Retrieve embeddings via the FAISS vector database.
- Interactive Q&A with context-aware answers via Groq’s `llama3-70b-8192` API.

## ⚙️ Key Functionalities

- Document Processing: Load and split text/PDFs using LangChain.
- Embedding Generation: Convert text into numerical vectors.
- Vector Storage/Retrieval: Efficient similarity search with FAISS.
- Contextual Answering: Use relevant chunks to generate accurate answers with Groq.

## 🧩 Project Structure

- `document_processor.py`: Load/split documents using `RecursiveCharacterTextSplitter`.
- `embedding_generator.py`: Generate embeddings with `all-MiniLM-L6-v2`.
- `vector_store.py`: Store/retrieve embeddings in FAISS.
- `query_processor.py`: Query FAISS + call Groq API for answers.
- `main.py`: Runs the complete command-line Q&A pipeline.

## ✨ Features

- Handles text or PDFs from 100 to 10,000+ characters
- CPU-friendly embeddings
- Fast vector similarity search via FAISS
- Groq-powered LLM response generation
- Dynamic input for flexible document loading

## 🛠️ Technologies Used

- HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- FAISS vector database
- Groq API (llama3-70b-8192)
- LangChain, PyPDF2 for document processing
- python-dotenv for environment variables

## 🚀 Installation and Setup (Single Command Flow)

```bash
# Clone the repository
git clone https://github.com/your-username/rag-project.git
cd rag-project

# Create requirements.txt
echo "torch==2.0.1
transformers==4.35.0
sentence-transformers==2.2.2
langchain==0.0.340
faiss-cpu==1.7.4
pypdf2==3.0.1
requests==2.31.0
python-dotenv==1.0.1" > requirements.txt

# Install dependencies
pip install -r requirements.txt

# Set up environment variable
echo "GROQ_API_KEY=your_groq_api_key" > .env

# Create data folder and sample document
mkdir data
echo "Large language models (LLMs) are built on transformer architectures and trained on massive corpora to understand and generate human-like text." > data/sample.txt

# Run the app
python main.py
```

## 🧪 Example Usage

```bash
Enter file path (e.g., data/sample.txt): data/sample.txt
Enter your question (or 'quit' to exit): What is LLM
```

Expected output:

```
INFO: Loaded API key: gsk_7tmnIE...figYLVhG
INFO: Loaded 1 document from data/sample.txt, length: 1500 characters
INFO: Split into 8 chunks
Chunk 1: Large language models (LLMs) are built on transformer architectures...
Question: What is LLM
Answer: Large language models (LLMs) are transformer-based neural networks trained on massive datasets to generate human-like text.
Context: ['Large language models (LLMs) are built on transformer architectures...', 'Retrieval-Augmented Generation (RAG) combines retrieval...']
```

## 👥 Contributors

- Your Name

## 🙏 Acknowledgements

Thanks to HuggingFace for the `all-MiniLM-L6-v2` model, Groq for the `llama3-70b-8192` API, LangChain for document processing, and FAISS for high-performance vector search.
