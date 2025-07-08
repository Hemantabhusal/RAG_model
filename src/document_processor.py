import logging
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_document(file_path):
    """
    Load a text or PDF file and return a list of documents.
    """
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return [Document(page_content=content, metadata={"source": file_path})]
        elif file_path.endswith('.pdf'):
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() or ""
            return [Document(page_content=content, metadata={"source": file_path})]
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    except Exception as e:
        logger.error(f"Error loading document from {file_path}: {e}")
        raise

def split_documents(documents, chunk_size=400, chunk_overlap=20):
    """
    Split documents into chunks for processing.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        raise