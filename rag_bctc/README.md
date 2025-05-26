# PDF-based RAG System

This project implements a Retrieval Augmented Generation (RAG) system using LangChain, Google's Generative AI (Gemini), and supports both Qdrant and FAISS as vector databases. The system is designed to process PDF documents and answer questions based on their content.

## Prerequisites

- Python 3.8+ (for local development)
- Docker and Docker Compose (for containerized deployment)
- Google AI API key

## Setup

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Vector Store Configuration (Optional)
VECTOR_STORE_TYPE=faiss  # Options: "qdrant" or "faiss" (default: "faiss")
FAISS_INDEX_PATH=data/faiss_index  # Only used if VECTOR_STORE_TYPE is "faiss"

# Qdrant Configuration (Required if using Qdrant)
QDRANT_URL=http://localhost:6333

# Application Configuration (Optional)
DEBUG=False
LOG_LEVEL=INFO
```

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your PDF files in the `data/` directory.

### Docker Deployment

1. Build and start the containers:
```bash
docker-compose up --build
```

The application will be available at http://localhost:7860

To stop the containers:
```bash
docker-compose down
```

To stop the containers and remove the Qdrant volume:
```bash
docker-compose down -v
```

## Usage

### Quick Demo

The `rag_demo.py` script provides a complete demonstration of the RAG system, including document ingestion, indexing, and querying.

1. Basic usage (processes PDFs and runs example queries):
```bash
python rag_demo.py
```

2. Interactive mode:
```bash
python rag_demo.py --interactive
```

3. Specify PDF directory:
```bash
python rag_demo.py --pdf-dir /path/to/pdfs
```

4. Use Qdrant instead of FAISS:
```bash
python rag_demo.py --vector-store qdrant
```

The demo script provides:
- Automatic PDF processing and indexing
- Interactive query mode
- Example queries with source documents
- Detailed error handling and reporting

### Command Line Interface (CLI)

The CLI tool allows you to ingest multiple PDFs into the vector database at once.

1. Basic usage:
```bash
python cli.py /path/to/pdf/directory
```

2. Specify a custom collection name:
```bash
python cli.py /path/to/pdf/directory --collection my_collection
```

The CLI tool will:
- Recursively find all PDF files in the specified directory
- Process each PDF and create embeddings
- Store the embeddings in the vector database
- Show progress and any errors during processing

### Web Interface

To use the web-based chat interface:

1. Run the chat interface:
```bash
python chat_interface.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:7860)

3. Upload a PDF file using the interface

4. Start chatting with your document!

The web interface provides:
- PDF file upload
- Interactive chat interface
- Example questions
- Real-time responses
- Status updates

### Programmatic Usage

1. Import and initialize the RAG system:
```python
from rag_system import RAGSystem

# Using FAISS (default)
rag = RAGSystem()

# Or using Qdrant
rag = RAGSystem(vector_store_type="qdrant")
```

2. Process a PDF file:
```python
from pdf_processor import load_and_split_pdf

documents = load_and_split_pdf("data/your_pdf_file.pdf")
rag.create_vector_store(documents)
rag.setup_qa_chain()
```

3. Query the system:
```python
result = rag.query("Your question here?")
print(result['result'])
```

## Features

- PDF document processing and chunking
- Vector storage using Qdrant or FAISS
- Question answering using Google's Gemini Pro
- Customizable prompt templates
- Source document retrieval
- Web-based chat interface using Gradio
- CLI tool for batch PDF ingestion
- Docker support for easy deployment
- Environment-based configuration
- Interactive demo script

## Project Structure

- `rag_system.py`: Main RAG implementation
- `pdf_processor.py`: PDF processing utilities
- `chat_interface.py`: Gradio web interface
- `cli.py`: Command-line tool for PDF ingestion
- `rag_demo.py`: Complete RAG system demonstration
- `requirements.txt`: Project dependencies
- `Dockerfile`: Container definition
- `docker-compose.yml`: Multi-container setup
- `data/`: Directory for PDF files and FAISS indices 