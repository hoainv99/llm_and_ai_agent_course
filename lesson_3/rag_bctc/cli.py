import os
import argparse
from pathlib import Path
from vector_store import QdrantVectorStore
from pdf_processor import load_and_split_pdf

def ingest_pdfs(pdf_dir: str, collection_name: str = "pdf_documents"):
    """
    Ingest all PDFs from a directory into the vector database.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        collection_name (str): Name of the Qdrant collection
    """
    # Initialize RAG system
    rag = QdrantVectorStore(collection_name=collection_name)
    
    # Get all PDF files
    pdf_files = list(Path(pdf_dir).glob("**/*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF
    for pdf_path in pdf_files:
        try:
            print(f"\nProcessing {pdf_path.name}...")
            documents = load_and_split_pdf(str(pdf_path))
            print(documents)
            assert False
            rag.add_documents(documents)
            print(f"Successfully processed {pdf_path.name}")
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector database")
    parser.add_argument(
        "pdf_dir",
        help="Directory containing PDF files to ingest"
    )
    parser.add_argument(
        "--collection",
        default="pdf_documents",
        help="Name of the Qdrant collection (default: pdf_documents)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_dir):
        print(f"Directory {args.pdf_dir} does not exist")
        return
    
    ingest_pdfs(args.pdf_dir, args.collection)

if __name__ == "__main__":
    main() 