import os
import argparse
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.vector_store import QdrantVectorStore
from src.load_split_data import load_and_split_pdf
from dotenv import load_dotenv

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("embedding_model", "models/embedding-001"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            task_type = "RETRIEVAL_DOCUMENT"
        )

def create_collection(collection_name: str) -> bool:
    """
    Create a new Qdrant collection.
    
    Args:
        collection_name (str): Name of the collection to create
    """
    try:
        vector_store = QdrantVectorStore(collection_name=collection_name, embeddings=embeddings)
        success = vector_store.create_collection()
        if success:
            print(f"Successfully created collection: {collection_name}")
        return success
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        return False

def delete_collection(collection_name: str) -> bool:
    """
    Delete a Qdrant collection.
    
    Args:
        collection_name (str): Name of the collection to delete
    """
    try:
        vector_store = QdrantVectorStore(collection_name=collection_name, embeddings=embeddings)
        success = vector_store.delete_collection()
        if success:
            print(f"Successfully deleted collection: {collection_name}")
        return success
    except Exception as e:
        print(f"Error deleting collection: {str(e)}")
        return False

def add_documents(pdf_dir: str, collection_name: str) -> bool:
    """
    Add documents from PDFs to a Qdrant collection.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        collection_name (str): Name of the collection to add documents to
    """
    try:
        # Initialize vector store
        vector_store = QdrantVectorStore(collection_name=collection_name, embeddings=embeddings)
        
        # Get all PDF files
        pdf_files = list(Path(pdf_dir).glob("**/*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return False
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        for pdf_path in pdf_files:
            try:
                print(f"\nProcessing {pdf_path.name}...")
                documents = load_and_split_pdf(str(pdf_path))
                success = vector_store.add_documents(documents)
                if success:
                    print(f"Successfully processed {pdf_path.name}")
                else:
                    print(f"Failed to process {pdf_path.name}")
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {str(e)}")
        
        return True
    except Exception as e:
        print(f"Error adding documents: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Qdrant Vector Store CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create collection command
    create_parser = subparsers.add_parser("create", help="Create a new collection")
    create_parser.add_argument(
        "collection_name",
        help="Name of the collection to create"
    )
    
    # Delete collection command
    delete_parser = subparsers.add_parser("delete", help="Delete a collection")
    delete_parser.add_argument(
        "collection_name",
        help="Name of the collection to delete"
    )
    
    # Add documents command
    add_parser = subparsers.add_parser("add", help="Add documents to a collection")
    add_parser.add_argument(
        "pdf_dir",
        help="Directory containing PDF files to add"
    )
    add_parser.add_argument(
        "--collection",
        default="pdf_documents",
        help="Name of the collection to add documents to (default: pdf_documents)"
    )
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_collection(args.collection_name)
    
    elif args.command == "delete":
        delete_collection(args.collection_name)
    
    elif args.command == "add":
        if not os.path.exists(args.pdf_dir):
            print(f"Error: Directory {args.pdf_dir} does not exist")
            return
        add_documents(args.pdf_dir, args.collection)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 