import os
import argparse
from pathlib import Path
from typing import List, Dict
from rag_system import RAGSystem
from pdf_processor import load_and_split_pdf

class RAGDemo:
    def __init__(self, vector_store_type: str = "faiss"):
        """
        Initialize the RAG demo system.
        
        Args:
            vector_store_type (str): Type of vector store to use ("qdrant" or "faiss")
        """
        self.rag = RAGSystem(vector_store_type=vector_store_type)
        self.processed_files = []
        
    def ingest_documents(self, pdf_dir: str) -> List[str]:
        """
        Ingest all PDFs from a directory.
        
        Args:
            pdf_dir (str): Directory containing PDF files
            
        Returns:
            List[str]: List of processed file names
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            print(f"Directory not found: {pdf_dir}")
            return []
            
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return []
        
        print(f"\nFound {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            try:
                print(f"\nProcessing {pdf_path.name}...")
                documents = load_and_split_pdf(str(pdf_path))
                self.rag.create_vector_store(documents)
                self.rag.setup_qa_chain()
                self.processed_files.append(pdf_path.name)
                print(f"Successfully processed {pdf_path.name}")
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {str(e)}")
        
        return self.processed_files
    
    def query_documents(self, question: str) -> Dict:
        """
        Query the processed documents.
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict: Query result with answer and source documents
        """
        if not self.processed_files:
            return {
                "error": "No documents have been processed yet. Please ingest documents first."
            }
        
        try:
            result = self.rag.query(question)
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            }
        except Exception as e:
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="RAG System Demo")
    parser.add_argument(
        "--pdf-dir",
        default="data",
        help="Directory containing PDF files to process"
    )
    parser.add_argument(
        "--vector-store",
        default="faiss",
        choices=["faiss", "qdrant"],
        help="Vector store type to use"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG demo
    demo = RAGDemo(vector_store_type=args.vector_store)
    
    # Ingest documents
    print(f"\nIngesting documents from {args.pdf_dir}...")
    processed_files = demo.ingest_documents(args.pdf_dir)
    
    if not processed_files:
        print("No documents were processed. Exiting.")
        return
    
    print(f"\nSuccessfully processed {len(processed_files)} documents:")
    for file in processed_files:
        print(f"- {file}")
    
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            question = input("\nEnter your question: ").strip()
            if question.lower() == 'exit':
                break
            
            result = demo.query_documents(question)
            
            if "error" in result:
                print(f"\nError: {result['error']}")
            else:
                print("\nAnswer:", result["answer"])
                print("\nSources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n{i}. {source['content']}")
                    if source['metadata']:
                        print(f"   Metadata: {source['metadata']}")
    else:
        # Example queries
        example_questions = [
            "What is the main topic of the documents?",
            "Can you summarize the key points?",
            "What are the main findings?"
        ]
        
        for question in example_questions:
            print(f"\nQuestion: {question}")
            result = demo.query_documents(question)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print("Answer:", result["answer"])
                print("\nSources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n{i}. {source['content']}")
                    if source['metadata']:
                        print(f"   Metadata: {source['metadata']}")

if __name__ == "__main__":
    main() 