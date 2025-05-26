from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """
    Load a PDF file and split it into chunks.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List: List of document chunks
    """
    try:
        # Convert string path to Path object
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        # print the first page
        print(pages[0])
        if not pages:
            raise ValueError(f"No content found in PDF: {pdf_path}")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(pages)
        return chunks
        
    except Exception as e:
        raise Exception(f"Error processing PDF {pdf_path}: {str(e)}") 