from typing import List
from pathlib import Path
import re
import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_law_text(text: str) -> List[Dict[str, str]]:
    # Tách theo Chương
    chapters = re.split(r"(Chương\s+[IVXLC]+\s+[^\n]+)", text)
    structured = []

    i = 1
    while i < len(chapters):
        chapter_title = chapters[i].strip()
        chapter_content = chapters[i + 1]

        # Tách theo Điều trong mỗi Chương
        articles = re.split(r"(Điều\s+\d+\.\s+[^\n]+)", chapter_content)
        j = 1
        while j < len(articles):
            article_title = articles[j].strip()
            article_content = articles[j + 1].strip()

            structured.append({
                "chapter": chapter_title,
                "article": article_title,
                "content": article_content
            })
            j += 2
        i += 2
    return structured

def load_and_split_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List:
    """
    Load a PDF file and split it into chunks.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List: List of document chunks
    """
    chunk_size = int(os.getenv("chunk_size","500"))
    chunk_overlap = int(os.getenv("chunk_overlap","50"))
    try:
        # Convert string path to Path object
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        # print the first page
        if not pages:
            raise ValueError(f"No content found in PDF: {pdf_path}")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",                       # đoạn văn
                "\n",                         # dòng
                ".",                          # câu
                " ",                          # từ
                ""                            # ký tự
            ]
        )
        
        chunks = text_splitter.split_documents(pages)
        for i, doc in enumerate(chunks[:5]):
            print(f"=== Chunk {i} ===")
            print(doc.page_content)
            print()
        # assert False
        return chunks
        
    except Exception as e:
        raise Exception(f"Error processing PDF {pdf_path}: {str(e)}") 