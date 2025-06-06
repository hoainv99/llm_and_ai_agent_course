from typing import List
from pathlib import Path
import re
import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# llm = ChatGoogleGenerativeAI(
#             model=os.getenv("llm_model", "gemini-2.0-flash"),
#             google_api_key=os.getenv("GOOGLE_API_KEY"),
#             temperature=0.0
#         )

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

def load_and_split_pdf(pdf_path: str, parent_retriver = True ) -> List:
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
        if not pages:
            raise ValueError(f"No content found in PDF: {pdf_path}")
        
        # Split text into chunks
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                        "\nChương [IVXLCDM]+\n",  # Ưu tiên tách theo tiêu đề Chương
                        "\nĐiều \d+\.",          # Ưu tiên tách theo tiêu đề Điều
                        "\n\n",                  # Đoạn văn mới
                        "\n",                    # Dòng mới
                        " ",                     # Khoảng trắng (tách từ)
                        ""                       # Tách ký tự cuối cùng
                    ],
            chunk_size=chunk_size,             # Kích thước tối đa của mỗi chunk (vẫn hợp lý)
            chunk_overlap=chunk_overlap,           # Số lượng ký tự trùng lặp giữa các chunk (vẫn hợp lý)
            length_function=len,         # Hàm tính độ dài của chunk
            is_separator_regex=True      # Bật chế độ regex cho separators
            )
        chunks = text_splitter.split_documents(pages)
        # Add unique id to each chunk's metadata
        total_chunks= []
        for idx, chunk in enumerate(chunks):
            if len(chunk.page_content) < chunk_overlap and chunk.metadata["page"] + 1 == chunks[idx+1].metadata["page"]:
                continue
            # Use a combination of source, page, and chunk index for uniqueness if available
            source = chunk.metadata.get("source", str(pdf_path))
            page = chunk.metadata.get("page", 0)
            chunk.metadata["id"] = f"{source}:{page}:{idx}"
            total_chunks.append(chunk)
        print(len(total_chunks))
        return total_chunks
        
    except Exception as e:
        raise Exception(f"Error processing PDF {pdf_path}: {str(e)}") 