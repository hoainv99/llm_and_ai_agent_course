import os
from typing import List, Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from .pdf_processor import load_and_split_pdf
from .vector_storage import QdrantVectorStore, FAISSVectorStore
load_dotenv()

class RAGSystem:
    def __init__(
        self,
        collection_name: str = "pdf_documents",
        vector_store_type: str = None,
        faiss_path: str = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            collection_name (str): Name of the vector store collection
            vector_store_type (str): Type of vector store to use ("qdrant" or "faiss")
            faiss_path (str): Path to store FAISS index (only used if vector_store_type is "faiss")
        """
        # Initialize Google AI
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("llm_model","gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        self.retriever = QdrantVectorStore(collection_name=collection_name)
        
    def setup_qa_chain(self):
        """
        Set up the QA chain with a custom prompt.
        """
        # Custom prompt template
        prompt_template = """Hãy sử dụng các phần nội dung sau để trả lời câu hỏi ở cuối.

        Chỉ sử dụng thông tin có trong phần văn bản. Nếu không tìm thấy câu trả lời trong nội dung cho trước, hãy trả lời là "Không có thông tin trong văn bản để trả lời câu hỏi này" và KHÔNG được tự suy đoán hay thêm thông tin từ bên ngoài.

        Khi trả lời, nếu có thể, hãy trích dẫn nguyên văn hoặc chỉ rõ nội dung liên quan trong văn bản để làm căn cứ.

        ---

        Văn bản pháp luật:  
        {context}

        Câu hỏi:  
        {question}

        Trả lời:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        def qa_chain(inputs):
            question = inputs["query"] if isinstance(inputs, dict) else inputs
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(question, k=5)
            # Concatenate context from docs
            context = "\n\n".join([doc["text"] for doc in docs])
            print(context)
            # Format the prompt
            prompt = PROMPT.format(context=context, question=question)
            # Get LLM response
            answer = self.llm.invoke(prompt)
            # Return answer and source documents
            return {
                "result": answer,
                "source_documents": docs
            }

        self.qa_chain = qa_chain
        
    def query(self, question: str) -> dict:
        """
        Query the RAG system.
        
        Args:
            question (str): The question to ask
            
        Returns:
            dict: The answer and source documents
        """
        return self.qa_chain({"query": question})

def main():
    # Example usage with FAISS (default)
    rag = RAGSystem()
    
    # Load and process PDF
    pdf_path = "data/hpg.pdf"  # Replace with your PDF path
    documents = load_and_split_pdf(pdf_path)
    
    # Create vector store
    rag.create_vector_store(documents)
    
    # Setup QA chain
    rag.setup_qa_chain()
    
    # Example query
    question = "What is the main topic of the document?"
    result = rag.query(question)
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print("\nSource Documents:")
    for doc in result['source_documents']:
        print(f"- {doc.page_content[:200]}...")

if __name__ == "__main__":
    main() 