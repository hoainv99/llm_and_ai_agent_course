import os
from typing import List, Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Qdrant, FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from pdf_processor import load_and_split_pdf

def load_environment():
    """Load environment variables from .env file"""
    # Try to load from project root
    if not load_dotenv():
        # Try to load from data directory
        load_dotenv("data/.env")
    
    # Validate required environment variables
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please create a .env file with the following variables:\n"
            "GOOGLE_API_KEY=your_google_api_key_here\n"
            "QDRANT_URL=http://localhost:6333 (if using Qdrant)\n"
            "VECTOR_STORE_TYPE=faiss (or qdrant)\n"
            "FAISS_INDEX_PATH=data/faiss_index (if using FAISS)"
        )

# Load environment variables
load_environment()

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
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        self.collection_name = collection_name
        self.vector_store_type = vector_store_type or os.getenv("VECTOR_STORE_TYPE", "faiss")
        self.faiss_path = faiss_path or os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
        self.vector_store = None
        
        if self.vector_store_type == "qdrant":
            # Initialize Qdrant client
            qdrant_url = os.getenv("QDRANT_URL")
            if not qdrant_url:
                raise EnvironmentError(
                    "QDRANT_URL environment variable is required when using Qdrant vector store"
                )
            self.qdrant_client = QdrantClient(url=qdrant_url)
        
    def create_vector_store(self, documents: List):
        """
        Create or update the vector store with documents.
        
        Args:
            documents (List): List of document chunks
        """
        if self.vector_store_type == "qdrant":
            # Create Qdrant vector store
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
            # Add documents to vector store
            self.vector_store.add_documents(documents)
        else:
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            # Save FAISS index
            os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
            self.vector_store.save_local(self.faiss_path)
        
    def setup_qa_chain(self):
        """
        Set up the QA chain with a custom prompt.
        """
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
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