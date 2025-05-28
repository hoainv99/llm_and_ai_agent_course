import os
from typing import List, Dict, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

class FAISSVectorStore:
    def __init__(
        self,
        index_path: str = "./data/faiss_index",
        embedding_model: str = "models/embedding-001"
    ):
        """
        Initialize FAISS vector store manager.
        
        Args:
            index_path (str): Path to store/load the FAISS index
            embedding_model (str): Name of the embedding model to use
        """
        self.index_path = index_path
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.vector_store = None
        
    def create_vector_store(self, documents: List[Document]) -> bool:
        """
        Create a new FAISS vector store with documents.
        
        Args:
            documents (List[Document]): List of documents to add to the vector store
            
        Returns:
            bool: True if vector store was created successfully
        """
        try:
            # Create FAISS vector store from documents
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save the index
            self.vector_store.save_local(self.index_path)
            print(f"Created and saved FAISS index at {self.index_path}")
            return True
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return False
    
    def load_vector_store(self) -> bool:
        """
        Load an existing FAISS vector store.
        
        Returns:
            bool: True if vector store was loaded successfully
        """
        try:
            if not os.path.exists(self.index_path):
                print(f"No existing index found at {self.index_path}")
                return False
            
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings
            )
            print(f"Loaded FAISS index from {self.index_path}")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the existing vector store.
        
        Args:
            documents (List[Document]): List of documents to add
            
        Returns:
            bool: True if documents were added successfully
        """
        try:
            if self.vector_store is None:
                if not self.load_vector_store():
                    return self.create_vector_store(documents)
            
            # Add documents to existing vector store
            self.vector_store.add_documents(documents)
            
            # Save the updated index
            self.vector_store.save_local(self.index_path)
            print(f"Added {len(documents)} documents to FAISS index")
            return True
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Document]: List of similar documents
        """
        try:
            if self.vector_store is None:
                if not self.load_vector_store():
                    return []
            
            # Search for similar documents
            results = self.vector_store.similarity_search(query, k=k)
            return results
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def get_index_info(self) -> Dict:
        """
        Get information about the vector store.
        
        Returns:
            Dict: Vector store information
        """
        try:
            if self.vector_store is None:
                if not self.load_vector_store():
                    return {"error": "No vector store loaded"}
            
            return {
                "index_path": self.index_path,
                "index_exists": os.path.exists(self.index_path),
                "embedding_model": self.embeddings.model_name
            }
            
        except Exception as e:
            return {"error": str(e)}

def main():
    # Example usage
    vector_store = FAISSVectorStore()
    
    # Create some test documents
    test_documents = [
        Document(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test1", "page": 1}
        ),
        Document(
            page_content="Machine learning is a subset of AI.",
            metadata={"source": "test2", "page": 1}
        )
    ]
    
    # Create vector store
    vector_store.create_vector_store(test_documents)
    
    # Search for similar documents
    query = "What is AI?"
    results = vector_store.similarity_search(query, k=2)
    
    print("\nSearch Results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Content: {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")
    
    # Get index info
    info = vector_store.get_index_info()
    print("\nIndex Info:", info)

if __name__ == "__main__":
    main() 