import os
import uuid
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str,
        embeddings: GoogleGenerativeAIEmbeddings,
    ):
        """
        Initialize Qdrant vector store manager.
        
        Args:
            collection_name (str): Name of the collection
            vector_size (int): Size of the embedding vectors
            qdrant_url (str): URL of the Qdrant server
        """
        self.collection_name = collection_name
        self.vector_size = os.getenv("vector_size", 768)
        # Initialize Qdrant client
        self.client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.embeddings = embeddings
        
    
    def create_collection(self) -> bool:
        """
        Create a new collection if it doesn't exist.
        
        Returns:
            bool: True if collection was created or already exists
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
            
            return True
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        Delete the collection if it exists.
        
        Returns:
            bool: True if collection was deleted or didn't exist
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name in collection_names:
                self.client.delete_collection(collection_name=self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} does not exist")
            
            return True
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the collection.
        
        Args:
            documents (List[Dict]): List of documents with text and metadata
            
        Returns:
            bool: True if documents were added successfully
        """
        try:
            # Create collection if it doesn't exist
            if not self.create_collection():
                return False
            
            # Prepare documents for insertion
            points = []
            for i, doc in enumerate(documents):
                # Generate embedding for the document text
                embedding = self.embeddings.embed_query(doc.page_content)
       
                # Create point with embedding and metadata
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    }
                )
                points.append(point)
            
            # Upload points to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"Added {len(documents)} documents to collection {self.collection_name}")
            return True
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False
        
    def search(self, query: str, k: int = 5):
        query_embedding = self.embeddings.embed_query(query)

        # Search for similar vectors
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        return results
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of similar documents with scores
        """
        try:
            # Search for similar vectors
            results = self.search(query, k)
            processed_results = []
            for scored_point in results:
                processed_results.append({
                    "text": scored_point.payload["text"],
                    "metadata": scored_point.payload.get("metadata", {}),
                    "score": scored_point.score
                })
            
            return processed_results
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the collection.
        
        Returns:
            Dict: Collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {}

def main():
    # Example usage
    embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("embedding_model", "models/embedding-001"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            task_type = "RETRIEVAL_QUERY"
        )
    vector_store = QdrantVectorStore(collection_name="thue", embeddings=embeddings)    

    # Search for similar documents
    query = "Thông tin các khoản chịu thuế"
    results = vector_store.get_relevant_documents(query, k=5)
    
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        print(f"Metadata: {result['metadata']}")

if __name__ == "__main__":
    main() 