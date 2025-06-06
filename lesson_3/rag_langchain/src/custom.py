import os
import uuid
from typing import List, Dict, Optional, Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
# THAY ĐỔI IMPORT NÀY:
from langchain_core.vectorstores import VectorStore # Kế thừa từ đây
from langchain_core.embeddings import Embeddings # Để type hint cho embedding_function

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

EMBEDDING_MODEL_NAME = os.getenv("embedding_model", "models/embedding-001")
try:
    VECTOR_SIZE = int(os.getenv("vector_size", 768)) # Nên là hằng số global hoặc truyền vào
except ValueError:
    print("Warning: vector_size environment variable is not a valid integer. Defaulting to 768.")
    VECTOR_SIZE = 768

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
class QdrantVectorStore(VectorStore):
    client: QdrantClient
    embeddings: Embeddings # Đây là trường Pydantic mong đợi
    collection_name: str
    
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
 
    def delete_collection(self) -> bool: # Giữ lại hàm này cho tiện quản lý
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted collection: '{self.collection_name}'")
            return True
        except Exception as e:
            print(f"Error deleting collection '{self.collection_name}' (it might not exist): {str(e)}")
            return False

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None, # ids này là id cho các text/chunk đang được thêm
        **kwargs: Any,
    ) -> List[str]:
        texts_list = list(texts) # Chuyển Iterable thành list để lấy len
        if ids is None:
            actual_ids = [str(uuid.uuid4()) for _ in texts_list]
        else:
            if len(ids) != len(texts_list):
                raise ValueError("If ids are provided, their length must match the number of texts.")
            actual_ids = ids
        
        if metadatas is None:
            actual_metadatas = [{} for _ in texts_list]
        else:
            if len(metadatas) != len(texts_list):
                raise ValueError("If metadatas are provided, their length must match the number of texts.")
            actual_metadatas = metadatas

        embeddings = self.embeddings.embed_documents(texts_list)

        points = []
        for i, text_content in enumerate(texts_list):
            point = PointStruct(
                id=actual_ids[i], # Sử dụng ID đã được xác định cho chunk này
                vector=embeddings[i],
                payload={
                    "text": text_content,
                    "metadata": actual_metadatas[i] or {} # metadata này chứa parent_id
                }
            )
            points.append(point)

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        return actual_ids # Trả về list các ID của các chunk đã được thêm


    def add_documents(self, documents: List[Document], *, ids: Optional[List[str]] = None, **kwargs: Any) -> List[str]:
        # `ids` ở đây (nếu được ParentDocumentRetriever truyền vào) là ID của parent documents.
        # Chúng ta không sử dụng trực tiếp `ids` này làm ID cho các Qdrant points (child chunks).
        # ParentDocumentRetriever đã đặt ID của parent vào metadata của child documents (documents ở đây).
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents] # metadata này đã chứa parent_id

        # Gọi add_texts, để nó tự tạo ID (UUID) cho các child chunks này.
        # Các metadata (bao gồm parent_id) sẽ được lưu cùng chunk.
        return self.add_texts(texts, metadatas=metadatas, ids=None, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict] = None, **kwargs: Any
    ) -> List[Document]:
        """Run similarity search with score."""
        results_with_scores = self.similarity_search_with_score(query, k, filter=filter, **kwargs)
        return [doc for doc, _score in results_with_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict] = None, **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """Run similarity search with Qdrant and return documents with scores."""
        query_embedding = self.embeddings.embed_query(query)

        qdrant_filter_model = None
        if filter:
            # Chuyển đổi dict filter của LangChain sang Qdrant filter
            # Đây là một ví dụ đơn giản, bạn có thể cần logic phức tạp hơn
            conditions = []
            for key, value in filter.items():
                # Giả định key có dạng "metadata.field_name" cho nested fields
                # hoặc "field_name" cho top-level payload fields.
                # Qdrant client mong đợi key không có "metadata." trong FieldCondition
                # nếu bạn đang filter trên payload.
                actual_key = key.replace("metadata.", "")
                conditions.append(
                    models.FieldCondition(
                        key=actual_key,
                        match=models.MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter_model = models.Filter(must=conditions)


        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter_model,
            limit=k,
            with_payload=True,
            with_vectors=False # Thường không cần trả về vector
        )

        docs_with_scores = []
        for hit in search_results:
            metadata = hit.payload.get("metadata", {})
            # `text` key trong payload nên là nội dung gốc của chunk/document
            page_content = hit.payload.get("text", "")
            doc = Document(page_content=page_content, metadata=metadata)
            docs_with_scores.append((doc, hit.score))
        return docs_with_scores

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        **kwargs: Any,
    ) -> "QdrantVectorStore":
        """Construct QdrantLangchainVectorStore from raw texts."""
        if collection_name is None:
            collection_name = "langchain-" + str(uuid.uuid4())
        
        qdrant_client = client if client is not None else QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

        instance = cls(collection_name, embedding, qdrant_client, **kwargs)
        instance.add_texts(texts, metadatas=metadatas, ids=ids)
        return instance

def main():
    # Example usage
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY
        # max_tokens không phải là tham số ở đây, kích thước vector phụ thuộc vào model
    )
    qdrant_global_client = QdrantClient(url=QDRANT_URL) # Tạo client một lần

    # Truyền embedding_function và client vào constructor
    child_vector_store = QdrantVectorStore(
        client=qdrant_global_client,
        embeddings=embedding_function,
        collection_name="test_pydantic_fix_v2" # Đặt tên collection
    )
    # vector_store.delete_collection()
    child_vector_store.create_collection()

    store = InMemoryStore() # lưu trữ tài liệu gốc
    parent_retriever = ParentDocumentRetriever(
        vectorstore=child_vector_store, # Pass the vector store instance directly
        docstore=store,                 # Nơi lưu trữ tài liệu gốc
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        # parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000), # Tùy chọn
        # child_metadata_fields=["source", "page"], # Các trường metadata từ parent muốn copy xuống child
    )
    # 4. Chuẩn bị tài liệu gốc (parent documents)
    parent_documents = [
        Document(
            page_content="Bộ Giao thông vận tải là cơ quan của Chính phủ Việt Nam, thực hiện chức năng quản lý nhà nước về giao thông vận tải đường bộ, đường sắt, đường thủy nội địa, hàng hải, hàng không trong phạm vi cả nước. Các dự án đường cao tốc Bắc-Nam đang được Bộ Giao thông vận tải đẩy nhanh tiến độ. Quy định mới về an toàn hàng không vừa được ban hành.",
            metadata={"source": "wikipedia_giao_thong", "doc_id": "doc_gtvt"} # doc_id quan trọng cho docstore
        ),
        Document(
            page_content="Bộ Khoa học và Công nghệ (MOST) chịu trách nhiệm quản lý về các hoạt động nghiên cứu khoa học, phát triển công nghệ và đổi mới sáng tạo. MOST cũng thúc đẩy hợp tác quốc tế trong lĩnh vực khoa học và công nghệ, nhằm nâng cao năng lực quốc gia.",
            metadata={"source": "website_most", "doc_id": "doc_khcn"}
        ),
         Document(
            page_content="An toàn thực phẩm là vấn đề quan trọng. Bộ Y Tế và Bộ Nông nghiệp cùng phối hợp quản lý. Các quy chuẩn về vệ sinh an toàn thực phẩm được cập nhật thường xuyên.",
            metadata={"source": "bao_suc_khoe", "doc_id": "doc_attp"}
        )
    ]
    
    # Thêm tài liệu gốc vào ParentDocumentRetriever
    # ParentDocumentRetriever sẽ:
    # - Lưu tài liệu gốc vào `docstore` (sử dụng `doc_id` từ metadata nếu có, hoặc tạo mới)
    # - Chia tài liệu gốc thành các chunk con bằng `child_splitter`
    # - Tạo embedding cho các chunk con
    # - Thêm các chunk con vào `vectorstore` (QdrantVectorStore của chúng ta)
    #   (Lưu ý: ParentDocumentRetriever sẽ gọi child_vector_store.add_documents)
    parent_retriever.add_documents(parent_documents, ids=None)
    # 3. Sử dụng retriever để tìm kiếm
    query = "Các cơ quan nào quản lý về giao thông?"
    
    # LangChain cung cấp 2 phương thức chính:
    # `invoke` là cách gọi mới, chuẩn hóa (tương đương với `get_relevant_documents`)
    results = parent_retriever.invoke(query, k=2)
    # print("results",results)
    print(f"\n--- Search Results for query: '{query}' ---")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Score: {doc['score']:.4f}")
        print(f"   Text: {doc['text']}")
        print(f"   Metadata: {doc['metadata']}")


if __name__ == "__main__":
    main()