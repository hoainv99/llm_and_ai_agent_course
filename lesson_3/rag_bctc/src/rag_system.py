import os
from typing import List, Literal, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryByteStore, InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf_processor import load_and_split_pdf
from vector_storage import QdrantVectorStore, FAISSVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
load_dotenv("../.env")

class RAGSystem:
    def __init__(
        self,
        collection_name: str = "pdf_documents",
        vector_store_type: Optional[str] = None,
        use_multi_vector: bool = False,
        use_parent_document: bool = False,
        parent_splitter: Optional[object] = None,
        child_splitter: Optional[object] = None,
        search_type: str = "similarity",  # NEW: 'similarity' or 'mmr'
        mmr_lambda_mult: float = 0.5,     # NEW: Lambda for MMR (0.0 to 1.0)
        k_retrieve: int = 4               # NEW: Number of documents to retrieve
    ):
        """
        Initialize the RAG system.

        Args:
            collection_name (str): Name of the vector store collection
            vector_store_type (str): Type of vector store to use ("qdrant", "faiss", or "chroma")
            use_multi_vector (bool): Whether to use MultiVectorRetriever from LangChain
            use_parent_document (bool): Whether to use ParentDocumentRetriever from LangChain
            parent_splitter (object): Optional, text splitter for parent docs (if using ParentDocumentRetriever)
            child_splitter (object): Optional, text splitter for child docs (if using ParentDocumentRetriever)
        """
        # Initialize Google AI
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("llm_model", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0
            # convert_system_message_to_human=True
        )

        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("embedding_model", "models/embedding-001"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.use_multi_vector = use_multi_vector
        self.use_parent_document = use_parent_document
        self.collection_name = collection_name
        self.vector_store_type = vector_store_type
        self.search_type = search_type
        self.mmr_lambda_mult = mmr_lambda_mult
        self.k_retrieve = k_retrieve
        self.parent_splitter = parent_splitter
        self.child_splitter = child_splitter

        self._setup_vector_store()
        self._setup_search_kwargs()
        self._setup_retriever()
        
        
    def _setup_vector_store(self):
        if self.vector_store_type == "chroma":
            persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./vector_store/chroma_db")
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            self.vectorstore.persist()
        elif self.vector_store_type == "faiss":
            self.vectorstore = FAISSVectorStore(
                collection_name=self.collection_name,
                index=os.getenv("FAISS_PATH", "./vector_store/faiss_index")
            )
        else:
            self.vectorstore = QdrantVectorStore(collection_name=self.collection_name)

    def _setup_search_kwargs(self):
        self.common_search_kwargs = {"k": self.k_retrieve}
        if self.search_type == "mmr":
            self.common_search_kwargs["search_type"] = "mmr"
            self.common_search_kwargs["lambda_mult"] = self.mmr_lambda_mult
            # For MMR, it's often beneficial to fetch more initial candidates (fetch_k)
            # than the final 'k' documents to re-rank from.
            self.common_search_kwargs["fetch_k"] = max(self.k_retrieve * 2, 10) # Fetch at least 10, or 2*k
        elif self.search_type == "similarity":
            self.common_search_kwargs["search_type"] = "similarity" # Explicitly set for clarity
        else:
            raise ValueError(f"Unsupported search_type: {self.search_type}. Must be 'similarity' or 'mmr'.") 
    
    def _setup_retriever(self):
        if self.use_multi_vector:
            byte_store = InMemoryByteStore()
            self.retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                byte_store=byte_store,
                search_kwargs=self.common_search_kwargs,
                id_key="id"  # assumes your documents have an "id" in metadata
            )
        elif self.use_parent_document:
            # ParentDocumentRetriever expects a vectorstore and a docstore
            # parent_splitter and child_splitter should be provided by the user
            # if parent_splitter is None or child_splitter is None:
            #     raise ValueError("parent_splitter and child_splitter must be provided for ParentDocumentRetriever.")
            docstore = InMemoryStore()
            self.retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=docstore,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
                search_kwargs=self.common_search_kwargs,
                id_key="id"
            )
        else:
            self.retriever = self.vectorstore.as_retriever(**self.common_search_kwargs)

    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store, supporting MultiVectorRetriever or ParentDocumentRetriever if enabled.
        """
        if self.use_multi_vector and hasattr(self.retriever.vectorstore, "add_documents"):
            print("processing add doc to vector store (MultiVectorRetriever)")
            self.retriever.vectorstore.add_documents(documents)
            doc_ids = []
            for doc in documents:
                doc_ids.append(doc.metadata["id"])
            self.retriever.docstore.mset(list(zip(doc_ids, documents)))
            self.retriever.vectorstore.persist()
            print("add data succesfully!!!")
        elif self.use_parent_document and hasattr(self.retriever.vectorstore, "add_documents"):
            print("processing add doc to vector store (ParentDocumentRetriever)")
            # ParentDocumentRetriever expects parent docs to be split and added
            # The retriever will handle splitting and storing parent/child docs
            self.retriever.add_documents(documents)
            if hasattr(self.retriever.vectorstore, "persist"):
                self.retriever.vectorstore.persist()
                print("persist data succesfully!!!")
            print("add data succesfully!!!")
        elif hasattr(self.retriever, "add_documents"):
            self.retriever.add_documents(documents)
            if hasattr(self.retriever, "persist"):
                self.retriever.persist()
        else:
            raise NotImplementedError("The current vector store does not support adding documents directly.")
    
    def setup_qa_chain(self):
        """
        Set up the QA chain using LangChain Expression Language (LCEL).
        """


        answer_prompt = PromptTemplate(
            template="""Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi:
            Nếu ngữ cảnh không cung cấp đủ thông tin, hãy nói rằng bạn không biết, đừng cố gắng tự bịa ra câu trả lời.
            Hãy chú ý đến ngữ cảnh của câu hỏi thay vì chỉ tìm kiếm các từ khóa tương tự.
            {context}
            Question: {input}
            Câu trả lời hữu ích:
            """,
            input_variables=["context", "input"]
        )

        document_qa_chain = create_stuff_documents_chain(self.llm, answer_prompt)

        self.qa_chain = create_retrieval_chain(self.retriever, document_qa_chain)
        # self.qa_chain = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     retriever=self.retriever,
        #     chain_type="stuff",
        #     return_source_documents=True,
        #     chain_type_kwargs={"prompt": answer_prompt} # Pass the custom prompt here
        # )
    # def setup_qa_chain(self):
    #     """
    #     Set up the QA chain with a custom prompt.
    #     """
    #     # Custom prompt template
    #     prompt_template = """Use the following pieces of context to answer the question about the story at the end.
    #         If the context doesn't provide enough information, just say that you don't know, don't try to make up an answer.
    #         Pay attention to the context of the question rather than just looking for similar keywords in the corpus.
    #         Use three sentences maximum and keep the answer as concise as possible.
    #         Always say "thanks for asking!" at the end of the answer. Generate answer by only Vietnamese.
    #         {context}
    #         Question: {question}
    #         Helpful Answer:
    #         """

    #     PROMPT = PromptTemplate(
    #         template=prompt_template, input_variables=["context", "question"]
    #     )

    #     def qa_chain(question):
    #         # Retrieve relevant documents
    #         if self.use_multi_vector or self.use_parent_document:
    #             # MultiVectorRetriever and ParentDocumentRetriever both have get_relevant_documents
    #             docs = self.retriever.get_relevant_documents(question, k=5)
    #             # For these retrievers, relevance scores are not typically returned by get_relevant_documents
    #             # You might need to implement a custom scoring or assume all retrieved are relevant.
    #             results = [(doc, 1.0) for doc in docs] # Assign a dummy score for consistency
    #         elif hasattr(self.retriever, "similarity_search_with_relevance_scores"):
    #             results = self.retriever.similarity_search_with_relevance_scores(question, k=5)
    #         elif hasattr(self.retriever, "similarity_search"):
    #             docs = self.retriever.similarity_search_with_relevance_scores(question, k=5)
    #             results = [(doc, 1.0) for doc in docs]
    #         else:
    #             raise NotImplementedError("Retriever does not support similarity search.")


    #         print(results)
    #         # Concatenate context from docs
    #         context = "\n---\n".join([
    #             doc.page_content for doc, _score in results
    #             if _score > float(os.getenv("SCORE_RELEVANCE", "0.8"))
    #         ])
    #         # Format the prompt
    #         prompt = PROMPT.format(context=context, question=question)
    #         # Get LLM response
    #         answer = self.llm.invoke(prompt)
    #         # Return answer and source documents
    #         return {
    #             "result": answer,
    #             "source_documents": results
    #         }
            

    #     # self.qa_chain = RetrievalQA.from_chain_type(
    #     #     llm=self.llm,
    #     #     retriever=self.retriever.as_retriever(search_kwargs={"k": 5}),
    #     #     chain_type="stuff",
    #     #     return_source_documents=True,
    #     #     chain_type_kwargs={"prompt": PROMPT} # Pass the custom prompt here
    #     # )

    #     self.qa_chain = qa_chain

    def query(self, question: str) -> dict:
        """
        Query the RAG system.
        Args:
            question (str): The question to ask

        Returns:
            dict: The answer and source documents
        """
        result = self.qa_chain.invoke({"input": question})
        return result

def main():
    # Example usage with Chroma and MultiVectorRetriever
    # To use ParentDocumentRetriever, set use_parent_document=True and provide splitters
    # Example: using ParentDocumentRetriever
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    rag = RAGSystem(
        collection_name="thue_tncn",
        vector_store_type="chroma",
        use_multi_vector=False,
        use_parent_document=True,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        search_type="similarity",         # Set search type to MMR
        mmr_lambda_mult=0.7,       # Adjust lambda_mult (e.g., 0.7 for more relevance bias)
        k_retrieve=5               # Retrieve 5 documents
    )
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # print(retrieval_qa_chat_prompt)
    # Load and process PDF
    pdf_path = "../data/thue_tncn.pdf"  # Replace with your PDF path
    documents = load_and_split_pdf(pdf_path)

    # Add documents to the vector store (supports ParentDocumentRetriever)
    # rag.add_documents(documents)

    # Setup QA chain
    rag.setup_qa_chain()

    # Example query
    question = "Các khoản thu nhập chịu thuế"
    result = rag.query(question)
    print(result)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    main()