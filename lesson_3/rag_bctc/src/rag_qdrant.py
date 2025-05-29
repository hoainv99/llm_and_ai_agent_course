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
            temperature=0.0,
            convert_system_message_to_human=True
        )

        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("embedding_model", "models/embedding-001"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.collection_name = collection_name
        self.vectorstore = QdrantVectorStore(collection_name=self.collection_name)

    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store, supporting MultiVectorRetriever or ParentDocumentRetriever if enabled.
        """
        self.vectorstore.add_documents(documents)


    def setup_qa_chain(self):
        """
        Set up the QA chain with a custom prompt.
        """
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question about the story at the end.
            If the context doesn't provide enough information, just say that you don't know, don't try to make up an answer.
            Pay attention to the context of the question rather than just looking for similar keywords in the corpus.
            Use three sentences maximum and keep the answer as concise as possible.
            Always say "thanks for asking!" at the end of the answer. Generate answer by only Vietnamese.
            {context}
            Question: {question}
            Helpful Answer:
            """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        def qa_chain(question):

            results = self.vectorstore.get_relevant_documents(question, k=10)

            print(results)
            # Concatenate context from docs
            context = "\n---\n".join([
                doc["text"] for doc in results
            ])
            # Format the prompt
            prompt = PROMPT.format(context=context, question=question)
            # Get LLM response
            answer = self.llm.invoke(prompt)
            # Return answer and source documents
            return {
                "result": answer,
                "source_documents": results
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
        result = self.qa_chain(question)
        return result

def main():

    rag = RAGSystem(
        collection_name="thue_tncn"
    )
    # Load and process PDF
    pdf_path = "../data/thue_tncn.pdf"  # Replace with your PDF path
    documents = load_and_split_pdf(pdf_path)

    # Add documents to the vector store (supports ParentDocumentRetriever)
    # rag.add_documents(documents)

    # Setup QA chain
    rag.setup_qa_chain()

    # Example query
    question = "Thu nhập từ tiền lương, tiền công là thu nhập người lao động nhận được từ người sử dụng lao động, bao gồm:"
    result = rag.query(question)
    print(result)
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")

if __name__ == "__main__":
    main()