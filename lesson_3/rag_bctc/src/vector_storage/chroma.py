import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings



class ChromaVectorStore:
    """
    A class to manage a ChromaDB vector store for RAG applications.
    It handles loading PDF documents, splitting them into chunks,
    adding/updating chunks in the Chroma database, and clearing the database.
    """

    def __init__(self, chroma_path: str = "chroma", data_path: str = "data"):
        """
        Initializes the ChromaVectorStore.

        Args:
            chroma_path (str): The directory path where the ChromaDB will be persisted.
            data_path (str): The directory path containing the PDF documents to load.
        """
        self.chroma_path = chroma_path
        self.data_path = data_path
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("embedding_model", "models/embedding-001"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_tokens=self.vector_size
        )
    def _load_documents(self) -> list[Document]:
        """
        Loads PDF documents from the specified data path.

        Returns:
            list[Document]: A list of loaded Langchain Document objects.
        """
        print(f"Loading documents from: {self.data_path}")
        document_loader = PyPDFDirectoryLoader(self.data_path)
        documents = document_loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Splits the loaded documents into smaller chunks.

        Args:
            documents (list[Document]): A list of Langchain Document objects to split.

        Returns:
            list[Document]: A list of chunked Langchain Document objects.
        """
        print(f"Splitting {len(documents)} documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        return chunks

    def _calculate_chunk_ids(self, chunks: list[Document]) -> list[Document]:
        """
        Calculates unique IDs for each document chunk based on source, page, and chunk index.
        IDs will be in the format "source/file.pdf:page_number:chunk_index".

        Args:
            chunks (list[Document]): A list of Langchain Document objects (chunks).

        Returns:
            list[Document]: The list of chunks with 'id' added to their metadata.
        """
        print("Calculating chunk IDs...")
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            
            # Ensure source and page are available for ID generation
            if source is None or page is None:
                # Fallback for documents without source/page metadata, or handle as error
                import uuid
                chunk_id = str(uuid.uuid4()) # Generate a unique ID if metadata is missing
                print(f"Warning: Chunk missing source or page metadata. Assigning random ID: {chunk_id}")
            else:
                current_page_id = f"{source}:{page}"

                # If the page ID is the same as the last one, increment the index.
                if current_page_id == last_page_id:
                    current_chunk_index += 1
                else:
                    current_chunk_index = 0

                # Calculate the chunk ID.
                chunk_id = f"{current_page_id}:{current_chunk_index}"
                last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id
        print("Chunk IDs calculated.")
        return chunks

    def add_documents(self, chunks: list[Document]):
        """
        Adds new document chunks to the ChromaDB vector store.
        Only chunks that do not already exist in the database are added.

        Args:
            chunks (list[Document]): A list of Langchain Document objects (chunks) to add.
        """
        print("Initializing ChromaDB...")
        db = Chroma(
            persist_directory=self.chroma_path, embedding_function=self.embedding_function
        )

        # Calculate Page IDs for the new chunks.
        chunks_with_ids = self._calculate_chunk_ids(chunks)

        # Get existing IDs from the database.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Filter out chunks that already exist.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()  # Save changes to disk
            print("✅ Documents added and persisted.")
        else:
            print("✅ No new documents to add. Database is up to date.")

    def clear_database(self):
        """
        Clears the entire ChromaDB database by removing its directory.
        """
        print(f"✨ Clearing database at: {self.chroma_path}")
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
            print("Database cleared successfully.")
        else:
            print("Database directory does not exist, nothing to clear.")

def main():
    """
    Main function to parse arguments and orchestrate the vector store operations.
    """
    parser = argparse.ArgumentParser(description="Manage a ChromaDB vector store for RAG.")
    parser.add_argument("--reset", action="store_true", help="Reset (clear) the database before adding documents.")
    args = parser.parse_args()

    # Initialize the vector store manager
    vector_store_manager = ChromaVectorStore()

    # Check if the database should be cleared (using the --reset flag).
    if args.reset:
        vector_store_manager.clear_database()

    # Load, split, and add documents to the data store.
    documents = vector_store_manager._load_documents()
    chunks = vector_store_manager._split_documents(documents)
    vector_store_manager.add_documents(chunks)


if __name__ == "__main__":
    main()