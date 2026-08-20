
from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import DocumentChunker

from embeddings.embedding_model import EmbeddingModel

from vectordb.chroma_db import ChromaDB


class IngestionPipeline:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

        self.loader = PDFLoader(pdf_path)

        self.chunker = DocumentChunker()

        self.embedding = EmbeddingModel()

        self.db = ChromaDB()

    def run(self):

        print("Loading PDF...")

        docs = self.loader.load()

        print(f"Documents Loaded: {len(docs)}")

        chunks = self.chunker.split_documents(docs)

        print(f"Chunks Created: {len(chunks)}")

        self.db.store_documents(chunks)

        print("Vector DB Ready")