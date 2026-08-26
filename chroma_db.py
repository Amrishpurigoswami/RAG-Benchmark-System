from langchain_chroma import Chroma

from embeddings.embedding_model import (
    EmbeddingModel
)


class ChromaDB:

    def __init__(self):

        self.embedding = (
            EmbeddingModel()
            .get_embedding()
        )

    def store_documents(self, chunks):

        Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory="db"
        )

    def get_retriever(self):

        db = Chroma(
            persist_directory="db",
            embedding_function=self.embedding
        )

        return db.as_retriever(
            search_kwargs={"k":10}
        )