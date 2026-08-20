from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    creation_count = 0

    def __init__(self):

        type(self).creation_count += 1

        self.embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

    @classmethod
    def get_creation_count(cls):

        return cls.creation_count

    def get_embedding(self):

        return self.embedding
    