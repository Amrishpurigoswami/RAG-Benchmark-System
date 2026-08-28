from vectordb.chroma_db import ChromaDB
from llm.llm_factory import get_llm


class SimpleRAG:

    def __init__(self):

        self.retriever = (
            ChromaDB()
            .get_retriever()
        )

        self.llm = get_llm()

    def ask(self, question):

        # Retrieve relevant chunks
        docs = self.retriever.invoke(question)

        retrieved_chunks = [doc.page_content for doc in docs]

        # Combine retrieved context
        context = "\n\n".join(
            retrieved_chunks
        )

        prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
        
        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "retrieved_chunks": retrieved_chunks
            
        }
