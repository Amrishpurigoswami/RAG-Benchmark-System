from vectordb.chroma_db import ChromaDB

from llm.llm_factory import get_llm

from rags.fusion_rag.hyde import HyDE

from rags.fusion_rag.rrf import RRF


class FusionRAG:

    def __init__(self):

        self.retriever = (
            ChromaDB()
            .get_retriever()
        )

        self.llm = get_llm()

        self.hyde = HyDE()

        self.rrf = RRF()

    def ask(self,question):
        # Retrieval 1
        vector_docs = (
            self.retriever.invoke(
                question
            )
        )
        # Retrieval 2 (HyDE)
        hypothetical_answer = (
            self.hyde
            .generate_hypothetical_answer(
                question
            )
        )
        
        hyde_docs = (
            self.retriever.invoke(
                hypothetical_answer
            )
        )
        # Fusion of retrieval results
        fused_chunks = (
            self.rrf.fuse(
                [
                    vector_docs,
                    hyde_docs
                ]
            )
        )

        context = "\n\n".join(
            fused_chunks[:5]
        )

        prompt = f"""
Answer only from context.

Context:
{context}

Question:
{question}
"""

        answer = (
            self.llm.generate(
                prompt
            )
        )

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "retrieved_chunks":
            fused_chunks[:5]
        }