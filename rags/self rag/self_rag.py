from vectordb.chroma_db import ChromaDB

from llm.llm_factory import get_llm

from rags.self_rag.critic import Critic


class SelfRAG:

    def __init__(self):

        self.retriever = (
            ChromaDB()
            .get_retriever()
        )

        self.llm = get_llm()

        self.critic = Critic()

    def ask(
        self,
        question
    ):

        docs = (
            self.retriever.invoke(
                question
            )
        )

        context = "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )

        prompt = f"""
Answer ONLY using context.

Context:
{context}

Question:
{question}
"""

        draft_answer = (
            self.llm.generate(
                prompt
            )
        )

        review = (
            self.critic.review(
                question,
                draft_answer,
                context
            )
        )

        if review == "PASS":

            final_answer = (
                draft_answer
            )

        else:

            correction_prompt = f"""
The previous answer may not be
fully grounded in the context.

Context:
{context}

Question:
{question}

Provide a corrected answer
using ONLY the context.
"""

            final_answer = (
                self.llm.generate(
                    correction_prompt
                )
            )

        return {
            "question": question,
            "answer": final_answer,
            "context": context,
            "review": review
        }