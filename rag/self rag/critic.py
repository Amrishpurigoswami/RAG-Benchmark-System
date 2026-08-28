from llm.llm_factory import get_llm


class Critic:

    def __init__(self):

        self.llm = get_llm()

    def review(
        self,
        question,
        answer,
        context
    ):

        prompt = f"""
You are an answer reviewer.

Question:
{question}

Context:
{context}

Answer:
{answer}

Determine whether the answer
is fully supported by the context.

Respond ONLY with:

PASS

or

FAIL
"""

        result = (
            self.llm.generate(
                prompt
            )
            .strip()
            .upper()
        )

        return result  