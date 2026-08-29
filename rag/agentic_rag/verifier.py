from llm.llm_factory import get_llm


class Verifier:

    def __init__(self):

        self.llm = get_llm()

    def verify(
        self,
        question,
        answer,
        context
    ):

        prompt = f"""
You are an AI Verifier.

Your job is to verify whether the answer is completely supported by the retrieved context.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Instructions:

1. Compare the answer with the context.

2. If every important statement in the answer is supported by the context,
respond with:

SUPPORTED

3. If any important statement is missing, incorrect, or hallucinated,
respond with:

NOT_SUPPORTED

Respond with ONLY one word:

SUPPORTED

or

NOT_SUPPORTED
"""

        result = (
            self.llm.generate(prompt)
            .strip()
            .upper()
        )

        if "SUPPORTED" in result and "NOT_SUPPORTED" not in result:
            status = "SUPPORTED"
        else:
            status = "NOT_SUPPORTED"

        return {
            "status": status,
            "raw_response": result
        }
    
    