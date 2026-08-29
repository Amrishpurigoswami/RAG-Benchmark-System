from llm.llm_factory import get_llm


class Router:

    def __init__(self):
        self.llm = get_llm()

    def route(self, question):

        prompt = f"""
You are a RAG Router.

Choose ONLY one option:

simple
fusion
self

Rules:

simple:
- Employee IDs
- Names
- Dates
- Salary values
- Direct factual lookup

fusion:
- Policy clauses
- Sections
- Rules
- HR policies
- Document references

self:
- Why questions
- Reasoning questions
- Explanation questions
- Multi-step questions

Question:
{question}

Respond with ONLY:
simple
fusion
self
"""

        decision = (
            self.llm.generate(prompt)
            .strip()
            .lower()
        )

        if decision not in [
            "simple",
            "fusion",
            "self"
        ]:
            decision = "simple"

        return decision
    
