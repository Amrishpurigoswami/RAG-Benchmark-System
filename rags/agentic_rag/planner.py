from llm.llm_factory import get_llm

class Planner:

    def __init__(self):
        self.llm = get_llm()

    def create_plan(self, question):

        prompt = f"""
You are an AI Retrieval Planner.

The system contains ONLY one indexed PDF stored in a vector database.

There are:
- No SQL databases
- No employee databases
- No websites
- No internet
- No APIs

Your task is ONLY to generate search queries that can retrieve relevant chunks
from the vector database.

Rules:

1. Do NOT write things like:
   - Search database
   - Find employee record
   - Access company system

2. Generate 3-5 short semantic search queries.

3. Every query should contain important keywords likely to exist in the PDF.

4. Queries should be short (2-8 words).

Return ONLY this format.

GOAL:
<one sentence>

SEARCH QUERIES:
1.
2.
3.
4.
Question:
{question}
"""

        plan = self.llm.generate(prompt)

        return {
            "question": question,
            "plan": plan
        }
    
    
#❌ It does not answer the question.
#❌ It does not retrieve any documents.
#✅ It only creates a plan for the next components.