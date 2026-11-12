from llm.llm_factory import get_llm

from rags.agentic_rag.planner import Planner
from rags.agentic_rag.executor import Executor
from rags.agentic_rag.memory import Memory
from rags.agentic_rag.verifier import Verifier


class AgenticRAG:

    def __init__(self):

        self.llm = get_llm()

        self.planner = Planner()

        self.executor = Executor()

        self.memory = Memory()

        self.verifier = Verifier()

    def ask(self, question):

        self.memory.clear()

        retrieval_attempts = 1

        # -----------------------------
        # STEP 1 : PLAN
        # -----------------------------

        plan_result = self.planner.create_plan(question)

        plan = plan_result["plan"]

        self.memory.add(
            "PLAN",
            plan
        )

        # -----------------------------
        # STEP 2 : EXECUTE
        # -----------------------------

        execution = self.executor.execute(
            question,
            plan
        )

        context = execution["context"]

        retrieved_chunks = execution["retrieved_chunks"]

        self.memory.add(
            "CONTEXT",
            context
        )

        # -----------------------------
        # STEP 3 : GENERATE ANSWER
        # -----------------------------

        prompt = f"""
You are an intelligent AI assistant.

Use ONLY the retrieved context.

If the answer is unavailable,
say:

'I could not find the answer in the provided document.'

Context:

{context}

Question:

{question}
"""

        answer = self.llm.generate(prompt)

        self.memory.add(
            "DRAFT ANSWER",
            answer
        )

        # -----------------------------
        # STEP 4 : VERIFY
        # -----------------------------

        verification = self.verifier.verify(
            question,
            answer,
            context
        )

        # -----------------------------
        # STEP 5 : SECOND RETRIEVAL
        # -----------------------------

        if verification["status"] == "NOT_SUPPORTED":

            retrieval_attempts += 1

            execution = self.executor.execute(
                question,
                plan
            )

            context = execution["context"]

            retrieved_chunks = execution["retrieved_chunks"]

            self.memory.add(
                "SECOND CONTEXT",
                context
            )

            retry_prompt = f"""
The previous answer was not fully supported.

Generate a new answer using ONLY
the following context.

Context:

{context}

Question:

{question}
"""

            answer = self.llm.generate(
                retry_prompt
            )

            verification = self.verifier.verify(
                question,
                answer,
                context
            )

            self.memory.add(
                "FINAL ANSWER",
                answer
            )

        else:

            self.memory.add(
                "FINAL ANSWER",
                answer
            )

        return {

            "question": question,

            "plan": plan,

            "queries_used": execution["queries_used"],

            "retrieved_chunks": retrieved_chunks,

            "verification": verification["status"],

            "retrieval_attempts": retrieval_attempts,

            "answer": answer,

            "memory": self.memory.get_history()
        }