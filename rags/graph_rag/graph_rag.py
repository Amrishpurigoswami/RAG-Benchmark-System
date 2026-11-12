import logging
import os
import time
from typing import Any, Dict, List, Optional

from rags.graph_rag.graph_retriever import GraphRetriever
from rags.graph_rag.llm_config import get_answer_client, get_answer_models

logger = logging.getLogger(__name__)


class GraphQuery:
    """
    ==========================================================
    GraphQuery — Synchronous answer generation

    Responsibilities
    ----------------
    1. Receive user's question.
    2. Ask GraphRetriever for graph facts.
    3. Build a clean prompt.
    4. Send prompt to LLM.
    5. Return final answer.

    This class NEVER talks directly to Neo4j.

    Neo4j
        ↑
    GraphStore
        ↑
    GraphRetriever
        ↑
    GraphQuery
        ↑
       User
    ==========================================================
    """

    def __init__(self):
        self.retriever = GraphRetriever()

        self.model_fallbacks = get_answer_models()
        self.use_answer_llm = os.getenv("ANSWER_USE_LLM", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }

        self.client = None
        try:
            self.client = get_answer_client()
        except Exception as e:
            logger.warning("Answer client initialization failed: %s", e)
            self.client = None

    # ======================================================
    # Prompt Builder
    # ======================================================

    def build_prompt(
        self,
        question: str,
        retrieved_facts: List[str],
    ) -> str:
        facts = "\n\n".join(
            f"[{i + 1}] {fact}" for i, fact in enumerate(retrieved_facts)
        )

        return f"""
You are an expert Question Answering assistant.

Answer ONLY from the graph facts provided below.

If multiple facts together answer the question,
combine them carefully.

Do NOT invent information.

Do NOT use outside knowledge.

If the answer is not supported by the graph facts,
reply exactly:

I could not find the answer in the graph.

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
GRAPH FACTS
--------------------------------------------------

{facts}

--------------------------------------------------
OUTPUT RULES
--------------------------------------------------

If the user asks for a specific word count, follow it as closely as possible.
• Answer in 2-5 sentences.
• Quote important values exactly.
• Preserve names exactly.
• Preserve dates exactly.
• Preserve employee IDs exactly.
• Preserve salary figures exactly.
• Preserve section numbers exactly.
• Do not mention graph facts or chunks.
• Return only the final answer.
""".strip()


    # ======================================================
    # Main Query Function
    # ======================================================
    
    def ask(
        self,
        question: str,
        top_k: int = 15,
    ) -> Dict[str, Any]:
        # Retrieve graph facts synchronously
        retrieved_facts = self.retriever.retrieve(question, top_k)

        # If no LLM client is available, return first fact as answer
        if self.client is None or not self.use_answer_llm:
            return {
                "question": question,
                "answer": retrieved_facts[0] if retrieved_facts else "I could not find the answer in the graph.",
                "facts": retrieved_facts,
            }

        prompt = self.build_prompt(question, retrieved_facts)

        # Retry small transient failures (rate limits / occasional transport errors).
        last_err: Optional[Exception] = None

        for model in self.model_fallbacks:
            for attempt in range(1, 4):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                    )

                    choice0 = response.choices[0]
                    message = choice0.message
                    content = getattr(message, "content", None)
                    if not content:
                        raise RuntimeError("LLM response missing content")

                    return {
                        "question": question,
                        "answer": content,
                        "facts": retrieved_facts,
                    }
                except Exception as e:
                    last_err = e
                    time.sleep(min(2 ** attempt, 4))

        # Facts were retrieved, so do not incorrectly claim that the graph has
        # no answer merely because the answer provider failed.
        return {
            "question": question,
            "answer": retrieved_facts[0] if retrieved_facts else "I could not find the answer in the graph.",
            "facts": retrieved_facts,
            "error": str(last_err) if last_err else None,
        }

    # ======================================================
    # Close Neo4j Connection
    # ======================================================

    def close(self):
        self.retriever.close()


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":
    graph = GraphQuery()
    result = graph.ask("What is Hemant Sharma's employee ID?")

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(result["question"])

    print("\n" + "=" * 80)
    print("GRAPH FACTS")
    print("=" * 80)
    for i, fact in enumerate(result["facts"], start=1):
        print(f"[{i}] {fact}")

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["answer"])