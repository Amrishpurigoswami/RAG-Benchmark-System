import time
import os
import asyncio
from pathlib import Path
from rags.simple_rag.simple_rag import SimpleRAG
from rags.fusion_rag.fusion_rag import FusionRAG
from rags.self_rag.self_rag import SelfRAG
from rags.adaptive_rag.adaptive_rag import AdaptiveRAG
from rags.agentic_rag.agentic_rag import AgenticRAG
from rags.graph_rag.graph_rag import GraphQuery
from rags.multimodal_rag.multimodal_rag import MultimodalRAG

QUESTION_FILE = "data/questions.txt"
REPORT_FILE = "reports/benchmark_report.md"


class Benchmark:

    def __init__(self):
        print("Initializing standard RAG models...")
        self.simple_rag = SimpleRAG()
        self.fusion_rag = FusionRAG()
        self.self_rag = SelfRAG()
        self.adaptive_rag = AdaptiveRAG()
        self.agentic_rag = AgenticRAG()

        print("Initializing Graph and Multimodal RAG models...")
        self.graph_rag = GraphQuery()
        self.multimodal_rag_pageindex = MultimodalRAG(use_pageindex=True)
        self.multimodal_rag_legacy = MultimodalRAG(use_pageindex=False)

    def load_questions(self):
        if not os.path.exists(QUESTION_FILE):
            default_questions = [
                "What is Hemant Sharma's employee ID?",
                "Who is Hemant's reporting manager?",
                "What was Hemant's monthly salary?",
                "What is Clause 6.2(e)?",
                "Why was Hemant's bonus withheld?"
            ]
            os.makedirs(os.path.dirname(QUESTION_FILE), exist_ok=True)
            with open(QUESTION_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(default_questions))
            return default_questions

        with open(QUESTION_FILE, "r", encoding="utf-8") as file:
            questions = [
                line.strip()
                for line in file.readlines()
                if line.strip()
            ]
        return questions

    def run(self):
        questions = self.load_questions()
        results = []

        print("\n" + "=" * 100)
        print("BENCHMARK STARTED")
        print("=" * 100)

        for index, question in enumerate(questions, start=1):
            print("\n" + "=" * 100)
            print(f"QUESTION {index}: {question}")
            print("=" * 100)

            q_results = {"question": question, "runs": {}}

            # 1. Simple RAG
            print("\nRunning Simple RAG...")
            start = time.perf_counter()
            try:
                ans = self.simple_rag.ask(question)
                latency = time.perf_counter() - start
                q_results["runs"]["Simple RAG"] = {"answer": ans["answer"], "latency": latency, "facts": 0}
            except Exception as e:
                q_results["runs"]["Simple RAG"] = {"answer": f"Error: {e}", "latency": 0.0, "facts": 0}

            # 2. Fusion RAG
            print("Running Fusion RAG...")
            start = time.perf_counter()
            try:
                ans = self.fusion_rag.ask(question)
                latency = time.perf_counter() - start
                q_results["runs"]["Fusion RAG"] = {"answer": ans["answer"], "latency": latency, "facts": 0}
            except Exception as e:
                q_results["runs"]["Fusion RAG"] = {"answer": f"Error: {e}", "latency": 0.0, "facts": 0}

            # 3. Self RAG
            print("Running Self RAG...")
            start = time.perf_counter()
            try:
                ans = self.self_rag.ask(question)
                latency = time.perf_counter() - start
                q_results["runs"]["Self RAG"] = {"answer": ans["answer"], "latency": latency, "facts": 0}
            except Exception as e:
                q_results["runs"]["Self RAG"] = {"answer": f"Error: {e}", "latency": 0.0, "facts": 0}

            # 4. Adaptive RAG
            print("Running Adaptive RAG...")
            start = time.perf_counter()
            try:
                ans = self.adaptive_rag.ask(question)
                latency = time.perf_counter() - start
                q_results["runs"]["Adaptive RAG"] = {"answer": ans["answer"], "latency": latency, "facts": 0}
            except Exception as e:
                q_results["runs"]["Adaptive RAG"] = {"answer": f"Error: {e}", "latency": 0.0, "facts": 0}

            # 5. Agentic RAG
            print("Running Agentic RAG...")
            start = time.perf_counter()
            try:
                ans = self.agentic_rag.ask(question)
                latency = time.perf_counter() - start
                q_results["runs"]["Agentic RAG"] = {"answer": ans["answer"], "latency": latency, "facts": 0}
            except Exception as e:
                q_results["runs"]["Agentic RAG"] = {"answer": f"Error: {e}", "latency": 0.0, "facts": 0}

            # 6. Graph RAG (Standard)
            print("Running Graph RAG...")
            start = time.perf_counter()
            try:
                ans = self.graph_rag.ask(question, top_k=20)
                latency = time.perf_counter() - start
                facts_count = len(ans.get("facts", []))
                q_results["runs"]["Graph RAG"] = {"answer": ans["answer"], "latency": latency, "facts": facts_count}
            except Exception as e:
                q_results["runs"]["Graph RAG"] = {"answer": f"Error: {e}", "latency": 0.0, "facts": 0}

            # 7. Multimodal RAG (PageIndex)
            print("Running Multimodal RAG (PageIndex)...")
            start = time.perf_counter()
            try:
                ans = asyncio.run(self.multimodal_rag_pageindex.ask_with_reasoner(question, top_k=20))
                latency = time.perf_counter() - start
                facts_count = len(ans.get("facts", []))
                q_results["runs"]["Multimodal RAG (PageIndex)"] = {"answer": ans["answer"], "latency": latency, "facts": facts_count}
            except Exception as e:
                q_results["runs"]["Multimodal RAG (PageIndex)"] = {"answer": f"Error: {e}", "latency": 0.0, "facts": 0}

            # 8. Multimodal RAG (Legacy)
            print("Running Multimodal RAG (Legacy)...")
            start = time.perf_counter()
            try:
                ans = self.multimodal_rag_legacy.ask(question, top_k=20)
                latency = time.perf_counter() - start
                facts_count = len(ans.get("facts", []))
                q_results["runs"]["Multimodal RAG (Legacy)"] = {"answer": ans["answer"], "latency": latency, "facts": facts_count}
            except Exception as e:
                q_results["runs"]["Multimodal RAG (Legacy)"] = {"answer": f"Error: {e}", "latency": 0.0, "facts": 0}

            results.append(q_results)

            for model_name, info in q_results["runs"].items():
                print(f"\n--- {model_name} (latency: {info['latency']:.2f}s, facts: {info['facts']}) ---")
                print(info["answer"])

        self.generate_report(results)

        print("\n" + "=" * 100)
        print("BENCHMARK COMPLETED")
        print("=" * 100)

    def generate_report(self, results):
        os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

        md = []
        md.append("# RAG Models Benchmark Evaluation Report")
        md.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        md.append("This report evaluates and compares **8 RAG variants** across the benchmark questions.\n")

        md.append("## Executive Comparative Summary")
        md.append("| Question | RAG Model | Latency (s) | Facts Used | Answer Preview |")
        md.append("| --- | --- | --- | --- | --- |")

        for idx, res in enumerate(results, start=1):
            q_short = res["question"]
            if len(q_short) > 40:
                q_short = q_short[:37] + "..."

            for model_name, info in res["runs"].items():
                ans_preview = info["answer"].replace("\n", " ").strip()
                if len(ans_preview) > 60:
                    ans_preview = ans_preview[:57] + "..."
                md.append(f"| Q{idx}: {q_short} | {model_name} | {info['latency']:.2f}s | {info['facts']} | {ans_preview} |")

        md.append("\n## Detailed Question by Question Breakdown")

        for idx, res in enumerate(results, start=1):
            md.append(f"\n### Question {idx}: {res['question']}\n")
            for model_name, info in res["runs"].items():
                md.append(f"#### {model_name}")
                md.append(f"- **Latency**: {info['latency']:.3f} seconds")
                md.append(f"- **Facts/Context Nodes Used**: {info['facts']}")
                md.append("\n**Response**:")
                md.append(f"{info['answer']}\n")
                md.append("---")

        Path(REPORT_FILE).write_text("\n".join(md), encoding="utf-8")
        print(f"\nBenchmark report successfully written to: {REPORT_FILE}")
