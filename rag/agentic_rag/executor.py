from vectordb.chroma_db import ChromaDB


class Executor:

    def __init__(self):

        self.retriever = ChromaDB().get_retriever()

    def _extract_steps(self, plan):

      queries = []

      collect = False

      for line in plan.splitlines():

        line = line.strip()

        if line.upper().startswith("SEARCH QUERIES"):

            collect = True

            continue

        if collect:

            if (
                line.startswith("1.")
                or line.startswith("2.")
                or line.startswith("3.")
                or line.startswith("4.")
                or line.startswith("5.")
            ):

                queries.append(
                    line.split(".",1)[1].strip()
                )

      return queries

    def execute(
        self,
        question,
        plan
    ):

        retrieval_queries = self._extract_steps(plan)

        if not retrieval_queries:
            retrieval_queries = [question]

        all_docs = []

        seen = set()

        for query in retrieval_queries:

            docs = self.retriever.invoke(query)

            for doc in docs:

                text = doc.page_content.strip()

                if text not in seen:

                    seen.add(text)

                    all_docs.append(doc)

        context = "\n\n".join(
            [
                doc.page_content
                for doc in all_docs
            ]
        )

        return {

            "question": question,

            "plan": plan,

            "documents": all_docs,

            "context": context,

            "retrieved_chunks": len(all_docs),

            "queries_used": retrieval_queries

        }