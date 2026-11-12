"""Multimodal RAG — Complete pipeline for multimodal graph construction + retrieval.

WITH PAGEINDEX REASONER:
When use_pageindex=True, the query flow uses PageReasoner to:
1. Select relevant document sections BEFORE querying Neo4j
2. Gather richer context from selected sections
3. Feed the LLM with section summaries + graph facts + multimodal evidence

Capabilities:
1. Build: PDF → PageIndex → [Text, Tables, Images, OCR per section] → Graph → Neo4j
2. Query: Question → Page Reasoner → Narrowed Graph Retrieval → LLM Answer
3. Reasoned Query: Question → Page Reasoner → Section Context + Graph Facts → LLM Answer
4. Multi-Document: Question → Document Selection → Section Selection → Graph Retrieval → LLM Answer (Phase 6)

Reuses:
  - graph_validator.py (validation)
  - graph_store.py (Neo4j persistence)
  - graph_retriever.py (Cypher fact retrieval)
  - graph_rag.py (answer generation with Cerebras)
  - llm_config.py (LLM provider config)
  - pageindex/ (PageIndex tree builder + reasoner + search)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rags.graph_rag.graph_rag import GraphQuery
from rags.multimodal_rag.multimodal_graph_builder import MultimodalGraphBuilder

from rags.pageindex.page_index_builder import PageIndexBuilder
from rags.pageindex.page_reasoner import PageReasoner
from rags.pageindex.page_search import PageSearch
from rags.pageindex.page_retriever import PageRetriever

logger = logging.getLogger(__name__)


class MultimodalRAG:
    """Top-level Multimodal Graph RAG system.

    Three query modes:
    1. ask() — Direct graph retrieval (no PageIndex, backward compatible)
    2. ask_with_reasoner() — PageReasoner selects sections first, then retrieves
    3. ask_multidoc() — Multi-document retrieval (Phase 6): document → section → graph → answer

    Usage:
        rag = MultimodalRAG(use_pageindex=True)

        # Build the graph from a PDF (section-based with PageIndex)
        stats = rag.build("data/report.pdf")

        # Ask with page-aware reasoning
        answer = rag.ask_with_reasoner("What is Hemant's salary?")

        # Or use direct graph retrieval
        answer = rag.ask("What is Hemant's salary?")

        # Multi-document retrieval (Phase 6)
        answer = rag.ask_multidoc("What are the top salaries across all documents?")
    """

    def __init__(
        self,
        enable_text: bool = True,
        enable_tables: bool = True,
        enable_images: bool = True,
        enable_ocr: bool = True,
        use_pageindex: bool = True,
    ):
        """
        Args:
            enable_text: Process text content.
            enable_tables: Extract and process tables.
            enable_images: Extract and caption images.
            enable_ocr: OCR scanned content.
            use_pageindex: If True, use PageIndex for both building and querying.
        """
        self.use_pageindex = use_pageindex

        # Graph builder (multimodal ingestion — passes use_pageindex through)
        self.builder = MultimodalGraphBuilder(
            enable_text=enable_text,
            enable_tables=enable_tables,
            enable_images=enable_images,
            enable_ocr=enable_ocr,
            use_pageindex=use_pageindex,
        )

        # Graph query (answer generation — reuses existing GraphQuery)
        self.query_engine = GraphQuery()

        # PageIndex components (for reasoned queries)
        self.page_index_builder = PageIndexBuilder()
        self.page_reasoner = PageReasoner()
        self.page_search = PageSearch()

        # Cached PageIndex trees (populated during build)
        self._page_trees: Dict[str, Dict[str, Any]] = {}

        # Ingestion stats
        self.last_build_stats: Dict[str, Any] = {}

    # ==============================================================
    # Build
    # ==============================================================

    def build(self, pdf_path: str | Path) -> Dict[str, Any]:
        """Ingest a PDF into the multimodal knowledge graph.

        When use_pageindex=True, also caches the PageIndex tree for reasoned queries.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Build statistics.
        """
        pdf_path = Path(pdf_path)
        print(f"\n{'='*60}")
        print(f"  Multimodal RAG — Building from: {pdf_path.name}")
        print(f"  Mode: PageIndex (section-based)")
        print(f"{'='*60}")

        self.last_build_stats = self.builder.build(pdf_path)

        if self.use_pageindex:
            try:
                tree = self.page_index_builder.load(pdf_path.stem)
                if tree:
                    self._page_trees[pdf_path.stem] = tree
                    print(f"  PageIndex tree cached for reasoned queries: {pdf_path.stem}")
            except Exception as e:
                logger.warning(f"Failed to cache PageIndex tree: {e}")

        print(f"\n{'='*60}")
        print(f"  Build Complete")
        print(f"{'='*60}")
        print(f"  Total entities stored:      {self.last_build_stats.get('total_entities_stored', 0)}")
        print(f"  Total relationships stored: {self.last_build_stats.get('total_relationships_stored', 0)}")
        if self.use_pageindex:
            print(f"  Sections processed:          {self.last_build_stats.get('sections_processed', 0)}")

        return self.last_build_stats

    def build_all(self, pdf_paths: List[str | Path]) -> Dict[str, Any]:
        """Build graphs from multiple PDFs.

        Args:
            pdf_paths: List of PDF file paths.

        Returns:
            Combined build statistics.
        """
        combined_stats = {
            "total_entities_stored": 0,
            "total_relationships_stored": 0,
            "documents_processed": 0,
        }

        for pdf_path in pdf_paths:
            stats = self.build(pdf_path)
            combined_stats["total_entities_stored"] += stats.get("total_entities_stored", 0)
            combined_stats["total_relationships_stored"] += stats.get("total_relationships_stored", 0)
            combined_stats["documents_processed"] += 1

            for key in ["text_chunks", "text_entities", "text_relationships",
                         "tables_found", "table_entities", "table_relationships",
                         "images_found", "image_entities", "image_relationships",
                         "ocr_pages", "ocr_entities", "ocr_relationships",
                         "sections_processed"]:
                if key not in combined_stats:
                    combined_stats[key] = 0
                combined_stats[key] += stats.get(key, 0)

        self.last_build_stats = combined_stats
        return combined_stats

    def _load_all_trees(self) -> None:
        """Load all cached PageIndex trees from profiles/ directory."""
        profile_dir = Path("profiles")
        if not profile_dir.is_dir():
            return

        for json_path in profile_dir.glob("*_pageindex.json"):
            try:
                tree = json.loads(json_path.read_text(encoding="utf-8"))
                stem = json_path.stem.replace("_pageindex", "")
                self._page_trees[stem] = tree
                print(f"  Loaded PageIndex tree: {stem}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load PageIndex tree {json_path.name}: {e}")

    # ==============================================================
    # Direct Query (backward compatible)
    # ==============================================================

    def ask(
        self,
        question: str,
        top_k: int = 20,
        include_multimodal_context: bool = True,
    ) -> Dict[str, Any]:
        """Ask a question using direct graph retrieval (no PageIndex).

        This is the backward-compatible query method. For page-aware retrieval,
        use ask_with_reasoner() instead. For multi-document retrieval, use
        ask_multidoc().

        Args:
            question: Natural language question.
            top_k: Max number of graph facts to retrieve.
            include_multimodal_context: If True, includes multimodal metadata.

        Returns:
            Dict with question, answer, facts, and metadata.
        """
        print(f"\n{'='*60}")
        print(f"  [Direct Graph Retrieval] Question: {question}")
        print(f"{'='*60}")

        result = self.query_engine.ask(question, top_k)

        result["rag_type"] = "multimodal_graph"
        result["modalities"] = self._detect_modalities_in_result(result)

        return result

    # ==============================================================
    # Reasoned Query (PageIndex-aware)
    # ==============================================================

    def ask_with_reasoner(
        self,
        question: str,
        top_k: int = 20,
        max_sections: int = 3,
        max_documents: int = 3,
        include_section_context: bool = True,
        include_graph_facts: bool = True,
    ) -> Dict[str, Any]:
        """Ask a question with PageReasoner — selects relevant sections first.

        Flow:
        1. Page Reasoner scans all cached PageIndex trees
        2. Selects most relevant documents, then sections within those documents
        3. Gathers section context (summaries + text)
        4. Retrieves graph facts with section-aware queries
        5. Feeds LLM with: question + section summaries + graph facts

        Args:
            question: Natural language question.
            top_k: Max graph facts to retrieve.
            max_sections: Max sections to consider.
            max_documents: Max documents to consider.
            include_section_context: Include section summaries in answer prompt.
            include_graph_facts: Include Neo4j graph facts in answer prompt.

        Returns:
            Dict with question, answer, facts, selected_sections, and metadata.
        """
        print(f"\n{'='*60}")
        print(f"  [Page Reasoner] Question: {question}")
        print(f"{'='*60}")

        if not self._page_trees:
            print(f"  ⚠️  No PageIndex trees cached. Loading from profiles/...")
            self._load_all_trees()

        if not self._page_trees:
            print(f"  ⚠️  No PageIndex trees available. Falling back to direct retrieval.")
            return self.ask(question, top_k)

        # Stage 1: Document + Section selection via PageReasoner
        selected_sections = self.page_reasoner.select_sections(
            question=question,
            trees=self._page_trees,
            max_sections=max_sections,
            max_documents=max_documents,
        )

        if not selected_sections:
            print(f"  ⚠️  No relevant sections found. Falling back to direct retrieval.")
            return self.ask(question, top_k)

        # Stage 2: Gather section context
        section_context = ""
        if include_section_context:
            section_context = self.page_reasoner.gather_context(
                selected_sections=selected_sections,
                trees=self._page_trees,
                include_summaries=True,
            )

        # Stage 3: Retrieve graph facts using section-aware queries
        retrieved_facts: List[str] = []
        if include_graph_facts:
            page_retriever = PageRetriever()
            retrieved_facts = page_retriever.retrieve(
                question=question,
                limit=top_k,
                sections=selected_sections,
            )
            page_retriever.close()
            logger.info(
                "[PageRetriever] Retrieved %d section-aware facts from %d section(s)",
                len(retrieved_facts),
                len(selected_sections),
            )

        # Stage 4: Build enriched prompt and generate answer
        enriched_answer = self._generate_with_section_context(
            question=question,
            section_context=section_context,
            graph_facts=retrieved_facts,
        )

        result = {
            "question": question,
            "answer": enriched_answer,
            "facts": retrieved_facts,
            "selected_sections": selected_sections,
            "section_context": section_context[:1000] + "..." if len(section_context) > 1000 else section_context,
            "rag_type": "multimodal_graph_reasoned",
            "modalities": self._detect_modalities_in_result({"facts": retrieved_facts}),
        }

        print(f"\n  Selected {len(selected_sections)} section(s) across {len(self._page_trees)} document(s)")
        for sec in selected_sections:
            pages = f"pp. {sec.get('start_index')}-{sec.get('end_index')}"
            print(f"    [{sec['document']}] {sec['title']} {pages}")
        print(f"  Graph facts retrieved: {len(retrieved_facts)}")
        print(f"\n  Answer: {enriched_answer[:200]}...")

        return result

    # ==============================================================
    # Phase 6: Multi-Document Retrieval
    # ==============================================================

    def ask_multidoc(
        self,
        question: str,
        top_k: int = 20,
        max_documents: int = 3,
        max_sections: int = 5,
        include_section_context: bool = True,
        include_graph_facts: bool = True,
    ) -> Dict[str, Any]:
        """Multi-document retrieval (Phase 6): optimized for 7+ PDFs.

        Three-stage strategy:
        1. Document Selection: Which PDFs are relevant? (PageReasoner compares question vs doc_description)
        2. Section Selection: Which sections within those PDFs? (PageReasoner compares question vs section summaries)
        3. Graph Retrieval: Narrowed Cypher queries filtered by document + section_id

        This is the primary retrieval strategy for multi-document scenarios.

        Args:
            question: Natural language question.
            top_k: Max graph facts to retrieve.
            max_documents: Max documents to consider (default 3).
            max_sections: Max sections to consider (default 5).
            include_section_context: Include section summaries in answer prompt.
            include_graph_facts: Include Neo4j graph facts in answer prompt.

        Returns:
            Dict with question, answer, facts, selected_documents, selected_sections, and metadata.
        """
        print(f"\n{'='*60}")
        print(f"  [Multi-Doc Retrieval] Question: {question}")
        print(f"{'='*60}")

        if not self._page_trees:
            print(f"  ⚠️  No PageIndex trees cached. Loading from profiles/...")
            self._load_all_trees()

        if not self._page_trees:
            print(f"  ⚠️  No PageIndex trees available. Falling back to direct retrieval.")
            return self.ask(question, top_k)

        # Stage 1: Document + Section selection
        selected_sections = self.page_reasoner.select_sections(
            question=question,
            trees=self._page_trees,
            max_sections=max_sections,
            max_documents=max_documents,
        )

        if not selected_sections:
            print(f"  ⚠️  No relevant sections found. Falling back to direct retrieval.")
            return self.ask(question, top_k)

        # Stage 2: Gather section context
        section_context = ""
        if include_section_context:
            section_context = self.page_reasoner.gather_context(
                selected_sections=selected_sections,
                trees=self._page_trees,
                include_summaries=True,
            )

        # Stage 3: Retrieve graph facts using section-aware queries
        retrieved_facts: List[str] = []
        if include_graph_facts:
            page_retriever = PageRetriever()
            retrieved_facts = page_retriever.retrieve(
                question=question,
                limit=top_k,
                sections=selected_sections,
            )
            page_retriever.close()
            logger.info(
                "[PageRetriever] Retrieved %d multi-doc section-aware facts from %d section(s)",
                len(retrieved_facts),
                len(selected_sections),
            )

        # Stage 4: Build enriched prompt and generate answer
        enriched_answer = self._generate_with_section_context(
            question=question,
            section_context=section_context,
            graph_facts=retrieved_facts,
        )

        # Identify which documents were used
        used_docs = sorted(set(sec.get("document", "") for sec in selected_sections))

        result = {
            "question": question,
            "answer": enriched_answer,
            "facts": retrieved_facts,
            "selected_documents": used_docs,
            "selected_sections": selected_sections,
            "section_context": section_context[:1000] + "..." if len(section_context) > 1000 else section_context,
            "rag_type": "multimodal_multidoc",
            "modalities": self._detect_modalities_in_result({"facts": retrieved_facts}),
            "documents_searched": len(self._page_trees),
            "documents_selected": len(used_docs),
            "sections_selected": len(selected_sections),
        }

        print(f"\n  Multi-doc retrieval complete:")
        print(f"    Documents searched: {len(self._page_trees)}")
        print(f"    Documents selected: {len(used_docs)}")
        print(f"    Sections selected:  {len(selected_sections)}")
        print(f"    Graph facts:        {len(retrieved_facts)}")
        print(f"\n  Answer: {enriched_answer[:200]}...")

        return result

    # ==============================================================
    # Internal helpers
    # ==============================================================

    def _generate_with_section_context(
        self,
        question: str,
        section_context: str,
        graph_facts: List[str],
    ) -> str:
        """Generate answer using enriched context (section summaries + graph facts)."""
        facts_text = "\n\n".join(
            f"[{i + 1}] {fact}" for i, fact in enumerate(graph_facts)
        ) if graph_facts else "No graph facts retrieved."

        prompt = f"""
You are an expert Question Answering assistant with access to document sections and graph facts.

Answer the user's question using:
1. The relevant document section summaries (provide context)
2. The graph facts (provide precise data)

Do NOT invent information. Do NOT use outside knowledge.
If the answer is not supported by the provided information, reply:
"I could not find the answer in the available information."

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
RELEVANT DOCUMENT SECTIONS
--------------------------------------------------

{section_context if section_context else "No specific sections identified."}

--------------------------------------------------
GRAPH FACTS
--------------------------------------------------

{facts_text}

--------------------------------------------------
OUTPUT RULES
--------------------------------------------------

• Answer in 2-5 sentences.
• Quote important values exactly.
• Preserve names, dates, IDs, and salary figures exactly.
• Do not mention graph facts, sections, or chunks.
• Return only the final answer.
"""

        try:
            client = self.query_engine.client
            if client is None:
                return graph_facts[0] if graph_facts else "I could not find the answer."

            model = self.query_engine.model_fallbacks[0]
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            return content or (graph_facts[0] if graph_facts else "I could not find the answer.")
        except Exception as e:
            logger.warning(f"Enriched answer generation failed: {e}")
            return graph_facts[0] if graph_facts else "I could not find the answer."

    @staticmethod
    def _detect_modalities_in_result(result: Dict[str, Any]) -> List[str]:
        """Detect which modalities contributed to the answer."""
        modalities = ["text"]
        facts = result.get("facts", [])

        for fact in facts:
            fact_lower = fact.lower()
            if "table" in fact_lower or "column" in fact_lower or "row" in fact_lower:
                modalities.append("table")
            if "image" in fact_lower or "caption" in fact_lower or "diagram" in fact_lower:
                modalities.append("image")
            if "ocr" in fact_lower:
                modalities.append("ocr")

        seen = set()
        unique = []
        for m in modalities:
            if m not in seen:
                seen.add(m)
                unique.append(m)

        return unique

    def get_stats(self) -> Dict[str, Any]:
        """Return build statistics from the last build operation."""
        return self.last_build_stats

    def print_report(self) -> None:
        """Print a formatted report of the last build."""
        stats = self.last_build_stats
        if not stats:
            print("No build has been run yet.")
            return

        print("\n" + "=" * 60)
        print("  MULTIMODAL RAG — BUILD REPORT")
        print("=" * 60)
        print(f"  Documents processed:      {stats.get('documents_processed', 1)}")
        print(f"  Sections processed:        {stats.get('sections_processed', 0)}")
        print(f"  Text entities extracted:  {stats.get('text_entities', 0)}")
        print(f"  Text relationships:       {stats.get('text_relationships', 0)}")
        print(f"  Tables found:             {stats.get('tables_found', 0)}")
        print(f"  Table entities:           {stats.get('table_entities', 0)}")
        print(f"  Table relationships:      {stats.get('table_relationships', 0)}")
        print(f"  Images extracted:         {stats.get('images_found', 0)}")
        print(f"  Image entities:           {stats.get('image_entities', 0)}")
        print(f"  Image relationships:      {stats.get('image_relationships', 0)}")
        print(f"  OCR pages processed:      {stats.get('ocr_pages', 0)}")
        print(f"  OCR entities:             {stats.get('ocr_entities', 0)}")
        print(f"  OCR relationships:        {stats.get('ocr_relationships', 0)}")
        print(f"  ─────────────────────────────────────")
        print(f"  Total entities stored:    {stats.get('total_entities_stored', 0)}")
        print(f"  Total relationships stored: {stats.get('total_relationships_stored', 0)}")
        print("=" * 60)


# ==============================================================
# Standalone Demo
# ==============================================================
if __name__ == "__main__":
    from pathlib import Path

    rag = MultimodalRAG(
        enable_text=True,
        enable_tables=True,
        enable_images=True,
        enable_ocr=True,
        use_pageindex=True,
    )

    pdf_path = Path("data/hemant_story.pdf")
    if pdf_path.is_file():
        rag.build(pdf_path)

        questions = [
            "What is Hemant Sharma's employee ID?",
            "Who is Hemant's reporting manager?",
            "What is Hemant's monthly salary?",
        ]

        for q in questions:
            print(f"\n\n{'='*70}")
            print(f"QUESTION: {q}")
            print(f"{'='*70}")

            result = rag.ask_with_reasoner(q)
            print(f"\nANSWER:")
            print(result["answer"])
            print(f"\n  Sections used: {len(result.get('selected_sections', []))}")
            print(f"  Facts retrieved: {len(result.get('facts', []))}")
            print(f"  RAG type: {result.get('rag_type')}")