"""Page Reasoner — LLM-based section selector for intelligent retrieval.

Takes a user question + PageIndex tree(s) for all PDFs, uses LLM reasoning
to identify which sections/nodes are relevant to the question.

This replaces the "search entire graph" approach with a targeted, section-aware
retrieval that:

1. Scans all PDF PageIndex trees (doc descriptions + section summaries)
2. Identifies the most relevant sections across documents
3. Returns section IDs + summaries for downstream graph queries

Key Design Principle — No Vector DB:
- Section selection is done via LLM reasoning over tree structure, NOT vector similarity
- The PageIndex tree is stored as JSON, not in any vector database
- Retrieval is grounded in explicit page/section references (traceable & explainable)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rags.graph_rag.llm_config import get_construction_client, get_construction_models

logger = logging.getLogger(__name__)


class PageReasoner:
    """LLM-based reasoner that selects relevant document sections for a query.

    Usage:
        reasoner = PageReasoner()

        # Select relevant sections across ALL PDF trees
        selections = reasoner.select_sections(
            question="What is Hemant's salary?",
            trees={"hemant_story": tree1, "hemant_conversations": tree2},
        )

        # Get the full context for the selected sections
        context = reasoner.gather_context(selections, trees)
    """

    def __init__(self):
        """Initialize the Page Reasoner with LLM client from existing config."""
        self.client = get_construction_client()
        self.model_fallbacks = get_construction_models()
        logger.info(f"[PageReasoner] Provider: {self.client.base_url}")
        logger.info(f"[PageReasoner] Models: {self.model_fallbacks}")

    # ==============================================================
    # Main API
    # ==============================================================

    def select_sections(
        self,
        question: str,
        trees: Dict[str, Dict[str, Any]],
        max_sections: int = 5,
        max_documents: int = 3,
    ) -> List[Dict[str, Any]]:
        """Select the most relevant sections across all PDF trees for a question.

        Uses a 2-stage approach:
        1. Document Selection: Which PDFs are relevant? (based on doc_description)
        2. Section Selection: Which sections within those PDFs? (based on summaries)

        Args:
            question: User's natural language question.
            trees: Dict mapping PDF stem -> PageIndex tree.
            max_sections: Max sections to return total.
            max_documents: Max documents to consider.

        Returns:
            List of selected section dicts with:
                - document: PDF filename
                - section_id: Node ID in tree
                - title: Section title
                - start_index: Start page
                - end_index: End page
                - summary: Section summary
                - relevance_score: int (1-10)
                - reason: Why this section was selected
        """
        if not trees:
            logger.warning("[PageReasoner] No trees provided for selection")
            return []

        print(f"\n{'='*60}")
        print(f"  Page Reasoner — Selecting sections for: {question}")
        print(f"{'='*60}")

        # --- Stage 1: Document Selection ---
        selected_docs = self._select_documents(question, trees, max_documents)

        if not selected_docs:
            print("  No relevant documents found.")
            return []

        print(f"  Selected {len(selected_docs)} document(s):")
        for doc_name, score in selected_docs:
            print(f"    - {doc_name} (relevance: {score}/10)")

        # --- Stage 2: Section Selection ---
        selected_sections = self._select_sections_from_docs(
            question, selected_docs, trees, max_sections
        )

        print(f"\n  Selected {len(selected_sections)} section(s):")
        for sec in selected_sections:
            pages = f"pp. {sec['start_index']}-{sec['end_index']}"
            print(f"    [{sec['document']}] {sec['title']} {pages} (score: {sec['relevance_score']}/10)")

        return selected_sections

    def gather_context(
        self,
        selected_sections: List[Dict[str, Any]],
        trees: Dict[str, Dict[str, Any]],
        include_summaries: bool = True,
    ) -> str:
        """Build a rich context string from selected sections.

        This context can be fed directly into the answer generation LLM.

        Args:
            selected_sections: Output from select_sections().
            trees: Dict mapping PDF stem -> PageIndex tree (same as select_sections).
            include_summaries: Whether to include section summaries.

        Returns:
            Formatted context string with document + section info.
        """
        if not selected_sections:
            return ""

        lines = [
            "=" * 60,
            "RELEVANT DOCUMENT SECTIONS",
            "=" * 60,
            "",
        ]

        for i, sec in enumerate(selected_sections, 1):
            doc_name = sec.get("document", "unknown")
            title = sec.get("title", "unknown")
            node_id = sec.get("section_id", "?")
            pages = f"Pages {sec.get('start_index')}-{sec.get('end_index')}"
            summary = sec.get("summary", "")
            reason = sec.get("reason", "")

            lines.append(f"[Section {i}]")
            lines.append(f"  Document: {doc_name}")
            lines.append(f"  Section:  {title} (ID: {node_id})")
            lines.append(f"  Pages:    {pages}")
            if include_summaries and summary:
                lines.append(f"  Summary:  {summary}")
            if reason:
                lines.append(f"  Reason:   {reason}")
            lines.append("")

            # Include the actual text content from the tree if available
            tree = trees.get(doc_name.replace(".pdf", ""))
            if tree:
                section_node = self._find_node(tree.get("structure", []), node_id)
                if section_node and section_node.get("text"):
                    text = section_node["text"]
                    # Truncate very long text to avoid token overflow
                    if len(text) > 3000:
                        text = text[:3000] + "...[truncated]"
                    lines.append(f"  Content:")
                    lines.append(f"  ```")
                    lines.append(text)
                    lines.append(f"  ```")
                    lines.append("")

        return "\n".join(lines)

    # ==============================================================
    # Stage 1: Document Selection
    # ==============================================================

    def _select_documents(
        self,
        question: str,
        trees: Dict[str, Dict[str, Any]],
        max_documents: int,
    ) -> List[Tuple[str, int]]:
        """Select relevant documents using doc_description + LLM reasoning.

        Returns list of (doc_name, relevance_score) tuples.
        """
        # Build document catalog
        doc_catalog = []
        for stem, tree in trees.items():
            desc = tree.get("doc_description", "")
            doc_name = tree.get("doc_name", f"{stem}.pdf")
            doc_catalog.append({
                "stem": stem,
                "name": doc_name,
                "description": desc or "No description available.",
            })

        if not doc_catalog:
            return []

        # If only 1 document, auto-select it
        if len(doc_catalog) == 1:
            return [(doc_catalog[0]["name"], 10)]

        # Build prompt for document selection
        catalog_text = "\n".join(
            f"[{i+1}] {d['name']}: {d['description']}"
            for i, d in enumerate(doc_catalog)
        )

        prompt = f"""You are a document retrieval expert. Your task is to select which documents are most relevant to answering the user's question.

QUESTION:
{question}

AVAILABLE DOCUMENTS:
{catalog_text}

TASK:
For each document, rate its relevance to the question on a scale of 1-10.
Consider the document description and what information it likely contains.

OUTPUT AS VALID JSON ONLY:
{{
    "documents": [
        {{
            "index": 1,
            "name": "document_name.pdf",
            "relevance_score": <int 1-10>,
            "reason": "Brief reason why this document is relevant or not"
        }},
        ...
    ]
}}

Return ONLY the JSON, no other text."""

        response = self._llm_call(prompt)
        if not response:
            # Fallback: return all documents with neutral score
            return [(d["name"], 5) for d in doc_catalog]

        try:
            result = json.loads(response)
            docs = result.get("documents", [])
            # Sort by relevance and take top max_documents
            docs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            selected = [
                (d["name"], d["relevance_score"])
                for d in docs
                if d.get("relevance_score", 0) >= 5
            ][:max_documents]
            return selected or [(doc_catalog[0]["name"], 5)]
        except (json.JSONDecodeError, KeyError):
            return [(d["name"], 5) for d in doc_catalog[:max_documents]]

    # ==============================================================
    # Stage 2: Section Selection
    # ==============================================================

    def _select_sections_from_docs(
        self,
        question: str,
        selected_docs: List[Tuple[str, int]],
        trees: Dict[str, Dict[str, Any]],
        max_sections: int,
    ) -> List[Dict[str, Any]]:
        """Select relevant sections from the chosen documents."""
        # Build section catalog for selected documents
        section_catalog = []
        for doc_name, doc_score in selected_docs:
            # Find the tree for this document
            stem = doc_name.replace(".pdf", "")
            tree = trees.get(stem)
            if not tree:
                # Try matching by checking doc_name
                for s, t in trees.items():
                    if t.get("doc_name") == doc_name:
                        tree = t
                        stem = s
                        break
            if not tree:
                continue

            # Collect all leaf sections (most granular)
            all_sections = self._flatten_sections(tree.get("structure", []))
            for sec in all_sections:
                section_catalog.append({
                    "document": doc_name,
                    "document_stem": stem,
                    "section_id": sec.get("node_id", "?"),
                    "title": sec.get("title", "Unknown"),
                    "start_index": sec.get("start_index", 0),
                    "end_index": sec.get("end_index", 0),
                    "summary": sec.get("summary", ""),
                    "parent_titles": self._get_parent_titles(tree.get("structure", []), sec.get("node_id", "")),
                })

        if not section_catalog:
            return []

        # Build prompt for section selection
        catalog_text = "\n---\n".join(
            f"[{i+1}] Doc: {s['document']}\n"
            f"    Section: {s['title']} (ID: {s['section_id']})\n"
            f"    Pages: {s['start_index']}-{s['end_index']}\n"
            f"    Path: {' > '.join(s['parent_titles'] + [s['title']])}\n"
            f"    Summary: {s['summary'][:300] if s['summary'] else 'N/A'}"
            for i, s in enumerate(section_catalog)
        )

        prompt = f"""You are a section retrieval expert. Your task is to select the most relevant sections from documents that can answer the user's question.

QUESTION:
{question}

AVAILABLE SECTIONS (across all documents):
{catalog_text}

TASK:
Select up to {max_sections} sections that are most likely to contain the answer.
Consider:
1. The section title and its path in the document hierarchy
2. The section summary (what it covers)
3. Which pages it spans

OUTPUT AS VALID JSON ONLY:
{{
    "sections": [
        {{
            "index": <int>,
            "relevance_score": <int 1-10>,
            "reason": "Why this section is relevant"
        }},
        ...
    ]
}}

Return ONLY the JSON, no other text."""

        response = self._llm_call(prompt)
        if not response:
            # Fallback: return first few sections
            return self._section_dicts(section_catalog[:max_sections], section_catalog)

        try:
            result = json.loads(response)
            selected_indices = result.get("sections", [])
            selected_indices.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            selected = []
            for sel in selected_indices[:max_sections]:
                idx = sel.get("index", 0) - 1  # Convert to 0-indexed
                if 0 <= idx < len(section_catalog):
                    sec = section_catalog[idx]
                    selected.append({
                        "document": sec["document"],
                        "section_id": sec["section_id"],
                        "title": sec["title"],
                        "start_index": sec["start_index"],
                        "end_index": sec["end_index"],
                        "summary": sec["summary"],
                        "relevance_score": sel.get("relevance_score", 5),
                        "reason": sel.get("reason", ""),
                    })

            return selected if selected else self._section_dicts(section_catalog[:max_sections], section_catalog)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[PageReasoner] Section selection parsing failed: {e}")
            return self._section_dicts(section_catalog[:max_sections], section_catalog)

    # ==============================================================
    # Helpers
    # ==============================================================

    def _llm_call(self, prompt: str) -> Optional[str]:
        """Call the LLM with fallback models."""
        last_err = None
        for model in self.model_fallbacks:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                if content:
                    # Extract JSON from potential markdown wrapping
                    content = content.strip()
                    if content.startswith("```"):
                        content = content.split("```", 1)[1]
                        if "```" in content:
                            content = content.rsplit("```", 1)[0]
                        # Remove 'json' label if present
                        if content.startswith("json"):
                            content = content[4:]
                    return content.strip()
            except Exception as e:
                last_err = e
                continue

        if last_err:
            logger.warning(f"[PageReasoner] All LLM calls failed: {last_err}")
        return None

    @staticmethod
    def _flatten_sections(structure: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten tree into list of all sections."""
        sections = []
        for node in structure:
            sections.append(node)
            if node.get("nodes"):
                sections.extend(PageReasoner._flatten_sections(node["nodes"]))
        return sections

    @staticmethod
    def _find_node(structure: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
        """Find a node by its node_id in the tree."""
        for node in structure:
            if node.get("node_id") == node_id:
                return node
            if node.get("nodes"):
                found = PageReasoner._find_node(node["nodes"], node_id)
                if found:
                    return found
        return None

    @staticmethod
    def _get_parent_titles(structure: List[Dict[str, Any]], target_id: str) -> List[str]:
        """Get the chain of parent titles leading to a node."""
        def _search(nodes, path):
            for node in nodes:
                if node.get("node_id") == target_id:
                    return path
                if node.get("nodes"):
                    result = _search(node["nodes"], path + [node.get("title", "")])
                    if result is not None:
                        return result
            return None

        result = _search(structure, [])
        return result or []

    @staticmethod
    def _section_dicts(
        catalog: List[Dict[str, Any]], full_catalog: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert catalog entries to output format (fallback)."""
        selected = []
        for sec in catalog:
            selected.append({
                "document": sec["document"],
                "section_id": sec["section_id"],
                "title": sec["title"],
                "start_index": sec["start_index"],
                "end_index": sec["end_index"],
                "summary": sec["summary"],
                "relevance_score": 5,
                "reason": "Fallback selection",
            })
        return selected


# ==============================================================
# Quick test
# ==============================================================
if __name__ == "__main__":
    from rags.pageindex.page_index_builder import PageIndexBuilder

    # Build or load trees
    builder = PageIndexBuilder()
    trees = {}

    test_pdfs = [
        Path("data/hemant_story.pdf"),
        Path("data/hemant_conversations (1).pdf"),
    ]

    for pdf in test_pdfs:
        if pdf.is_file():
            print(f"Loading/ Building tree for: {pdf.name}")
            tree = builder.build(pdf, force_rebuild=False)
            trees[pdf.stem] = tree

    if not trees:
        print("No PDFs found to test with.")
    else:
        reasoner = PageReasoner()

        # Test questions
        questions = [
            "What is Hemant Sharma's employee ID?",
            "What is Hemant's monthly salary?",
            "Who is Hemant's reporting manager?",
        ]

        for q in questions:
            print(f"\n\n{'='*70}")
            print(f"QUESTION: {q}")
            print(f"{'='*70}")

            selections = reasoner.select_sections(q, trees, max_sections=3)
            context = reasoner.gather_context(selections, trees)

            print(f"\nContext for answer generation:")
            print(context[:1000])
            print("...")

