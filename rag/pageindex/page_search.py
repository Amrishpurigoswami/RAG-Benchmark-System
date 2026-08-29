"""Page Search — Cross-document section search and context gathering.

Provides efficient section-level search across multiple PDF PageIndex trees
without using any vector database. Uses keyword matching + LLM-based scoring.

Supports:
- Multi-document section search
- Cross-document section matching by title/summary
- Section-based context gathering
- Document-level filtering before section-level search
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PageSearch:
    """Cross-document section search using PageIndex trees.

    All searching is done via keyword matching + LLM reasoning over the tree
    structure — no vector DB, no embedding model, no similarity search.

    Usage:
        searcher = PageSearch()
        results = searcher.search("salary", trees)
        # Or across all cached trees:
        results = searcher.search_across_profiles("Hemant", profile_dir="profiles/")
    """

    def __init__(self):
        self._tree_cache: Dict[str, Dict[str, Any]] = {}

    # ==============================================================
    # Main Search API
    # ==============================================================

    def search(
        self,
        query: str,
        trees: Dict[str, Dict[str, Any]],
        max_results: int = 10,
        search_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple PageIndex trees by keyword.

        Matches against section titles, summaries, and text content.
        No vector DB — pure text matching + scoring.

        Args:
            query: Search query string.
            trees: Dict mapping PDF stem -> PageIndex tree.
            max_results: Max sections to return.
            search_fields: Fields to search in (default: title, summary, text).

        Returns:
            List of matching sections with relevance scores.
        """
        if not trees:
            return []

        search_fields = search_fields or ["title", "summary", "text"]
        query_lower = query.lower()
        query_terms = query_lower.split()
        results = []

        for stem, tree in trees.items():
            doc_name = tree.get("doc_name", f"{stem}.pdf")
            all_sections = self._flatten_sections(tree.get("structure", []))

            for section in all_sections:
                score = self._score_section(section, query_lower, query_terms, search_fields)
                if score > 0:
                    results.append({
                        "document": doc_name,
                        "document_stem": stem,
                        "section_id": section.get("node_id", ""),
                        "title": section.get("title", ""),
                        "start_index": section.get("start_index", 0),
                        "end_index": section.get("end_index", 0),
                        "summary": section.get("summary", ""),
                        "text_preview": (section.get("text", "") or "")[:200],
                        "relevance_score": score,
                        "parent_titles": self._get_parent_titles(tree.get("structure", []), section.get("node_id", "")),
                    })

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]

    def search_across_profiles(
        self,
        query: str,
        profile_dir: str | Path = "profiles",
        max_results: int = 10,
        pattern: str = "*_pageindex.json",
    ) -> List[Dict[str, Any]]:
        """Search across all cached PageIndex files in the profiles directory.

        Automatically loads all PageIndex JSON files.

        Args:
            query: Search query string.
            profile_dir: Directory containing PageIndex JSON files.
            max_results: Max sections to return.
            pattern: Glob pattern for matching PageIndex files.

        Returns:
            List of matching sections.
        """
        profile_dir = Path(profile_dir)
        if not profile_dir.is_dir():
            logger.warning(f"[PageSearch] Profile directory not found: {profile_dir}")
            return []

        trees = {}
        for json_path in sorted(profile_dir.glob(pattern)):
            try:
                tree = json.loads(json_path.read_text(encoding="utf-8"))
                stem = json_path.stem.replace("_pageindex", "")
                trees[stem] = tree
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[PageSearch] Failed to load {json_path.name}: {e}")

        return self.search(query, trees, max_results)

    # ==============================================================
    # Section Lookup
    # ==============================================================

    def get_section_context(
        self,
        trees: Dict[str, Dict[str, Any]],
        document_stem: str,
        section_id: str,
        include_text: bool = True,
        include_children: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Get full context for a specific section across documents.

        Args:
            trees: Dict mapping PDF stem -> PageIndex tree.
            document_stem: PDF stem (e.g., "hemant_story").
            section_id: Node ID (e.g., "0003").
            include_text: Include the full text content.
            include_children: Include child section summaries.

        Returns:
            Section context dict, or None if not found.
        """
        tree = trees.get(document_stem)
        if not tree:
            return None

        section = self._find_node(tree.get("structure", []), section_id)
        if not section:
            return None

        context = {
            "document": tree.get("doc_name", f"{document_stem}.pdf"),
            "title": section.get("title", ""),
            "section_id": section_id,
            "start_index": section.get("start_index", 0),
            "end_index": section.get("end_index", 0),
            "summary": section.get("summary", ""),
            "parent_titles": self._get_parent_titles(tree.get("structure", []), section_id),
            "page_count": (section.get("end_index", 0) - section.get("start_index", 0) + 1),
        }

        if include_text and section.get("text"):
            text = section["text"]
            # Truncate very long text
            if len(text) > 5000:
                text = text[:5000] + "...[truncated]"
            context["text"] = text

        if include_children and section.get("nodes"):
            context["child_sections"] = [
                {
                    "title": child.get("title", ""),
                    "section_id": child.get("node_id", ""),
                    "pages": f"{child.get('start_index')}-{child.get('end_index')}",
                    "summary": child.get("summary", ""),
                }
                for child in section["nodes"]
            ]

        return context

    def get_sections_by_keywords(
        self,
        trees: Dict[str, Dict[str, Any]],
        keywords: List[str],
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find sections that contain all specified keywords.

        Useful for narrowing down sections before answering specific questions.

        Args:
            trees: Dict mapping PDF stem -> PageIndex tree.
            keywords: List of required keywords (all must match).
            max_results: Max sections to return.

        Returns:
            List of matching sections.
        """
        combined_query = " ".join(keywords)
        return self.search(combined_query, trees, max_results)

    def find_overlapping_sections(
        self,
        trees: Dict[str, Dict[str, Any]],
        page: int,
    ) -> List[Dict[str, Any]]:
        """Find all sections across documents that contain a specific page.

        Args:
            trees: Dict mapping PDF stem -> PageIndex tree.
            page: Page number (1-indexed).

        Returns:
            List of sections spanning this page.
        """
        results = []
        for stem, tree in trees.items():
            doc_name = tree.get("doc_name", f"{stem}.pdf")
            for section in self._flatten_sections(tree.get("structure", [])):
                s = section.get("start_index", 0)
                e = section.get("end_index", 0)
                if s <= page <= e:
                    results.append({
                        "document": doc_name,
                        "document_stem": stem,
                        "section_id": section.get("node_id", ""),
                        "title": section.get("title", ""),
                        "start_index": s,
                        "end_index": e,
                        "summary": section.get("summary", ""),
                    })
        return results

    # ==============================================================
    # Internal Helpers
    # ==============================================================

    def _score_section(
        self,
        section: Dict[str, Any],
        query_lower: str,
        query_terms: List[str],
        search_fields: List[str],
    ) -> float:
        """Score a section's relevance to a query. Returns 0 if no match."""
        score = 0.0

        for field in search_fields:
            text = (section.get(field) or "").lower()
            if not text:
                continue

            # Exact phrase match (highest weight)
            if query_lower in text:
                if field == "title":
                    score += 10.0
                elif field == "summary":
                    score += 5.0
                else:
                    score += 3.0

            # Individual term matches
            for term in query_terms:
                if term and len(term) > 2 and term in text:
                    if field == "title":
                        score += 3.0
                    elif field == "summary":
                        score += 2.0
                    else:
                        score += 1.0

        return score

    @staticmethod
    def _flatten_sections(structure: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten tree into list of all sections."""
        sections = []
        for node in structure:
            sections.append(node)
            if node.get("nodes"):
                sections.extend(PageSearch._flatten_sections(node["nodes"]))
        return sections

    @staticmethod
    def _find_node(structure: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
        """Find a node by its node_id in the tree."""
        for node in structure:
            if node.get("node_id") == node_id:
                return node
            if node.get("nodes"):
                found = PageSearch._find_node(node["nodes"], node_id)
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


# ==============================================================
# Quick test
# ==============================================================
if __name__ == "__main__":
    from pathlib import Path

    searcher = PageSearch()

    # Search across cached PageIndex files
    results = searcher.search_across_profiles(
        "Hemant salary employee",
        profile_dir="profiles",
        max_results=5,
    )

    if results:
        print(f"\nFound {len(results)} relevant sections:")
        for r in results:
            pages = f"pp. {r['start_index']}-{r['end_index']}"
            print(f"  [{r['document']}] {r['title']} {pages} (score: {r['relevance_score']})")
            if r.get("summary"):
                print(f"    Summary: {r['summary'][:150]}...")

            # Get full context for the top result
            if results.index(r) == 0:
                context = searcher.get_section_context(
                    {r["document_stem"]: {"doc_name": r["document"], "structure": []}},
                    r["document_stem"],
                    r["section_id"],
                )
                if context:
                    print(f"\nFull context for top result:")
                    print(f"  Document: {context['document']}")
                    print(f"  Section: {context['title']} ({context['section_id']})")
                    print(f"  Pages: {context['start_index']}-{context['end_index']}")
                    print(f"  Summary: {context.get('summary', 'N/A')[:200]}")
    else:
        print("No PageIndex files found in profiles/. Build some first with PageIndexBuilder.")

