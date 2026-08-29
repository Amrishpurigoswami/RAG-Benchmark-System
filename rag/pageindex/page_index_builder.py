"""PageIndex Builder — Generate hierarchical document tree from PDFs.

Wraps the PageIndex library to produce a semantic tree structure
(sections with titles, summaries, page ranges) for each PDF.

Output saved to: profiles/<pdf_name>_pageindex.json

Uses litellm + existing OPENROUTER_API_KEY for LLM reasoning.
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PageIndexBuilder:
    """Build and manage PageIndex trees for PDF documents.

    Usage:
        builder = PageIndexBuilder()
        tree = builder.build("data/hemant_story.pdf")
        tree = builder.load("hemant_story")  # Load cached tree

    Tree Format:
    {
        "doc_name": "hemant_story.pdf",
        "doc_description": "...",
        "structure": [
            {
                "title": "Section Title",
                "node_id": "0001",
                "start_index": 1,
                "end_index": 5,
                "summary": "...",
                "nodes": [...]
            }
        ]
    }
    """

    def __init__(self, model: Optional[str] = None):
        """
        Args:
            model: LiteLLM model string (e.g., "openrouter/meta-llama/llama-3.3-70b-instruct:free").
                   If None, uses env var LITELLM_MODEL or OPENROUTER_MODEL, or default.
        """
        self._model = model or os.getenv(
            "LITELLM_MODEL",
            os.getenv(
                "OPENROUTER_MODEL",
                "openrouter/meta-llama/llama-3.3-70b-instruct:free",
            ),
        )
        self._output_dir = Path("profiles")
        self._output_dir.mkdir(exist_ok=True)

        # Cache loaded page indices
        self._trees: Dict[str, Dict[str, Any]] = {}

    # ==============================================================
    # Build PageIndex for a PDF
    # ==============================================================

    def build(
        self,
        pdf_path: str | Path,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """Generate a PageIndex tree for a PDF document.

        Uses the PageIndex library to produce a hierarchical tree structure
        with logical sections, summaries, and page ranges.

        Args:
            pdf_path: Path to the PDF file.
            force_rebuild: If True, rebuild even if a cached tree exists.

        Returns:
            PageIndex tree dict with doc_name, doc_description, and structure.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        stem = pdf_path.stem
        cache_path = self._output_dir / f"{stem}_pageindex.json"

        # Use cached version if available
        if cache_path.exists() and not force_rebuild:
            logger.info(f"[PageIndexBuilder] Loading cached tree for {stem}")
            tree = json.loads(cache_path.read_text(encoding="utf-8"))
            self._trees[stem] = tree
            return tree

        print(f"\n{'='*60}")
        print(f"  PageIndex Builder — Building tree for: {stem}")
        print(f"{'='*60}")

        # --- Run PageIndex tree generation ---
        tree = self._build_tree(pdf_path)

        # --- Save to cache ---
        cache_path.write_text(
            json.dumps(tree, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._trees[stem] = tree

        print(f"  Tree saved to: {cache_path}")
        print(f"  Sections found: {len(self._get_all_sections(tree))}")
        print(f"{'='*60}")

        return tree

    def _build_tree(self, pdf_path: Path) -> Dict[str, Any]:
        """Internal: actually call the PageIndex library to build the tree.

        Uses litellm with the configured model to drive the PageIndex
        tree-building pipeline (TOC extraction, section detection, etc.).

        The PageIndex library already has all the logic:
        - page index detection
        - section boundary detection
        - summary generation per node
        - document description generation
        """
        try:
            # Import PageIndex's main function
            # The pageindex_lib is a custom copy of the official PageIndex repo
            from rags.pageindex_lib.page_index import page_index_main
            from rags.pageindex_lib.utils import ConfigLoader
        except ImportError as e:
            raise ImportError(
                f"PageIndex library not available: {e}. "
                "Make sure pageindex_lib/ is properly installed."
            )

        # --- Configure PageIndex options ---
        # Use litellm model format compatible with OpenRouter
        config = {
            "model": self._model,
            "toc_check_page_num": 20,
            "max_page_num_each_node": 10,
            "max_token_num_each_node": 20000,
            "if_add_node_id": "yes",
            "if_add_node_summary": "yes",
            "if_add_doc_description": "yes",
            "if_add_node_text": "yes",
        }

        opt = ConfigLoader().load(config)

        print(f"  Running PageIndex tree generation with model: {self._model}")
        print(f"  This may take a while for large documents...")

        # --- Run the PageIndex pipeline ---
        # page_index_main returns:
        # {
        #   "doc_name": "...",
        #   "doc_description": "...",
        #   "structure": [ ... tree nodes ... ]
        # }
        tree = page_index_main(str(pdf_path), opt)

        print(f"  PageIndex generation complete.")
        print(f"  Doc description: {tree.get('doc_description', 'N/A')[:100]}...")

        return tree

    # ==============================================================
    # Load cached PageIndex
    # ==============================================================

    def load(self, pdf_stem: str) -> Dict[str, Any]:
        """Load a previously built PageIndex tree from cache.

        Args:
            pdf_stem: PDF filename stem (e.g., "hemant_story" from "hemant_story.pdf").

        Returns:
            PageIndex tree dict.
        """
        if pdf_stem in self._trees:
            return self._trees[pdf_stem]

        cache_path = self._output_dir / f"{pdf_stem}_pageindex.json"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cached PageIndex found for '{pdf_stem}'. "
                f"Run build('{pdf_stem}.pdf') first."
            )

        tree = json.loads(cache_path.read_text(encoding="utf-8"))
        self._trees[pdf_stem] = tree
        return tree

    def load_all(self, pdf_stems: List[str]) -> Dict[str, Dict[str, Any]]:
        """Load multiple PageIndex trees by their stems.

        Args:
            pdf_stems: List of PDF filename stems.

        Returns:
            Dict mapping stem -> tree.
        """
        result = {}
        for stem in pdf_stems:
            result[stem] = self.load(stem)
        return result

    # ==============================================================
    # Query / Traverse
    # ==============================================================

    def get_section(self, tree: Dict[str, Any], section_id: str) -> Optional[Dict[str, Any]]:
        """Find a section by node_id in the tree.

        Args:
            tree: PageIndex tree dict.
            section_id: Node ID (e.g., "0001").

        Returns:
            Section dict, or None if not found.
        """
        for node in self._iter_nodes(tree.get("structure", [])):
            if node.get("node_id") == section_id:
                return node
        return None

    def get_leaf_sections(self, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all leaf-level sections (sections without children).

        These are the most granular processing units.

        Args:
            tree: PageIndex tree dict.

        Returns:
            List of leaf section dicts.
        """
        leaves = []
        self._collect_leaves(tree.get("structure", []), leaves)
        return leaves

    def get_section_by_title(self, tree: Dict[str, Any], title: str) -> Optional[Dict[str, Any]]:
        """Find a section by its title (case-insensitive partial match).

        Args:
            tree: PageIndex tree dict.
            title: Section title to search for.

        Returns:
            First matching section dict, or None.
        """
        title_lower = title.lower()
        for node in self._iter_nodes(tree.get("structure", [])):
            node_title = (node.get("title") or "").lower()
            if title_lower in node_title:
                return node
        return None

    def get_sections_by_page_range(
        self, tree: Dict[str, Any], start_page: int, end_page: int
    ) -> List[Dict[str, Any]]:
        """Get sections that fall within a page range.

        Args:
            tree: PageIndex tree dict.
            start_page: Start page (inclusive).
            end_page: End page (inclusive).

        Returns:
            List of matching section dicts.
        """
        matches = []
        for node in self._iter_nodes(tree.get("structure", [])):
            s = node.get("start_index")
            e = node.get("end_index")
            if s is not None and e is not None:
                if s <= end_page and e >= start_page:
                    matches.append(node)
        return matches

    def get_all_sections(self, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all sections (flat list) from a tree.

        Args:
            tree: PageIndex tree dict.

        Returns:
            List of all section dicts.
        """
        return self._get_all_sections(tree)

    # ==============================================================
    # Summary / Stats
    # ==============================================================

    def print_tree(self, tree: Dict[str, Any], indent: int = 0) -> None:
        """Print a readable tree structure to console.

        Args:
            tree: PageIndex tree dict.
            indent: Indentation level.
        """
        self._print_nodes(tree.get("structure", []), indent)

    def tree_summary(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of the tree structure.

        Args:
            tree: PageIndex tree dict.

        Returns:
            Summary dict with counts and metadata.
        """
        all_sections = self._get_all_sections(tree)
        leaf_sections = self.get_leaf_sections(tree)

        return {
            "doc_name": tree.get("doc_name", ""),
            "doc_description": tree.get("doc_description", ""),
            "total_sections": len(all_sections),
            "leaf_sections": len(leaf_sections),
            "depth": self._tree_depth(tree.get("structure", [])),
            "page_range": self._page_range(tree.get("structure", [])),
            "top_level_sections": len(tree.get("structure", [])),
        }

    def tree_context(self, tree: Dict[str, Any]) -> str:
        """Generate a text summary of the tree suitable for LLM context.

        This is designed for the Page Reasoner to efficiently scan
        across documents and select relevant sections.

        Args:
            tree: PageIndex tree dict.

        Returns:
            Formatted string with all sections and summaries.
        """
        lines = [
            f"Document: {tree.get('doc_name', 'unknown')}",
            f"Description: {tree.get('doc_description', 'N/A')}",
            "---",
        ]
        self._build_context(tree.get("structure", []), lines, indent=0)
        return "\n".join(lines)

    # ==============================================================
    # Internal Helpers
    # ==============================================================

    @staticmethod
    def _iter_nodes(structure: List[Dict[str, Any]]):
        """Iterate over all nodes in the tree (DFS)."""
        for node in structure:
            yield node
            for child in node.get("nodes", []):
                yield from PageIndexBuilder._iter_nodes([child])

    @staticmethod
    def _collect_leaves(structure: List[Dict[str, Any]], leaves: List[Dict[str, Any]]):
        """Collect leaf nodes."""
        for node in structure:
            if not node.get("nodes"):
                leaves.append(node)
            else:
                PageIndexBuilder._collect_leaves(node.get("nodes", []), leaves)

    @staticmethod
    def _print_nodes(structure: List[Dict[str, Any]], indent: int = 0):
        """Print nodes recursively."""
        for node in structure:
            title = node.get("title", "?")
            node_id = node.get("node_id", "?")
            pages = f"pp. {node.get('start_index')}-{node.get('end_index')}" if node.get("start_index") else ""
            summary = node.get("summary", "")
            summary_short = f" — {summary[:80]}..." if summary else ""
            print(f"{'  ' * indent}[{node_id}] {title} {pages}{summary_short}")
            if node.get("nodes"):
                PageIndexBuilder._print_nodes(node.get("nodes", []), indent + 1)

    @staticmethod
    def _build_context(structure: List[Dict[str, Any]], lines: List[str], indent: int = 0):
        """Build context string recursively."""
        for node in structure:
            prefix = "  " * indent
            title = node.get("title", "?")
            node_id = node.get("node_id", "?")
            pages = f"(pp. {node.get('start_index')}-{node.get('end_index')})" if node.get("start_index") else ""
            summary = node.get("summary", "")
            if summary:
                lines.append(f"{prefix}[{node_id}] {title} {pages}: {summary[:200]}")
            else:
                lines.append(f"{prefix}[{node_id}] {title} {pages}")
            if node.get("nodes"):
                PageIndexBuilder._build_context(node.get("nodes", []), lines, indent + 1)

    @staticmethod
    def _get_all_sections(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all sections from tree as a flat list."""
        sections = []
        PageIndexBuilder._collect_sections(tree.get("structure", []), sections)
        return sections

    @staticmethod
    def _collect_sections(structure: List[Dict[str, Any]], sections: List[Dict[str, Any]]):
        """Recursively collect all sections."""
        for node in structure:
            sections.append(node)
            if node.get("nodes"):
                PageIndexBuilder._collect_sections(node.get("nodes", []), sections)

    @staticmethod
    def _tree_depth(structure: List[Dict[str, Any]]) -> int:
        """Calculate max depth of the tree."""
        if not structure:
            return 0
        max_depth = 0
        for node in structure:
            depth = 1
            if node.get("nodes"):
                depth += PageIndexBuilder._tree_depth(node.get("nodes", []))
            max_depth = max(max_depth, depth)
        return max_depth

    @staticmethod
    def _page_range(structure: List[Dict[str, Any]]) -> tuple:
        """Get overall page range from the tree."""
        min_page = float("inf")
        max_page = float("-inf")
        for node in PageIndexBuilder._iter_nodes(structure):
            s = node.get("start_index")
            e = node.get("end_index")
            if s is not None:
                min_page = min(min_page, s)
            if e is not None:
                max_page = max(max_page, e)
        if min_page == float("inf"):
            return (0, 0)
        return (int(min_page), int(max_page))


# ==============================================================
# Quick test
# ==============================================================
if __name__ == "__main__":
    import time

    builder = PageIndexBuilder()

    # Test with a known PDF
    test_pdf = Path("data/hemant_story.pdf")
    if test_pdf.is_file():
        start = time.time()
        tree = builder.build(test_pdf, force_rebuild=True)
        elapsed = time.time() - start

        print(f"\nTree built in {elapsed:.1f}s")
        print(f"Doc description: {tree.get('doc_description', 'N/A')}")
        print(f"\nTree structure:")
        builder.print_tree(tree)

        print(f"\nTree summary:")
        summary = builder.tree_summary(tree)
        for k, v in summary.items():
            print(f"  {k}: {v}")

        print(f"\nContext preview (first 500 chars):")
        context = builder.tree_context(tree)
        print(context[:500])
    else:
        print(f"No test PDF found at {test_pdf}")
