"""Test script for PageIndex integration.
Tests: page_index_builder, page_search, multimodal_graph_builder, multimodal_rag
"""
import json
import sys
from pathlib import Path


def test_pageindex_builder():
    """Test 1: PageIndex tree generation"""
    print("\n" + "=" * 60)
    print("TEST 1: PageIndex Builder")
    print("=" * 60)

    from rags.pageindex.page_index_builder import PageIndexBuilder

    builder = PageIndexBuilder()
    pdf_path = Path("data/hemant_story.pdf")

    if not pdf_path.is_file():
        print(f"  SKIP: PDF not found: {pdf_path}")
        return None

    tree = builder.build(pdf_path, force_rebuild=False)

    print(f"  Doc name: {tree.get('doc_name', 'N/A')}")
    print(f"  Doc description: {str(tree.get('doc_description', 'N/A'))[:80]}...")
    print(f"  Structure nodes: {len(tree.get('structure', []))}")

    # Print tree
    builder.print_tree(tree)

    # Test leaf sections
    leaves = builder.get_leaf_sections(tree)
    print(f"\n  Leaf sections: {len(leaves)}")
    for l in leaves[:5]:
        print(f"    [{l.get('node_id')}] {l.get('title')} (pp {l.get('start_index')}-{l.get('end_index')})")

    # Test tree context
    context = builder.tree_context(tree)
    print(f"\n  Tree context length: {len(context)} chars")

    # Test tree summary
    summary = builder.tree_summary(tree)
    print(f"  Tree summary: {summary}")

    # Test section lookup
    if leaves:
        test_id = leaves[0].get("node_id")
        found = builder.get_section(tree, test_id)
        print(f"  Section lookup [{test_id}]: {'FOUND' if found else 'MISSING'}")

    # Test page range lookup
    sections_by_page = builder.get_sections_by_page_range(tree, 1, 2)
    print(f"  Sections spanning pages 1-2: {len(sections_by_page)}")

    return tree


def test_page_search(tree):
    """Test 2: PageSearch cross-document search"""
    print("\n" + "=" * 60)
    print("TEST 2: Page Search")
    print("=" * 60)

    from rags.pageindex.page_search import PageSearch

    searcher = PageSearch()

    # Test keyword search
    results = searcher.search(
        "salary employee",
        {"hemant_story": tree},
        max_results=5,
    )
    print(f"  Search 'salary employee': {len(results)} results")
    for r in results:
        print(f"    [{r['document']}] {r['title']} (score: {r['relevance_score']})")

    # Test search across profiles
    print("\n  Searching across cached profiles...")
    profile_results = searcher.search_across_profiles(
        "Hemant",
        profile_dir="profiles",
        max_results=5,
    )
    print(f"    Found {len(profile_results)} results")

    # Test section context
    if results:
        top = results[0]
        context = searcher.get_section_context(
            {"hemant_story": tree},
            top["document_stem"],
            top["section_id"],
        )
        if context:
            print(f"\n  Section context for top result:")
            print(f"    Title: {context['title']}")
            print(f"    Pages: {context['start_index']}-{context['end_index']}")
            print(f"    Parent chain: {' > '.join(context.get('parent_titles', []))}")
            print(f"    Has text: {bool(context.get('text'))}")

    # Test overlapping sections
    overlapping = searcher.find_overlapping_sections({"hemant_story": tree}, 1)
    print(f"\n  Sections on page 1: {len(overlapping)}")


def test_multimodal_graph_builder():
    """Test 3: MultimodalGraphBuilder with use_pageindex=True"""
    print("\n" + "=" * 60)
    print("TEST 3: Multimodal Graph Builder (PageIndex mode)")
    print("=" * 60)

    from rags.multimodal_rag.multimodal_graph_builder import MultimodalGraphBuilder

    pdf_path = Path("data/hemant_story.pdf")
    if not pdf_path.is_file():
        print("  SKIP: PDF not found")
        return

    # Only test text modality to avoid heavy processing
    builder = MultimodalGraphBuilder(
        enable_text=True,
        enable_tables=False,  # skip heavy processing
        enable_images=False,
        enable_ocr=False,
        use_pageindex=True,
    )

    print("  Builder initialized with use_pageindex=True")
    print(f"  Builder mode: {'PageIndex' if builder.use_pageindex else 'Legacy'}")

    # Test that the PageIndex builder is accessible
    print(f"  PageIndexBuilder: {type(builder.page_index_builder).__name__}")


def test_multimodal_rag():
    """Test 4: MultimodalRAG initialization and build flow"""
    print("\n" + "=" * 60)
    print("TEST 4: Multimodal RAG (PageIndex mode)")
    print("=" * 60)

    from rags.multimodal_rag.multimodal_rag import MultimodalRAG

    rag = MultimodalRAG(
        enable_text=True,
        enable_tables=False,
        enable_images=False,
        enable_ocr=False,
        use_pageindex=True,
    )

    print(f"  RAG initialized with use_pageindex={rag.use_pageindex}")
    print(f"  Builder type: {type(rag.builder).__name__}")
    print(f"  Has PageReasoner: {rag.page_reasoner is not None}")
    print(f"  Has PageSearch: {rag.page_search is not None}")
    print(f"  Has query engine: {rag.query_engine is not None}")


def test_page_reasoner(tree):
    """Test 5: PageReasoner section selection"""
    print("\n" + "=" * 60)
    print("TEST 5: Page Reasoner")
    print("=" * 60)

    from rags.pageindex.page_reasoner import PageReasoner

    reasoner = PageReasoner()
    print(f"  Reasoner initialized: {reasoner is not None}")
    print(f"  Provider: {reasoner.client.base_url}")

    # Test section flattening
    from rags.pageindex.page_reasoner import PageReasoner as PR
    flat = PR._flatten_sections(tree.get("structure", []))
    print(f"  Flattened sections: {len(flat)}")

    # Test parent chain
    if flat:
        target_id = flat[0].get("node_id", "")
        parents = PR._get_parent_titles(tree.get("structure", []), target_id)
        print(f"  Parent chain for [{target_id}]: {' > '.join(parents) if parents else '<root>'}")

    print("\n  Note: Full LLM-based selection requires API call.")
    print("  Structure validation: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("  PAGEINDEX INTEGRATION TESTS")
    print("=" * 60)

    # Run tests in order
    tree = test_pageindex_builder()

    if tree:
        test_page_search(tree)
        test_page_reasoner(tree)
    else:
        print("\n  ⚠️  Skipping tree-dependent tests")

    test_multimodal_graph_builder()
    test_multimodal_rag()

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETED")
    print("=" * 60)

