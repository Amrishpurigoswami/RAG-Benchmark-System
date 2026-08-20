import asyncio
import warnings
import config
from pathlib import Path

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality.*")
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub.*")

from ingestion.pdf_loader import PDFLoader
from rags.simple_rag.simple_rag import SimpleRAG
from rags.fusion_rag.fusion_rag import FusionRAG
from rags.self_rag.self_rag import SelfRAG
from rags.adaptive_rag.adaptive_rag import AdaptiveRAG
from rags.agentic_rag.agentic_rag import AgenticRAG

# Graph RAG imports
from rags.graph_rag.graph_builder import GraphBuilder
from rags.graph_rag.graph_rag import GraphQuery

# Multimodal RAG imports
from rags.multimodal_rag.multimodal_rag import MultimodalRAG


DATA_FOLDER = Path("data")
PDF_FILES = [str(pdf) for pdf in DATA_FOLDER.glob("*.pdf")]


def show_status(label, value, warning=False):
    prefix = "⚠️" if warning else "ℹ️"
    print(f"{prefix} {label}: {value}")


def show_menu():
    print("\n" + "=" * 60)
    print("  RAG BENCHMARK — MAIN MENU")
    print("=" * 60)
    print("  1. Simple RAG")
    print("  2. Fusion RAG")
    print("  3. Self RAG")
    print("  4. Adaptive RAG")
    print("  5. Agentic RAG")
    print("  6. Compare All (Standard RAGs)")
    print("  ────────────────────────────")
    print("  7. Graph RAG — Build Graph (Neo4j)")
    print("  8. Graph RAG — Ask Question")
    print("  ────────────────────────────")
    print("  9. Multimodal RAG (PageIndex) — Build Single PDF")
    print("  10. Multimodal RAG (Legacy Chunking) — Build Single PDF")
    print("  11. Multimodal RAG (PageIndex) — Build All PDFs")
    print("  12. Multimodal RAG (Legacy Chunking) — Build All PDFs")
    print("  ────────────────────────────")
    print("  13. Multimodal RAG — Ask (Page Reasoner)")
    print("  14. Multimodal RAG — Ask (Direct Graph Retrieval)")
    print("  15. Multimodal RAG — Ask (Multi-Doc Reasoner)")
    print("  ────────────────────────────")
    print("  0. Exit")


def show_embedding_status(label):
    from embeddings.embedding_model import EmbeddingModel

    print(f"\n📊 {label}")
    print(f"   Embedding model instances created: {EmbeddingModel.get_creation_count()}")


def build_database():
    print("\nBuilding Vector Database...\n")
    
    all_documents = []
    for pdf in PDF_FILES:
        show_status("Loading PDF", Path(pdf).name)
        loader = PDFLoader(pdf)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"   Pages loaded so far: {len(all_documents)}")
    
    print(f"\nTotal PDFs Loaded : {len(PDF_FILES)}")
    print(f"Total Pages Loaded: {len(all_documents)}")
    
    # Now chunk and store all documents together
    from ingestion.chunker import DocumentChunker
    from embeddings.embedding_model import EmbeddingModel
    from vectordb.chroma_db import ChromaDB
    
    chunker = DocumentChunker()
    embedding = EmbeddingModel()
    db = ChromaDB()
    
    show_status("Chunking", f"Splitting {len(all_documents)} pages into document chunks")
    chunks = chunker.split_documents(all_documents)
    print(f"   Chunks prepared: {len(chunks)}")
    print(f"   Embedding model instances created so far: {EmbeddingModel.get_creation_count()}")
    show_status("Storing", "Saving chunks into vector database")
    db.store_documents(chunks)
    print("Vector DB Ready!\n")

def compare_answers(
    question,
    simple_rag,
    fusion_rag,
    self_rag,
    adaptive_rag,
    agentic_rag,
    graph_rag
):

    simple_result = simple_rag.ask(
        question
    )

    fusion_result = fusion_rag.ask(
        question
    )
    self_result = self_rag.ask(
        question
    )
    adaptive_result = adaptive_rag.ask(
        question
    )
    agentic_result = agentic_rag.ask(
        question
    )
    graph_rag_result = graph_rag.ask(
        question,
    )

    print("\n")
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(question)

    print("\n")
    print("=" * 80)
    print("SIMPLE RAG")
    print("=" * 80)

    print(
        simple_result["answer"]
    )

    print("\n")
    print("=" * 80)
    print("FUSION RAG")
    print("=" * 80)

    print(
        fusion_result["answer"]
    )

    print("\n")
    print("=" * 80)
    print("SELF RAG")
    print("=" * 80)

    print(
        self_result["answer"]
    )

    print("\n")
    print("=" * 80)
    print("ADAPTIVE RAG")
    print("=" * 80)

    print(
        adaptive_result["answer"]
    )

    print("\n")
    print("=" * 80)
    print("AGENTIC RAG")
    print("=" * 80)
    print("Verification :", agentic_result["verification"])
    print("Attempts     :", agentic_result["retrieval_attempts"])
    print(
        agentic_result["answer"]
    )

    print("\n")
    print("=" * 80)
    print("END")
    print("=" * 80)
    


def main():

    print("=" * 80)
    print("RAG BENCHMARK SYSTEM — LIVE TERMINAL PROCESS VIEW")
    print("=" * 80)
    print("A status message will stay visible for each document, embedding, page load, and RAG action.")
    print("The system keeps the terminal moving and returns to the main menu after each selection.\n")

    choice = input(
        "\nCreate Embeddings? (y/n): "
    )

    if choice.lower() == "y":
        build_database()

    # Standard RAGs
    show_status("Creating RAG objects", "Initializing standard retrievers and language models")
    simple_rag = SimpleRAG()
    fusion_rag = FusionRAG()
    self_rag = SelfRAG()
    adaptive_rag = AdaptiveRAG()
    agentic_rag = AgenticRAG()
    show_embedding_status("After standard RAG object creation")

    # Graph RAG (Neo4j-based)
    show_status("Creating Graph RAG", "Initializing graph builder and query object")
    graph_builder = GraphBuilder()
    graph_rag = GraphQuery()
    show_embedding_status("After graph RAG creation")

    # Multimodal RAG (PageIndex)
    show_status("Creating multimodal RAG", "PageIndex mode is being prepared")
    multimodel_rag_pageindex = MultimodalRAG(
        enable_text=True,
        enable_tables=True,
        enable_images=True,
        enable_ocr=True,
        use_pageindex=True,
    )

    # Multimodal RAG (Legacy Chunking)
    show_status("Creating multimodal RAG", "Legacy chunking mode is being prepared")
    multimodel_rag_legacy = MultimodalRAG(
        enable_text=True,
        enable_tables=True,
        enable_images=True,
        enable_ocr=True,
        use_pageindex=False,
    )
    show_embedding_status("After multimodal RAG creation")

    # Track if graphs have been built
    graph_built = False
    multimodel_pageindex_built = False
    multimodel_legacy_built = False

    while True:

        show_menu()
        option = input(
            "\nSelect Option: "
        ).strip()

        if option == "0":
            break

        if option in ("7", "8", "9", "10", "11", "12", "13", "14", "15"):
            question = None
            if option in ("8", "13", "14", "15"):
                question = input("\nEnter Question: ")
        else:
            if option in ("1", "2", "3", "4", "5", "6"):
                question = input("\nEnter Question: ")

        # ==============================================================
        # Standard RAGs (1-6)
        # ==============================================================

        if option == "1":

            result = simple_rag.ask(question)
            print("\n")
            print("=" * 80)
            print("SIMPLE RAG ANSWER")
            print("=" * 80)
            print(result["answer"])

        elif option == "2":

            result = fusion_rag.ask(question)
            print("\n")
            print("=" * 80)
            print("FUSION RAG ANSWER")
            print("=" * 80)
            print(result["answer"])
        
        elif option == "3":

            result = self_rag.ask(question)
            print("\n")
            print("=" * 80)
            print("SELF RAG ANSWER")
            print("=" * 80)
            print(result["answer"])
        
        elif option == "4":
            
            result = adaptive_rag.ask(question)
            print("\n")
            print("=" * 80)
            print("ADAPTIVE RAG ANSWER")
            print("=" * 80)
            print(f"Selected RAG : {result['selected_rag']}")
            print(f"Reason       : {result['reason']}")
            print("\nAnswer:\n")
            print(result["answer"])

        elif option == "5":

            result = agentic_rag.ask(question)
            print("\n")
            print("=" * 80)
            print("AGENTIC RAG ANSWER")
            print("=" * 80)
            print("\nPLAN\n")
            print(result["plan"])
            print("\nRetrieved Chunks :", result["retrieved_chunks"])
            print("Verification     :", result["verification"])
            print("Attempts         :", result["retrieval_attempts"])
            print("\nANSWER\n")
            print(result["answer"])

        elif option == "6":

            compare_answers(
                question,
                simple_rag,
                fusion_rag,
                self_rag,
                adaptive_rag,
                agentic_rag,
                graph_rag
            )

        # ==============================================================
        # Graph RAG (7-8)
        # ==============================================================

        elif option == "7":
            print("\n")
            print("=" * 80)
            print("GRAPH RAG — Building Knowledge Graph (Neo4j)")
            print("=" * 80)
            print(f"PDFs found: {len(PDF_FILES)}")
            for pdf in PDF_FILES:
                print(f"  - {Path(pdf).name}")
            show_status("Graph build", "Starting Neo4j graph extraction and relation build")
            try:
                graph_builder.build()
                graph_built = True
                show_embedding_status("After graph knowledge build")
                print("\n✅ Graph RAG build complete! Graph stored in Neo4j.")
            except Exception as e:
                print(f"\n❌ Graph RAG build failed: {e}")
                print("   Make sure Neo4j is running and credentials are set in .env")

        elif option == "8":
            print("\n")
            print("=" * 80)
            print("GRAPH RAG — Asking Question")
            print("=" * 80)
            if not graph_built:
                print("⚠️  Graph may not have been built yet. Attempting retrieval anyway...")
            try:
                result = graph_rag.ask(question, top_k=20)
                print("\n")
                print("=" * 80)
                print("GRAPH RAG ANSWER")
                print("=" * 80)
                print(f"Question: {result['question']}")
                print(f"\nAnswer:\n{result['answer']}")
                print(f"\nFacts retrieved: {len(result.get('facts', []))}")
                print(f"Graph facts used to generate this answer:")
                for i, fact in enumerate(result.get('facts', []), 1):
                    print(f"  [{i}] {fact}")
            except Exception as e:
                print(f"\n❌ Graph query failed: {e}")

        # ==============================================================
        # Multimodal RAG Ingestion (9-12)
        # ==============================================================

        elif option in ("9", "10"):
            is_pageindex = (option == "9")
            rag_instance = multimodel_rag_pageindex if is_pageindex else multimodel_rag_legacy
            
            print("\n")
            print("=" * 80)
            print(f"MULTIMODAL RAG ({'PageIndex' if is_pageindex else 'Legacy Chunking'}) — Build Single PDF")
            print("=" * 80)
            
            if not PDF_FILES:
                print("No PDFs found in data/ folder.")
            else:
                print("Select PDF to build:")
                for idx, pdf in enumerate(PDF_FILES, 1):
                    print(f"  {idx}. {Path(pdf).name}")
                
                try:
                    choice_idx = int(input("\nSelect file number: ").strip()) - 1
                    if 0 <= choice_idx < len(PDF_FILES):
                        pdf = PDF_FILES[choice_idx]
                        name = Path(pdf).name
                        print(f"\n{'─'*60}")
                        print(f"  Building: {name}")
                        print(f"{'─'*60}")
                        show_status("Multimodal build", f"Processing {name} with {'PageIndex' if is_pageindex else 'Legacy Chunking'}")
                        stats = rag_instance.build(pdf)
                        if is_pageindex:
                            multimodel_pageindex_built = True
                        else:
                            multimodel_legacy_built = True
                        show_embedding_status("After single-PDF multimodal ingestion")
                        print(f"  ✅ Build complete for: {name}")
                        rag_instance.print_report()
                    else:
                        print("Invalid file number choice.")
                except ValueError:
                    print("Please enter a valid number.")
                except Exception as e:
                    print(f"  ❌ Build failed: {e}")

        elif option in ("11", "12"):
            is_pageindex = (option == "11")
            rag_instance = multimodel_rag_pageindex if is_pageindex else multimodel_rag_legacy
            
            print("\n")
            print("=" * 80)
            print(f"MULTIMODAL RAG ({'PageIndex' if is_pageindex else 'Legacy Chunking'}) — Build All PDFs")
            print("=" * 80)
            
            if not PDF_FILES:
                print("No PDFs found in data/ folder.")
            else:
                print(f"Building multimodal graph for {len(PDF_FILES)} PDF(s)...")
                show_status("Bulk build", f"Processing all {len(PDF_FILES)} PDF(s) in the data folder")
                try:
                    stats = rag_instance.build_all([Path(p) for p in PDF_FILES])
                    if is_pageindex:
                        multimodel_pageindex_built = True
                    else:
                        multimodel_legacy_built = True
                    show_embedding_status("After bulk multimodal ingestion")
                    print(f"\n{'='*60}")
                    print("  BUILD COMPLETE — ALL PDFs")
                    print(f"{'='*60}")
                    print(f"  Documents processed:  {stats.get('documents_processed', 0)}")
                    print(f"  Total entities:       {stats.get('total_entities_stored', 0)}")
                    print(f"  Total relationships:  {stats.get('total_relationships_stored', 0)}")
                    if is_pageindex:
                        print(f"  Total sections:       {stats.get('sections_processed', 0)}")
                except Exception as e:
                    print(f"\n❌ Build failed: {e}")

        # ==============================================================
        # Multimodal RAG Querying (13-15)
        # ==============================================================

        elif option == "13":
            print("\n")
            print("=" * 80)
            print("MULTIMODAL RAG — Page Reasoner (Section-Aware Retrieval)")
            print("=" * 80)
            
            if not multimodel_pageindex_built:
                print("⚠️  No PageIndex multimodal graph has been built yet. Building now...")
                for pdf in PDF_FILES:
                    try:
                        multimodel_rag_pageindex.build(pdf)
                        multimodel_pageindex_built = True
                    except Exception as e:
                        print(f"  ❌ Build failed for {pdf}: {e}")
            try:
                result = asyncio.run(multimodel_rag_pageindex.ask_with_reasoner(question, top_k=20))
                print("\n")
                print("=" * 80)
                print("MULTIMODAL RAG (Page Reasoner) — ANSWER")
                print("=" * 80)
                print(f"Question: {result['question']}")
                print(f"\nAnswer:\n{result['answer']}")
                print(f"\nSelected sections:")
                for sec in result.get('selected_sections', []):
                    pages = f"pp. {sec.get('start_index')}-{sec.get('end_index')}"
                    print(f"  [{sec['document']}] {sec['title']} {pages} (score: {sec.get('relevance_score', '?')}/10)")
                print(f"\nFacts retrieved: {len(result.get('facts', []))}")
                print(f"RAG type: {result.get('rag_type', 'multimodal_graph_reasoned')}")
            except Exception as e:
                print(f"\n❌ Reasoned query failed: {e}")

        elif option == "14":
            print("\n")
            print("=" * 80)
            print("MULTIMODAL RAG — Direct Graph Retrieval")
            print("=" * 80)
            
            if not multimodel_legacy_built and not multimodel_pageindex_built and not graph_built:
                print("⚠️  No graph has been built yet. Building legacy now...")
                for pdf in PDF_FILES:
                    try:
                        multimodel_rag_legacy.build(pdf)
                        multimodel_legacy_built = True
                    except Exception as e:
                        print(f"  ❌ Build failed for {pdf}: {e}")
            try:
                result = graph_rag.ask(question, top_k=20)
                print("\n")
                print("=" * 80)
                print("MULTIMODAL RAG (Direct) — ANSWER")
                print("=" * 80)
                print(f"Question: {result['question']}")
                print(f"\nAnswer:\n{result['answer']}")
                print(f"\nFacts retrieved: {len(result.get('facts', []))}")
                print(f"RAG type: {result.get('rag_type', 'multimodal_graph')}")
            except Exception as e:
                print(f"\n❌ Query failed: {e}")

        elif option == "15":
            print("\n")
            print("=" * 80)
            print("MULTIMODAL RAG — Multi-Doc Reasoner")
            print("=" * 80)
            
            if not multimodel_pageindex_built:
                print("⚠️  No PageIndex multimodal graph has been built yet. Building now...")
                for pdf in PDF_FILES:
                    try:
                        multimodel_rag_pageindex.build(pdf)
                        multimodel_pageindex_built = True
                    except Exception as e:
                        print(f"  ❌ Build failed for {pdf}: {e}")
            try:
                result = multimodel_rag_pageindex.ask_multidoc(question, top_k=20)
                print("\n")
                print("=" * 80)
                print("MULTIMODAL RAG (Multi-Doc Reasoner) — ANSWER")
                print("=" * 80)
                print(f"Question: {result['question']}")
                print(f"\nAnswer:\n{result['answer']}")
                print(f"\nSelected documents: {', '.join(result.get('selected_documents', []))}")
                print(f"Selected sections:")
                for sec in result.get('selected_sections', []):
                    pages = f"pp. {sec.get('start_index')}-{sec.get('end_index')}"
                    print(f"  [{sec['document']}] {sec['title']} {pages} (score: {sec.get('relevance_score', '?')}/10)")
                print(f"\nFacts retrieved: {len(result.get('facts', []))}")
                print(f"RAG type: {result.get('rag_type', 'multimodal_multidoc')}")
            except Exception as e:
                print(f"\n❌ Multi-Doc query failed: {e}")

        else:
            print("\nInvalid option. Please try again.")


if __name__ == "__main__":
    main()
