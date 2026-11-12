"""Multimodal Graph Builder — Orchestrator for ALL modalities.

NOW WITH PAGEINDEX SUPPORT:
When use_pageindex=True, replaces semantic_chunk with PageIndex tree sections.
Each section becomes one processing unit with all modalities combined.

Pipeline (PageIndex mode):
    PDF
     │
     ├── PageIndex Builder (generates hierarchical tree)
     │
     ├── For EACH section in tree:
     │      ├── Text (from section pages)
     │      ├── Tables (from section pages)
     │      ├── Images (from section pages)
     │      └── OCR (from section pages)
     │           │
     │           └── Combined context → GraphExtractor (LLM)
     │
     └── Merge all section graphs → GraphValidator → GraphStore → Neo4j

Legacy mode (use_pageindex=False — original behavior preserved):
    PDF
        ├── Text → graph_utils → graph_extractor
        ├── Tables → table_parser → table_graph_builder
        ├── Images → image_extractor → image_caption
        └── OCR → ocr_engine → graph_extractor
            │
            ▼
    Merge All Graph JSON → Validate → Store → Neo4j

All modality outputs follow the same schema:
{
    "entities": [{"id": "...", "label": "...", "properties": {...}}],
    "relationships": [{"type": "...", "source": "...", "target": "...", "properties": {...}}]
}
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from rags.graph_rag.graph_profile import GraphProfile
from rags.graph_rag.graph_utils import GraphUtils
from rags.graph_rag.graph_extractor import GraphExtractor
from rags.graph_rag.graph_validator import GraphValidator
from rags.graph_rag.graph_store import GraphStore

from rags.multimodal_rag.table_parser import TableParser
from rags.multimodal_rag.table_graph_builder import TableGraphBuilder
from rags.multimodal_rag.image_extractor import ImageExtractor
from rags.multimodal_rag.image_caption import ImageCaption
from rags.multimodal_rag.ocr_engine import OCREngine

from rags.pageindex.page_index_builder import PageIndexBuilder

logger = logging.getLogger(__name__)


class MultimodalGraphBuilder:
    """Orchestrator: PDF → [Text, Tables, Images, OCR] → Graph JSON → Validate → Neo4j.

    When use_pageindex=True, processing is section-based instead of chunk-based,
    with all modalities combined per section.
    """

    def __init__(
        self,
        enable_text: bool = True,
        enable_tables: bool = True,
        enable_images: bool = True,
        enable_ocr: bool = True,
        use_pageindex: bool = True,
        temp_dir: Optional[Path] = None,
    ):
        """
        Args:
            enable_text: Process PDF text content.
            enable_tables: Extract and process tables.
            enable_images: Extract and caption images.
            enable_ocr: OCR scanned/text-in-image content.
            use_pageindex: If True, use PageIndex sections (no chunking).
                           If False, use legacy semantic_chunk approach.
            temp_dir: Directory for temporary image storage.
        """
        self.enable_text = enable_text
        self.enable_tables = enable_tables
        self.enable_images = enable_images
        self.enable_ocr = enable_ocr
        self.use_pageindex = use_pageindex

        # Text pipeline (existing Graph RAG) — kept for legacy mode
        self.utils = GraphUtils()
        self.profile_builder = GraphProfile()
        self.extractor = GraphExtractor()

        # Table pipeline
        self.table_parser = TableParser()
        self.table_graph_builder = TableGraphBuilder()

        # Image pipeline
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="mmrag_images_"))
        self.image_extractor = ImageExtractor(output_dir=self.temp_dir)
        self.image_caption = ImageCaption()

        # OCR pipeline
        self.ocr_engine = OCREngine(backend="easyocr")

        # PageIndex (for section-based mode)
        self.page_index_builder = PageIndexBuilder()

        # Shared validation and storage
        self.validator = GraphValidator()
        self.store = GraphStore()

        # Tracking
        self.stats = {
            "text_chunks": 0,
            "text_entities": 0,
            "text_relationships": 0,
            "tables_found": 0,
            "table_entities": 0,
            "table_relationships": 0,
            "images_found": 0,
            "image_entities": 0,
            "image_relationships": 0,
            "ocr_pages": 0,
            "ocr_entities": 0,
            "ocr_relationships": 0,
            "total_sections": 0,
            "sections_processed": 0,
            "total_entities_stored": 0,
            "total_relationships_stored": 0,
        }

    def build(self, pdf_path: Path) -> Dict[str, Any]:
        """Run the full multimodal pipeline on a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Stats dict with counts for each modality.
        """
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"\n{'='*60}")
        print(f"  Multimodal Graph Builder: {pdf_path.name}")
        print(f"  Mode: {'PageIndex (no chunking)' if self.use_pageindex else 'Legacy (semantic_chunk)'}")
        print(f"{'='*60}")

        # Reset stats
        for key in self.stats:
            self.stats[key] = 0

        # Initialize Neo4j (gracefully handle if unavailable)
        neo4j_available = True
        try:
            self.store.initialize()
            logger.info("Neo4j connected successfully")
        except Exception as e:
            neo4j_available = False
            logger.warning(
                "Neo4j unavailable (%s). "
                "Pipeline will process all modalities but skip Neo4j storage. "
                "Graph JSON will still be generated and printed.",
                e,
            )
            print(f"\n  ⚠️  Neo4j not available: {e}")
            print("     Pipeline continues — modalities will be processed but NOT stored to DB.")
            print("     Graph JSON will still be generated for inspection.\n")

        all_entities: List[Dict[str, Any]] = []
        all_relationships: List[Dict[str, Any]] = []

        if self.use_pageindex:
            # ==============================================================
            # PAGEINDEX MODE — Section-based processing (NO CHUNKING)
            # ==============================================================
            result = self._build_with_pageindex(pdf_path)
            all_entities.extend(result["entities"])
            all_relationships.extend(result["relationships"])
            self.stats["sections_processed"] = result.get("sections_processed", 0)

            # Copy section-based stats
            for k, v in result.get("stats", {}).items():
                if k in self.stats:
                    self.stats[k] = v
        else:
            # ==============================================================
            # LEGACY MODE — Original modality-independent processing
            # ==============================================================

            # --- 1. Build / load profile ---
            profile = self._load_or_build_profile(pdf_path)
            extraction_prompt = self._load_extraction_prompt(pdf_path)

            # --- 2. Text Processing ---
            if self.enable_text and extraction_prompt:
                text_graph = self._process_text(pdf_path, extraction_prompt, profile)
                all_entities.extend(text_graph["entities"])
                all_relationships.extend(text_graph["relationships"])

            # --- 3. Table Processing ---
            if self.enable_tables:
                table_graph = self._process_tables(pdf_path)
                all_entities.extend(table_graph["entities"])
                all_relationships.extend(table_graph["relationships"])

            # --- 4. Image Processing ---
            if self.enable_images:
                image_graph = self._process_images(pdf_path)
                all_entities.extend(image_graph["entities"])
                all_relationships.extend(image_graph["relationships"])

            # --- 5. OCR Processing ---
            if self.enable_ocr:
                ocr_graph = self._process_ocr(pdf_path, extraction_prompt)
                all_entities.extend(ocr_graph["entities"])
                all_relationships.extend(ocr_graph["relationships"])

        # --- 6. Merge, Validate, Store (shared for both modes) ---
        print(f"\n{'='*60}")
        print(f"  Merging & Storing Combined Graph")
        print(f"{'='*60}")

        merged_graph = {
            "entities": all_entities,
            "relationships": all_relationships,
        }

        print(f"  Total raw entities: {len(all_entities)}")
        print(f"  Total raw relationships: {len(all_relationships)}")

        # Validate (dedup, referential integrity, type checking)
        validated = self.validator.validate(merged_graph)

        print(f"  After validation:")
        print(f"    Valid entities: {len(validated['entities'])}")
        print(f"    Valid relationships: {len(validated['relationships'])}")

        # Store to Neo4j
        if neo4j_available:
            if validated["entities"] or validated["relationships"]:
                try:
                    store_stats = self.store.insert_graph(validated)
                    self.stats["total_entities_stored"] = store_stats.get("entities_created", 0)
                    self.stats["total_relationships_stored"] = store_stats.get("relationships_created", 0)
                    print(f"  Stored to Neo4j:")
                    print(f"    Entities created: {store_stats['entities_created']}")
                    print(f"    Entities skipped: {store_stats['entities_skipped']}")
                    print(f"    Relationships created: {store_stats['relationships_created']}")
                    print(f"    Relationships skipped: {store_stats['relationships_skipped']}")
                except Exception as e:
                    neo4j_available = False
                    logger.warning("Failed to store to Neo4j: %s", e)
                    print(f"\n  ⚠️  Neo4j storage failed: {e}")
                    print("     Graph JSON was generated but NOT stored to DB.\n")
        else:
            print(f"\n  ⚠️  Neo4j unavailable — graph JSON was generated but NOT stored to DB.")
            print(f"      Validated entities: {len(validated['entities'])}")
            print(f"      Validated relationships: {len(validated['relationships'])}")

        try:
            self.store.close()
        except Exception:
            pass

        # --- Print final report ---
        print(f"\n{'='*60}")
        print(f"  Build Complete: {pdf_path.name}")
        print(f"{'='*60}")
        if self.use_pageindex:
            print(f"  Mode:     PageIndex (sections: {self.stats['total_sections']}, processed: {self.stats['sections_processed']})")
        else:
            print(f"  Mode:     Legacy chunk-based")
        print(f"  Text:    {self.stats['text_chunks']} chunks → {self.stats['text_entities']} entities, {self.stats['text_relationships']} rels")
        print(f"  Tables:  {self.stats['tables_found']} tables → {self.stats['table_entities']} entities, {self.stats['table_relationships']} rels")
        print(f"  Images:  {self.stats['images_found']} images → {self.stats['image_entities']} entities, {self.stats['image_relationships']} rels")
        print(f"  OCR:     {self.stats['ocr_pages']} pages → {self.stats['ocr_entities']} entities, {self.stats['ocr_relationships']} rels")
        print(f"  Stored:  {self.stats['total_entities_stored']} entities, {self.stats['total_relationships_stored']} relationships")

        return dict(self.stats)

    # ==============================================================
    # PAGEINDEX-BASED BUILD (No Chunking)
    # ==============================================================

    def _build_with_pageindex(self, pdf_path: Path) -> Dict[str, Any]:
        """Build graph using PageIndex sections — each section gets all modalities combined.

        Flow:
        1. Build/load PageIndex tree
        2. For each leaf section:
           a. Get text from section pages
           b. Get tables from section pages
           c. Get images from section pages
           d. Get OCR from section pages
           e. Combine ALL into one prompt → GraphExtractor (LLM)
        3. Collect all section graphs
        """
        print(f"\n  --- Building PageIndex Tree ---")
        tree = self.page_index_builder.build(pdf_path, force_rebuild=False)
        all_sections = self.page_index_builder.get_leaf_sections(tree)
        self.stats["total_sections"] = len(all_sections)

        print(f"  Total leaf sections: {len(all_sections)}")
        print(f"  Doc description: {tree.get('doc_description', 'N/A')[:100]}...")

        # Print tree overview
        self.page_index_builder.print_tree(tree)

        # Load extraction prompt (from profile)
        extraction_prompt = self._load_extraction_prompt(pdf_path)
        if not extraction_prompt:
            print(f"  ⚠️  No extraction prompt found. Building profile first.")
            profile = self._load_or_build_profile(pdf_path)
            extraction_prompt = self._load_extraction_prompt(pdf_path)

        # Extract tables and images ONCE per PDF (they're needed across sections)
        all_tables = self.table_parser.extract_tables_from_pdf(pdf_path) if self.enable_tables else []
        all_images = self.image_extractor.extract_from_pdf(pdf_path, save_images=True) if self.enable_images else []
        ocr_results = []
        if self.enable_ocr and OCREngine.needs_ocr(pdf_path):
            ocr_results = self.ocr_engine.ocr_pdf(pdf_path)
        elif self.enable_ocr:
            print(f"  PDF has extractable text. OCR skipped.")
        else:
            ocr_results = []

        self.stats["tables_found"] = len(all_tables)
        self.stats["images_found"] = len(all_images)
        self.stats["ocr_pages"] = len(ocr_results)

        all_entities: List[Dict[str, Any]] = []
        all_relationships: List[Dict[str, Any]] = []

        # For each leaf section, process with ALL modalities
        for idx, section in enumerate(all_sections, 1):
            section_id = section.get("node_id", "?")
            section_title = section.get("title", "Unknown")
            start_page = section.get("start_index", 1)
            end_page = section.get("end_index", 1)

            print(f"\n  {'─'*50}")
            print(f"  Processing Section {idx}/{len(all_sections)}: [{section_id}] {section_title}")
            print(f"  Pages: {start_page}-{end_page}")

            # Build combined context for this section
            section_context = self._build_section_context(
                pdf_path=pdf_path,
                section=section,
                start_page=start_page,
                end_page=end_page,
                all_tables=all_tables,
                all_images=all_images,
                ocr_results=ocr_results,
            )

            if not section_context.strip() or len(section_context.split()) < 30:
                print(f"  ⏭️  Section too short, skipping.")
                continue

            # Run GraphExtractor with the COMBINED context
            chunk = {
                "page": start_page,
                "chunk": idx,
                "heading": section_title,
                "section_id": section_id,
                "text": section_context,
                "section_title": section_title,
                "page_range": f"{start_page}-{end_page}",
                "source": "pageindex_section",
            }

            try:
                extracted = self.extractor.extract(extraction_prompt, chunk)

                # Add section metadata to each entity and relationship
                for entity in extracted.get("entities", []):
                    props = entity.get("properties", {})
                    props["section_id"] = section_id
                    props["section_title"] = section_title
                    props["page_range"] = f"{start_page}-{end_page}"
                    props["document"] = pdf_path.name
                    entity["properties"] = props

                for rel in extracted.get("relationships", []):
                    rel_props = rel.get("properties", {})
                    rel_props["section_id"] = section_id
                    rel_props["section_title"] = section_title
                    rel_props["page_range"] = f"{start_page}-{end_page}"
                    rel_props["document"] = pdf_path.name
                    rel["properties"] = rel_props

                all_entities.extend(extracted.get("entities") or [])
                all_relationships.extend(extracted.get("relationships") or [])
                self.stats["text_chunks"] += 1  # Reuse for count of sections processed

                ents = len(extracted.get("entities", []))
                rels = len(extracted.get("relationships", []))
                print(f"  ✓ Extracted: {ents} entities, {rels} relationships")

            except Exception as e:
                logger.warning(f"Section {section_id} extraction failed: {e}")
                print(f"  ✗ Extraction failed: {e}")
                continue

        self.stats["text_entities"] = len(all_entities)
        self.stats["text_relationships"] = len(all_relationships)
        self.stats["sections_processed"] = self.stats["text_chunks"]

        return {
            "entities": all_entities,
            "relationships": all_relationships,
            "sections_processed": self.stats["sections_processed"],
            "stats": dict(self.stats),
        }

    def _build_section_context(
        self,
        pdf_path: Path,
        section: Dict[str, Any],
        start_page: int,
        end_page: int,
        all_tables: List[Dict[str, Any]],
        all_images: List[Dict[str, Any]],
        ocr_results: List[Dict[str, Any]],
    ) -> str:
        """Build a combined context string for a section — text + tables + images + OCR.

        Args:
            pdf_path: Path to the PDF.
            section: PageIndex section dict.
            start_page: Section start page.
            end_page: Section end page.
            all_tables: All tables extracted from the PDF (will be filtered by page).
            all_images: All images extracted from the PDF (will be filtered by page).
            ocr_results: All OCR results (will be filtered by page).

        Returns:
            Combined context string with all modalities for this section.
        """
        parts = []

        # --- 1. Section header ---
        section_title = section.get("title", "Unknown")
        section_summary = section.get("summary", "")
        parts.append(f"=== SECTION: {section_title} (Pages {start_page}-{end_page}) ===")
        if section_summary:
            parts.append(f"Summary: {section_summary}")
        parts.append("")

        # --- 2. Text from section pages ---
        text_content = section.get("text", "")
        if text_content:
            parts.append("--- TEXT CONTENT ---")
            parts.append(text_content)
            parts.append("")

        # --- 3. Tables in section page range ---
        if self.enable_tables:
            section_tables = [
                t for t in all_tables
                if start_page <= t.get("page", 0) <= end_page
            ]
            if section_tables:
                parts.append(f"--- TABLES ({len(section_tables)} found in this section) ---")
                for table_data in section_tables:
                    parts.append(TableParser.table_to_text(table_data))
                    parts.append("")

        # --- 4. Images in section page range ---
        if self.enable_images:
            section_images = [
                img for img in all_images
                if start_page <= img.get("page", 0) <= end_page
            ]
            if section_images:
                parts.append(f"--- IMAGES ({len(section_images)} found in this section) ---")
                for img_data in section_images:
                    img_page = img_data.get("page", 0)
                    img_idx = img_data.get("image_index", 0)
                    img_size = f"{img_data.get('width', 0)}x{img_data.get('height', 0)}"
                    parts.append(f"[Image on page {img_page}, index {img_idx}, size {img_size}]")

                    # Try to get caption if available
                    caption_path = img_data.get("path", "")
                    if caption_path:
                        parts.append(f"  Location: {caption_path}")
                    parts.append("")

        # --- 5. OCR results in section page range ---
        if self.enable_ocr and ocr_results:
            section_ocr = [
                ocr for ocr in ocr_results
                if start_page <= ocr.get("page", 0) <= end_page
            ]
            if section_ocr:
                parts.append(f"--- OCR TEXT ({len(section_ocr)} pages in this section) ---")
                for ocr_page in section_ocr:
                    ocr_text = ocr_page.get("text", "").strip()
                    if ocr_text:
                        parts.append(f"[OCR from page {ocr_page.get('page', 0)}]")
                        parts.append(ocr_text)
                        parts.append("")

        context = "\n".join(parts).strip()
        return context

    # ==============================================================
    # LEGACY METHODS (unchanged — for use_pageindex=False mode)
    # ==============================================================

    def _load_or_build_profile(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Load existing profile or build a new one."""
        profile_json_path = Path("profiles") / f"{pdf_path.stem}_profile.json"

        if profile_json_path.exists():
            return json.loads(profile_json_path.read_text(encoding="utf-8"))

        # Build new profile
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])[:120000]

        if text.strip():
            self.profile_builder.build_profile(pdf_path.name, text)

        if profile_json_path.exists():
            return json.loads(profile_json_path.read_text(encoding="utf-8"))

        return None

    def _load_extraction_prompt(self, pdf_path: Path) -> str:
        """Load the extraction prompt for a PDF."""
        prompt_file = Path("profiles") / f"{pdf_path.stem}_extraction_prompt.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return ""

    def _process_text(
        self,
        pdf_path: Path,
        extraction_prompt: str,
        profile: Optional[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy: Run text-based graph extraction on chunks."""
        print(f"\n  --- Text Processing (legacy chunk-based) ---")

        chunks = self.utils.process_pdf(pdf_path, profile=profile)
        self.stats["text_chunks"] = len(chunks)
        print(f"  Chunks: {len(chunks)}")

        all_entities: List[Dict[str, Any]] = []
        all_relationships: List[Dict[str, Any]] = []

        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if len(text.split()) < 25:
                continue

            try:
                extracted = self.extractor.extract(extraction_prompt, chunk)
                # Add document metadata to each entity
                page = chunk.get("page", 0)
                for entity in extracted.get("entities") or []:
                    props = entity.get("properties", {})
                    props["document"] = pdf_path.name
                    props["page_range"] = str(page)
                    props["source_modality"] = "text"
                    entity["properties"] = props
                for rel in extracted.get("relationships") or []:
                    rel_props = rel.get("properties", {})
                    rel_props["document"] = pdf_path.name
                    rel_props["page_range"] = str(page)
                    rel_props["source_modality"] = "text"
                    rel["properties"] = rel_props
                all_entities.extend(extracted.get("entities") or [])
                all_relationships.extend(extracted.get("relationships") or [])
            except Exception as e:
                logger.warning(f"Text extraction failed for chunk {chunk.get('chunk')}: {e}")

        self.stats["text_entities"] = len(all_entities)
        self.stats["text_relationships"] = len(all_relationships)
        print(f"  Entities: {len(all_entities)}, Relationships: {len(all_relationships)}")

        return {"entities": all_entities, "relationships": all_relationships}

    def _process_tables(self, pdf_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy: Extract tables and convert to graph JSON."""
        print(f"\n  --- Table Processing (legacy) ---")

        tables = self.table_parser.extract_tables_from_pdf(pdf_path)
        self.stats["tables_found"] = len(tables)
        print(f"  Tables found: {len(tables)}")

        all_entities: List[Dict[str, Any]] = []
        all_relationships: List[Dict[str, Any]] = []

        for table_data in tables:
            self.table_graph_builder.source_pdf = pdf_path.name
            graph = self.table_graph_builder.build_graph(table_data)
            page = table_data.get("page", 0)
            for entity in graph.get("entities") or []:
                props = entity.get("properties", {})
                props["document"] = pdf_path.name
                props["page_range"] = str(page)
                props["source_modality"] = "table"
                entity["properties"] = props
            for rel in graph.get("relationships") or []:
                rel_props = rel.get("properties", {})
                rel_props["document"] = pdf_path.name
                rel_props["page_range"] = str(page)
                rel_props["source_modality"] = "table"
                rel["properties"] = rel_props
            all_entities.extend(graph.get("entities") or [])
            all_relationships.extend(graph.get("relationships") or [])

        self.stats["table_entities"] = len(all_entities)
        self.stats["table_relationships"] = len(all_relationships)
        print(f"  Entities: {len(all_entities)}, Relationships: {len(all_relationships)}")

        return {"entities": all_entities, "relationships": all_relationships}

    def _process_images(self, pdf_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy: Extract images, caption them, and convert to graph JSON."""
        print(f"\n  --- Image Processing (legacy) ---")

        images = self.image_extractor.extract_from_pdf(pdf_path, save_images=True)
        self.stats["images_found"] = len(images)
        print(f"  Images extracted: {len(images)}")

        all_entities: List[Dict[str, Any]] = []
        all_relationships: List[Dict[str, Any]] = []

        if images:
            captioned = self.image_caption.caption_images_batch(images)
            for img_data, result in zip(images, captioned):
                page = img_data.get("page", 0)
                graph = result.get("graph", {})
                for entity in graph.get("entities") or []:
                    props = entity.get("properties", {})
                    props["document"] = pdf_path.name
                    props["page_range"] = str(page)
                    props["source_modality"] = "image"
                    entity["properties"] = props
                for rel in graph.get("relationships") or []:
                    rel_props = rel.get("properties", {})
                    rel_props["document"] = pdf_path.name
                    rel_props["page_range"] = str(page)
                    rel_props["source_modality"] = "image"
                    rel["properties"] = rel_props
                all_entities.extend(graph.get("entities") or [])
                all_relationships.extend(graph.get("relationships") or [])

        self.stats["image_entities"] = len(all_entities)
        self.stats["image_relationships"] = len(all_relationships)
        print(f"  Entities: {len(all_entities)}, Relationships: {len(all_relationships)}")

        return {"entities": all_entities, "relationships": all_relationships}

    def _process_ocr(
        self,
        pdf_path: Path,
        extraction_prompt: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy: OCR scanned pages and run text extraction on results."""
        print(f"\n  --- OCR Processing (legacy) ---")

        needs_ocr = OCREngine.needs_ocr(pdf_path)
        if not needs_ocr and self.enable_text:
            print(f"  PDF has extractable text. OCR skipped (text pipeline handles it).")
            return {"entities": [], "relationships": []}

        print(f"  Running OCR on all pages...")
        ocr_results = self.ocr_engine.ocr_pdf(pdf_path)
        self.stats["ocr_pages"] = len(ocr_results)
        print(f"  OCR completed for {len(ocr_results)} pages")

        all_entities: List[Dict[str, Any]] = []
        all_relationships: List[Dict[str, Any]] = []

        if extraction_prompt:
            for page_result in ocr_results:
                text = (page_result.get("text") or "").strip()
                if len(text.split()) < 25:
                    continue

                page = page_result.get("page", 0)
                chunk = {
                    "page": page,
                    "chunk": 1,
                    "heading": "OCR Text",
                    "text": text,
                    "source": "ocr",
                }

                try:
                    extracted = self.extractor.extract(extraction_prompt, chunk)
                    for entity in extracted.get("entities") or []:
                        props = entity.get("properties", {})
                        props["document"] = pdf_path.name
                        props["page_range"] = str(page)
                        props["source_modality"] = "ocr"
                        entity["properties"] = props
                    for rel in extracted.get("relationships") or []:
                        rel_props = rel.get("properties", {})
                        rel_props["document"] = pdf_path.name
                        rel_props["page_range"] = str(page)
                        rel_props["source_modality"] = "ocr"
                        rel["properties"] = rel_props
                    all_entities.extend(extracted.get("entities") or [])
                    all_relationships.extend(extracted.get("relationships") or [])
                except Exception as e:
                    logger.warning(f"OCR extraction failed for page {page_result.get('page')}: {e}")

        self.stats["ocr_entities"] = len(all_entities)
        self.stats["ocr_relationships"] = len(all_relationships)
        print(f"  Entities: {len(all_entities)}, Relationships: {len(all_relationships)}")

        return {"entities": all_entities, "relationships": all_relationships}


# ---------------------------------------------------------------------------
# Standalone Testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    def demo():
        # Test with PageIndex mode (no chunking)
        print("\n" + "=" * 60)
        print("  TEST: PageIndex Mode (No Chunking)")
        print("=" * 60)

        builder = MultimodalGraphBuilder(
            enable_text=True,
            enable_tables=True,
            enable_images=True,
            enable_ocr=True,
            use_pageindex=True,  # NEW: section-based processing
        )

        pdf_path = Path("data/hemant_story.pdf")
        if pdf_path.is_file():
            stats = builder.build(pdf_path)
            print(f"\nFinal stats: {json.dumps(stats, indent=2)}")
        else:
            print(f"No test PDF found at {pdf_path}")

        print("\n")
        # Uncomment to test legacy mode:
        # builder2 = MultimodalGraphBuilder(use_pageindex=False)
        # builder2.build(pdf_path)

    demo()
