# PageIndex Integration Plan

## Vision: PageIndex as the Foundation Layer

```
                    PDF
                     │
              PageIndex (Core)
                     │
         ┌───────────┴───────────┐
         │                       │
  Multimodal Processing     Graph Construction
(Text, Tables, Images, OCR)      │
         └───────────┬───────────┘
                     │
                   Neo4j
                     │
            Page-aware Retrieval
                     │
              Answer Generation
```

## Key Principle

- **graph_rag stays UNCHANGED** — When `app.py` calls graph_rag, it works independently (builder → extractor → validator → store → retriever → query)
- **multimodal_rag** is the EXTENSION that uses graph_rag folder + multimodal_rag folder + pageindex folder
- **PageIndex** is the FOUNDATION that replaces current semantic_chunk() with hierarchical document tree

---

## Phase 1 — PageIndex Foundation (`rags/pageindex/`) ✅ COMPLETE

### 1.1 ✅ Create `page_index_builder.py`
- ✅ Wraps the PageIndex library from `_pageindex_temp/pageindex/`
- ✅ Takes PDFs → generates hierarchical tree structure (TOC-style)
- ✅ Saves `_pageindex.json` for every PDF in `profiles/` folder
- ✅ Methods: `build()`, `load()`, `load_all()`, `tree_context()`, `print_tree()`, `tree_summary()`, `get_leaf_sections()`, `get_section()`, `get_sections_by_page_range()`
- ✅ PageIndex library copied to `rags/pageindex_lib/`
- Output format:
```json
{
  "doc_name": "hemant_story.pdf",
  "doc_description": "...",
  "structure": [
    {
      "title": "Employment Details",
      "node_id": "0001",
      "start_index": 1,
      "end_index": 3,
      "summary": "...",
      "nodes": [
        {
          "title": "Personal Information",
          "node_id": "0002",
          "start_index": 1,
          "end_index": 1,
          "summary": "..."
        }
      ]
    }
  ]
}
```

### 1.2 ✅ Create `page_reasoner.py`
- ✅ Takes a user question + PageIndex tree(s) for all PDFs
- ✅ Uses LLM (2-stage: document selection → section selection) to identify relevant sections
- ✅ Returns relevant section IDs, titles, summaries, page ranges, relevance scores
- ✅ `gather_context()` builds full context string for answer generation LLM
- ✅ No vector DB used — pure LLM reasoning over tree structure
- ✅ Narrow search space before querying Neo4j

### 1.3 ✅ Create `page_search.py`
- ✅ Search across multiple PDF PageIndex trees (keyword-based, no vector DB)
- ✅ Cross-document section matching by title/summary/text
- ✅ Section-based context gathering with `get_section_context()`
- ✅ `search_across_profiles()` — auto-loads all cached PageIndex files
- ✅ `find_overlapping_sections()` — find sections spanning a specific page
- ✅ `get_sections_by_keywords()` — keyword intersection search

---

## Phase 2 — Replace Chunking in Multimodal Pipeline ✅ COMPLETE

### 2.1 ✅ Modify `multimodal_graph_builder.py`
- ✅ NEW `use_pageindex=True/False` flag — dual-mode support
- ✅ PageIndex mode: Each section becomes one processing unit (no chunking)
- ✅ Legacy mode preserved: When `use_pageindex=False`, original chunk-based behavior works exactly as before
- ✅ `_build_with_pageindex()` — orchestrates section-based processing
- ✅ `_build_section_context()` — combines text + tables + images + OCR for one section
- ✅ Section metadata (section_id, section_title, page_range) added to every entity and relationship
- ✅ Text, tables, images, OCR are all scoped per-section (filtered by page range)
- ✅ GraphExtractor receives the FULL combined context of the section, not isolated modalities
- ✅ Stats tracking for sections

### 2.2 New Processing Flow (PageIndex mode)
```
PDF
 │
 ├── PageIndex Builder (generates hierarchical tree)
 │
 ├── For EACH leaf section:
 │      ├── Text (from section pages)
 │      ├── Tables (from section pages, via TableParser)
 │      ├── Images (from section pages, via ImageExtractor)
 │      └── OCR (from section pages, via OCREngine)
 │           │
 │           └── Combined context → GraphExtractor (LLM)
 │                        section_id, section_title, page_range
 │                            added to entities/relationships
 │
 └── Merge all section graphs → GraphValidator → GraphStore → Neo4j
```

---

## Phase 3 — Page-Aware Graph Extraction ✅ COMPLETE

### 3.1 ✅ Each entity/relationship gets section metadata
```json
{
  "id": "Employee::Hemant Sharma",
  "label": "Employee",
  "properties": {
    "name": "Hemant Sharma",
    "section_id": "0001",
    "section_title": "Employment Details",
    "page_range": "1-3",
    "source_modality": "text",
    "document": "hemant_story.pdf",
    ...
  }
}
```

### 3.2 ✅ GraphExtractor receives full section context
```
Section: "Employment Details" (pages 1-3)
  Text chunk + Table data + Image captions + OCR text
       │
       ▼
  GraphExtractor LLM
       │
       ▼
  Entities + Relationships (with section metadata)
```

### 3.3 ✅ Implementation in `multimodal_graph_builder.py`
- `_build_with_pageindex()` — iterates over PageIndex leaf sections
- `_build_section_context()` — combines text + tables + images + OCR per section
- Section metadata (section_id, section_title, page_range, document) added to every entity and relationship
- GraphExtractor receives the FULL combined context of the section, not isolated modalities

### 3.4 ✅ Fix: Model updated to working free model
- `google/gemini-2.0-flash-exp:free` was removed from OpenRouter
- Changed to `google/gemma-4-31b-it:free` (available and free)
- Updated in: `page_index_builder.py`, `config.yaml`
- `graph_rag/llm_config.py` uses different models (OpenRouter for construction, Cerebras for answer) — kept as-is

---

## Phase 4 — Page-Aware Retrieval

### 4.1 New Query Flow
```
Question
  │
  ├── Page Reasoner
  │    ├── Scans PageIndex trees of ALL PDFs
  │    ├── Identifies relevant sections
  │    └── Returns section IDs + summaries
  │
  ├── GraphRetriever (narrowed by sections)
  │    ├── Queries Neo4j for entities in those sections
  │    └── Returns facts
  │
  ├── Section Context Gatherer
  │    ├── Collects all modality data for relevant sections
  │    └── Returns text + tables + images + OCR for those sections
  │
  └── GraphQuery (LLM with enriched context)
       ├── Question
       ├── Relevant section summaries
       ├── Graph facts
       └── Multimodal evidence
            │
            ▼
       Final Answer
```

### 4.2 GraphRetriever modifications (optional, in multimodal_rag context)
- Add section-aware Cypher queries
- Filter by section_id when PageReasoner provides it

---

## Phase 5 — Integration with app.py ⏳ PENDING (multimodal_rag.py done, app.py pending)

### 5.1 ✅ `multimodal_rag.py` updated (internal API ready)
- ✅ `use_pageindex=True/False` flag — dual-mode support for both build and query
- ✅ `build()` — now caches PageIndex trees for reasoned queries
- ✅ `build_all()` — process multiple PDFs with combined stats
- ✅ `ask()` — backward-compatible direct graph retrieval
- ✅ NEW `ask_with_reasoner()` — full PageReasoner flow:
    1. PageReasoner selects relevant sections from ALL cached trees
    2. Gathers section context (summaries + text)
    3. Retrieves graph facts from Neo4j
    4. Feeds enriched prompt to Cerebras LLM
- ✅ `_generate_with_section_context()` — custom prompt with section summaries + graph facts
- ✅ `_load_all_trees()` — auto-loads all cached PageIndex trees from profiles/
- ✅ `query_with_reasoner()` — sync wrapper

### 5.2 ⏳ `app.py` NOT yet updated — No CLI option for Multimodal RAG yet
Missing from `app.py`:
- [ ] Option to select "Multimodal Graph RAG (PageIndex)"
- [ ] Option to select "Multimodal Graph RAG (Legacy Chunking)"
- [ ] Multi-PDF build for multimodal_rag
- [ ] ask_with_reasoner() vs ask() choice

### 5.3 graph_rag stays 100% standalone
```python
# Existing graph_rag still works independently
from rags.graph_rag.graph_builder import GraphBuilder
from rags.graph_rag.graph_query import GraphQuery

builder = GraphBuilder()
builder.build()

query = GraphQuery()
result = query.query("What is Hemant's salary?")
```

---

## Phase 6 — Handling 7+ PDFs (Scalability)

### 6.1 Multi-PDF PageIndex
- One PageIndex tree per PDF
- Cross-document Page Reasoner that searches across all trees
- Document-level filtering before section-level search

### 6.2 Storage
- PageIndex trees stored in `profiles/<pdf_name>_pageindex.json`
- All entities in Neo4j tagged with `document` and `section_id`
- Quick document/section lookups

### 6.3 Retrieval Strategy
```
Question
  │
  ├── Step 1: Document Selection (which PDFs are relevant?)
  │   ├── Page Reasoner compares question vs doc_description
  │   └── Selects top 1-3 PDFs
  │
  ├── Step 2: Section Selection (which sections within those PDFs?)
  │   ├── Page Reasoner compares question vs section summaries
  │   └── Selects top 2-5 sections
  │
  ├── Step 3: Graph Retrieval (narrowed queries)
  │   ├── Cypher queries filtered by document + section_id
  │   └── Returns facts
  │
  └── Step 4: Answer Generation
```

---

## File Hierarchy (Final)

```
rags/
│
├── pageindex/                    ★ NEW - Foundation Layer
│      __init__.py
│      page_index_builder.py      ★ Wraps PageIndex library for PDF tree generation
│      page_reasoner.py           ★ LLM-based section selector for queries
│      page_search.py             ★ Cross-document section search
│
├── graph_rag/                    ★ UNCHANGED - Standalone Graph RAG
│      graph_profile.py
│      graph_extractor.py
│      graph_validator.py
│      graph_store.py
│      graph_retriever.py
│      graph_query.py
│      graph_builder.py
│      graph_utils.py
│      graph_schema.py
│      llm_config.py
│
├── multimodal_rag/               ★ MODIFIED - Uses PageIndex
│      __init__.py
│      multimodal_rag.py          ★ Updated: build() uses PageIndex
│      multimodal_graph_builder.py ★ Updated: section-based processing
│      table_parser.py
│      table_graph_builder.py
│      image_extractor.py
│      image_caption.py
│      ocr_engine.py
│      table_processor.py
│      image_processor.py
```

---

## Implementation Order (Actual)

| # | Phase | What | Status | File(s) |
|---|-------|------|--------|---------|
| 1 | Phase 1a | PageIndex Builder — wraps PageIndex library for tree generation | ✅ Done | `page_index_builder.py` |
| 2 | Phase 1b | PageReasoner — LLM-based section selector for queries | ✅ Done | `page_reasoner.py` |
| 3 | Phase 1c | PageSearch — cross-document keyword search | ✅ Done | `page_search.py` |
| 4 | Phase 2 | Replace Chunking — section-based processing in multimodal_graph_builder | ✅ Done | `multimodal_graph_builder.py` |
| 5 | Phase 3 | Page-Aware Graph Extraction — section metadata in entities/relationships | ✅ Done | `multimodal_graph_builder.py` |
| 6 | Phase 4 | **Page-Aware Retrieval** — section-filtered graph queries (wraps GraphRetriever) | ✅ Done | `page_retriever.py` |
| 7 | Phase 5 | app.py Integration — add Multimodal Graph RAG options | 🔜 Pending | `app.py` |
| 8 | Phase 6 | Multi-document Scalability — support 7+ PDFs | 🔜 Pending | Various |
| 9 | Phase 7 | Benchmarking — compare Graph RAG vs Multimodal RAG vs PageIndex RAG | 🔜 Pending | `benchmark.py` |

## Current Status

```
✅ Phase 1 (a-c):  PageIndex Foundation (Builder + Reasoner + Search)
✅ Phase 2:        Replace Chunking with PageIndex sections
✅ Phase 3:        Page-Aware Graph Extraction (section metadata)
✅ Phase 4:        Page-Aware Retrieval (section-filtered Cypher) ← DONE
⏳ Phase 5:        app.py Integration (multimodal_rag.py ready, app.py pending)
⏳ Phase 6:        Multi-document Scalability (7+ PDFs)
⏳ Phase 7:        Benchmarking
```

#The PageIndex tree structure (sections with titles, summaries, page ranges) is stored as JSON files in profiles/ — not in any vector database.

# CRITICAL RULE: graph_rag/ folder MUST stay 100% UNCHANGED.
# When app.py calls graph_rag, it works independently.
# When app.py calls multimodal_rag, it uses graph_rag + multimodal_rag + pageindex.
# Do NOT modify graph_rag/graph_retriever.py or graph_rag/graph_query.py directly.
# Instead, create a new section-aware retriever in multimodal_rag/ or pageindex/.
