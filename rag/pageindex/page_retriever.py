"""Page Retriever — Section-aware graph retrieval wrapper.

Wraps GraphRetriever from graph_rag/ to add section-filtered Cypher queries
using PageIndex metadata (section_id, document, page_range).

CRITICAL: graph_rag/graph_retriever.py is NEVER modified.
This module sits on top and adds section-awareness.

When PageReasoner provides relevant sections:
  Cypher queries are filtered by document + section_id
  → Faster, more relevant results

When no sections are provided:
  Falls back to the standard GraphRetriever.retrieve()

Usage:
    retriever = PageRetriever()
    facts = retriever.retrieve(question, sections=[...])
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from rags.graph_rag.graph_retriever import GraphRetriever

load_dotenv()

logger = logging.getLogger(__name__)


class PageRetriever:
    """Section-aware graph retriever that wraps GraphRetriever.

    Two modes:
    1. Section mode: Uses section_id + document filters in Cypher queries
       → Faster, more precise, narrows search space
    2. Fallback mode: Delegates to the standard GraphRetriever.retrieve()
       → Backward compatible, works when no PageIndex is available
    """

    def __init__(self):
        """Initialize with a standard GraphRetriever as fallback."""
        # Reuse the same Neo4j env vars as GraphRetriever
        self._neo_uri = os.getenv("NEO4J_URI")
        self._neo_user = os.getenv("NEO4J_USERNAME")
        self._neo_pass = os.getenv("NEO4J_PASSWORD")
        self._driver = None

        # Fallback retriever (used when no sections provided)
        self._fallback = GraphRetriever()

    def _ensure_driver(self) -> bool:
        """Lazily create and verify Neo4j driver. Returns True if connected."""
        if self._driver is not None:
            return True
        if not self._neo_uri or not self._neo_user or not self._neo_pass:
            logger.warning("[PageRetriever] Missing Neo4j env vars — retrieval disabled")
            return False
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._neo_uri, auth=(self._neo_user, self._neo_pass)
            )
            self._driver.verify_connectivity()
            logger.info("[PageRetriever] Neo4j connected successfully")
            return True
        except Exception as e:
            logger.warning("[PageRetriever] Neo4j unavailable (%s) — using fallback", e)
            self._driver = None
            return False

    def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
        self._driver = None
        self._fallback.close()

    # ==============================================================
    # Main Retrieval Method
    # ==============================================================

    def retrieve(
        self,
        question: str,
        limit: int = 40,
        sections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Retrieve graph facts, optionally filtered by sections.

        Args:
            question: Natural language question.
            limit: Maximum number of facts to return.
            sections: Optional list of section dicts from PageReasoner.
                      Each dict should have:
                        - "document": str (PDF filename stem)
                        - "section_id": str (PageIndex node_id)
                        - "title": str (section title)
                        - "start_index": int (page start)
                        - "end_index": int (page end)
                        - "summary": str (section summary)

        Returns:
            List of fact strings.
        """
        if not sections:
            # No section context — use standard GraphRetriever
            logger.info("[PageRetriever] No sections provided. Using fallback GraphRetriever.")
            return self._fallback.retrieve(question, limit)

        # Section-aware retrieval
        logger.info(
            "[PageRetriever] Section-aware retrieval with %d section(s)",
            len(sections),
        )

        if not self._ensure_driver():
            logger.warning("[PageRetriever] Neo4j unavailable. Using fallback GraphRetriever.")
            return self._fallback.retrieve(question, limit)

        facts: List[str] = []
        seen_facts: set = set()

        with self._driver.session() as session:
            # --- Strategy 1: Per-section entity retrieval ---
            # For each section, query entities that were created from that section
            for sec in sections:
                document = sec.get("document", "")
                section_id = sec.get("section_id", "")
                title = sec.get("title", "")

                if not section_id:
                    continue

                # Query 1: Get entities with matching section_id
                try:
                    res = session.run(
                        """
                        MATCH (n)
                        WHERE n.section_id = $section_id
                          AND n.document = $document
                          AND n.name IS NOT NULL
                        OPTIONAL MATCH (n)-[r]->(m)
                        WHERE m.name IS NOT NULL
                        RETURN n.name AS source_name,
                               n.section_title AS section_title,
                               n.source_modality AS modality,
                               type(r) AS rel_type,
                               m.name AS target_name
                        LIMIT $limit
                        """,
                        section_id=section_id,
                        document=document,
                        limit=limit,
                    )
                    for row in res:
                        source = row.get("source_name")
                        rel = row.get("rel_type")
                        target = row.get("target_name")
                        mod = row.get("modality") or "text"
                        sec_title = row.get("section_title") or title

                        if source and rel and target:
                            fact = f"[{sec_title}] {source} -[{rel}]-> {target} (modality: {mod})"
                        elif source:
                            fact = f"[{sec_title}] Entity: {source} (modality: {mod})"
                        else:
                            continue

                        if fact not in seen_facts:
                            seen_facts.add(fact)
                            facts.append(fact)

                        if len(facts) >= limit:
                            break
                except Exception as e:
                    logger.warning(
                        "[PageRetriever] Section query failed for %s/%s: %s",
                        document,
                        section_id,
                        e,
                    )

                if len(facts) >= limit:
                    break

            # --- Strategy 2: Page-range-based retrieval ---
            # If we still need more facts, query by page range
            if len(facts) < limit:
                for sec in sections:
                    document = sec.get("document", "")
                    start_page = sec.get("start_index")
                    end_page = sec.get("end_index")
                    title = sec.get("title", "")

                    if start_page is None or end_page is None:
                        continue

                    try:
                        res = session.run(
                            """
                            MATCH (n)
                            WHERE n.document = $document
                              AND n.page IS NOT NULL
                              AND n.page >= $start_page
                              AND n.page <= $end_page
                              AND n.name IS NOT NULL
                            OPTIONAL MATCH (n)-[r]->(m)
                            WHERE m.name IS NOT NULL
                            RETURN n.name AS source_name,
                                   n.page AS page,
                                   type(r) AS rel_type,
                                   m.name AS target_name
                            LIMIT $limit
                            """,
                            document=document,
                            start_page=start_page,
                            end_page=end_page,
                            limit=limit - len(facts),
                        )
                        for row in res:
                            source = row.get("source_name")
                            rel = row.get("rel_type")
                            target = row.get("target_name")
                            page = row.get("page")

                            if source and rel and target:
                                fact = f"[{title} p.{page}] {source} -[{rel}]-> {target}"
                            elif source:
                                fact = f"[{title} p.{page}] Entity: {source}"
                            else:
                                continue

                            if fact not in seen_facts:
                                seen_facts.add(fact)
                                facts.append(fact)

                            if len(facts) >= limit:
                                break
                    except Exception as e:
                        logger.warning(
                            "[PageRetriever] Page-range query failed for %s pp.%s-%s: %s",
                            document,
                            start_page,
                            end_page,
                            e,
                        )

                    if len(facts) >= limit:
                        break

            # --- Strategy 3: Named-entity matching ---
            # If we still have room, try matching question terms to entity names
            if len(facts) < limit:
                try:
                    q_lower = (question or "").lower()
                    # Extract potential entity names from question
                    # Use the existing fallback's named-entity query
                    res = session.run(
                        """
                        MATCH (n)
                        WHERE n.name IS NOT NULL
                          AND size(n.name) > 2
                          AND (
                            toLower($question) CONTAINS toLower(n.name)
                            OR toLower(n.name) CONTAINS toLower($question)
                          )
                        WITH n
                        ORDER BY size(n.name) DESC
                        RETURN n.name AS name,
                               n.section_title AS section_title,
                               n.document AS document,
                               n.evidence AS evidence
                        LIMIT $limit
                        """,
                        question=q_lower,
                        limit=limit - len(facts),
                    )
                    for row in res:
                        name = row.get("name")
                        sec_title = row.get("section_title") or ""
                        doc = row.get("document") or ""
                        evidence = row.get("evidence") or ""

                        if name:
                            fact = f"Entity: {name}"
                            if sec_title:
                                fact = f"[{sec_title}] {fact}"
                            if evidence:
                                fact = f"{fact} — {evidence[:200]}"

                            if fact not in seen_facts:
                                seen_facts.add(fact)
                                facts.append(fact)

                            if len(facts) >= limit:
                                break
                except Exception as e:
                    logger.warning(
                        "[PageRetriever] Named-entity query failed: %s", e
                    )

        # --- Fallback: supplement with standard retrieval if still empty ---
        if not facts:
            logger.info(
                "[PageRetriever] Section-aware queries returned no results. "
                "Falling back to standard GraphRetriever."
            )
            return self._fallback.retrieve(question, limit)

        return facts[:limit]

# ==============================================================
# Quick test
# ==============================================================
if __name__ == "__main__":
    retriever = PageRetriever()

    # Test 1: No sections (should use fallback)
    print("Test 1: No sections (fallback to standard GraphRetriever)")
    facts = retriever.retrieve("What is Hemant's employee ID?", limit=10)
    print(f"  Facts: {len(facts)}")
    for f in facts[:5]:
        print(f"  - {f}")

    # Test 2: With sections (if available)
    print("\nTest 2: With section context")
    test_sections = [
        {
            "document": "hemant_story",
            "section_id": "0001",
            "title": "Employment Details",
            "start_index": 1,
            "end_index": 3,
            "summary": "Hemant's employment information",
        }
    ]
    facts = retriever.retrieve(
        "What is Hemant's salary?",
        limit=10,
        sections=test_sections,
    )
    print(f"  Facts: {len(facts)}")
    for f in facts[:5]:
        print(f"  - {f}")

    retriever.close()
