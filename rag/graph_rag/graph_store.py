import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from rags.graph_rag.graph_schema import (
    VALID_ENTITY_TYPES as SCHEMA_ENTITY_TYPES,
    VALID_RELATIONSHIP_TYPES as SCHEMA_RELATION_TYPES,
)

load_dotenv()

logger = logging.getLogger(__name__)


class GraphStore:

    """Neo4j-only persistence layer.

    Responsibilities
    - Connect/close Neo4j
    - Validate graph JSON shape and whitelists
    - Insert graph via MERGE only (no CREATE)

    Public API (only these methods are meant to be used):
    - connect
    - close
    - insert_graph
    - create_entity
    - create_relationship
    - execute_query

    NOTE: Neo4j driver is now created lazily on connect() / initialize().
          This allows the pipeline to run even when Neo4j is not available.
    """

    VALID_ENTITY_TYPES = SCHEMA_ENTITY_TYPES
    VALID_RELATION_TYPES = SCHEMA_RELATION_TYPES

    def __init__(self):
        self._neo_uri = os.getenv("NEO4J_URI")
        self._neo_user = os.getenv("NEO4J_USERNAME")
        self._neo_pass = os.getenv("NEO4J_PASSWORD")
        self._driver = None
        self._connected = False

    def _create_driver(self):
        """Lazily create the Neo4j driver (not in __init__)."""
        if self._driver is not None:
            return self._driver
        if not self._neo_uri or not self._neo_user or not self._neo_pass:
            raise RuntimeError(
                "Missing Neo4j env vars. Need NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
            )
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(self._neo_uri, auth=(self._neo_user, self._neo_pass))
        return self._driver

    def connect(self) -> None:
        try:
            driver = self._create_driver()
            driver.verify_connectivity()
            self._connected = True
            logger.info("Connected to Neo4j")

            # Optional: minimal schema for id MERGE.
            # We keep it lightweight; if constraints already exist, ignore.
            with driver.session() as session:
                for label in self.VALID_ENTITY_TYPES:
                    session.run(
                        f"""
                        CREATE CONSTRAINT entity_id_unique_{label}
                        IF NOT EXISTS
                        FOR (n:{label})
                        REQUIRE n.id IS UNIQUE
                        """
                    )
        except Exception as e:
            self._connected = False
            logger.warning("Neo4j connection failed: %s", e)
            raise

    def close(self) -> None:
        if self._driver:
            self._driver.close()
        self._driver = None
        self._connected = False

    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self._connected:
            self.connect()

        if self._driver is None:
            logger.warning("Neo4j driver is None — returning empty results")
            return []

        parameters = parameters or {}
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    @staticmethod
    def _validate_entity_payload(entity: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Return (ok, label)."""
        if not isinstance(entity, dict):
            return False, None

        entity_id = entity.get("id")
        label = entity.get("label")
        props = entity.get("properties") or {}
        name = props.get("name")

        if not entity_id or not isinstance(entity_id, str):
            return False, None
        if not label or not isinstance(label, str):
            return False, None
        if label not in SCHEMA_ENTITY_TYPES:
            return False, None
        if not name or not isinstance(name, str):
            # require name for retrieval friendliness
            return False, None

        # Validate ID scheme: Label::...
        if "::" not in entity_id:
            return False, None

        prefix = entity_id.split("::", 1)[0]
        # For safety, ensure the id prefix matches the label.
        if prefix != label:
            return False, None

        return True, label

    @staticmethod
    def _validate_relationship_payload(rel: Dict[str, Any]) -> bool:
        if not isinstance(rel, dict):
            return False

        rel_type = rel.get("type")
        source = rel.get("source")
        target = rel.get("target")

        if not rel_type or not isinstance(rel_type, str):
            return False
        if rel_type not in SCHEMA_RELATION_TYPES:
            return False
        if not source or not target:
            return False
        if "::" not in source or "::" not in target:
            return False
        return True

    def initialize(self) -> None:
        """Connect to Neo4j idempotently."""
        if not self._connected:
            self.connect()

    def upsert_graph(self, graph_json: Dict[str, Any]) -> Dict[str, int]:
        """Upsert graph data (compatibility alias for insert_graph)."""
        return self.insert_graph(graph_json)

    def insert_graph(self, graph_json: Dict[str, Any]) -> Dict[str, int]:


        if not isinstance(graph_json, dict):
            raise TypeError("graph_json must be a dict")

        entities = graph_json.get("entities") or []
        relationships = graph_json.get("relationships") or []

        # Validate entity payloads and collect.
        valid_entities: List[Dict[str, Any]] = []
        entities_skipped = 0
        for e in entities:
            ok, label = self._validate_entity_payload(e)
            if not ok:
                entities_skipped += 1
                continue

            valid_entities.append(
                {
                    "id": e["id"],
                    "label": label,
                    "properties": {
                        **(e.get("properties") or {}),
                    },
                }
            )

        entity_ids = {e["id"] for e in valid_entities}

        # Validate relationships payloads and referential integrity.
        valid_relationships: List[Dict[str, Any]] = []
        rel_skipped = 0
        for r in relationships:
            if not self._validate_relationship_payload(r):
                rel_skipped += 1
                continue

            if r.get("source") not in entity_ids or r.get("target") not in entity_ids:
                rel_skipped += 1
                continue

            valid_relationships.append(
                {
                    "type": r["type"],
                    "source": r["source"],
                    "target": r["target"],
                    "properties": {
                        **(r.get("properties") or {}),
                    },
                }
            )

        # Batch insertion via UNWIND.
        # Entities
        entities_created = 0
        if valid_entities:
            # MERGE with dynamic label using apoc not guaranteed; so we split by label.
            by_label: Dict[str, List[Dict[str, Any]]] = {}
            for e in valid_entities:
                by_label.setdefault(e["label"], []).append(e)

            for label, rows in by_label.items():
                self.execute_query(
                    f"""
                    UNWIND $rows AS row
                    MERGE (e:{label} {{id: row.id}})
                    SET e += row.properties
                    """,
                    {"rows": rows},
                )
                entities_created += len(rows)

        # Relationships
        relationships_created = 0
        if valid_relationships:
            # Relationship types require interpolation; split by type.
            by_type: Dict[str, List[Dict[str, Any]]] = {}
            for r in valid_relationships:
                by_type.setdefault(r["type"], []).append(r)

            for rel_type, rows in by_type.items():
                # MERGE based on (source_id, target_id, type). Neo4j cannot MERGE on rel type without interpolation.
                self.execute_query(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a {{id: row.source}})
                    MATCH (b {{id: row.target}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r += row.properties
                    """,
                    {"rows": rows},
                )
                relationships_created += len(rows)

        return {
            "entities_created": entities_created,
            "entities_skipped": entities_skipped,
            "relationships_created": relationships_created,
            "relationships_skipped": rel_skipped,
        }

    def create_entity(self, entity: Dict[str, Any]) -> bool:
        graph_json = {"entities": [entity], "relationships": []}
        stats = self.insert_graph(graph_json)
        return stats.get("entities_created", 0) == 1

    def create_relationship(self, rel: Dict[str, Any]) -> bool:
        graph_json = {"entities": [], "relationships": [rel]}
        stats = self.insert_graph(graph_json)
        return stats.get("relationships_created", 0) == 1

        