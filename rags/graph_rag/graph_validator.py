import logging
import re
from typing import Any, Dict, List, Set, Tuple

from rags.graph_rag.graph_schema import (
    VALID_RELATIONSHIP_TYPES as SCHEMA_RELATION_TYPES,
)


logger = logging.getLogger(__name__)


class GraphValidator:
    """Validation layer ensuring JSON consistency before storage.

    Responsibilities:
    - Remove duplicates
    - Ensure relationships reference existing entities
    - Normalize IDs/labels
    - Enforce relationship types allowlist (optional)
    """

    # Must match graph_schema.py relationship type allowlist.
    ALLOWED_RELATIONSHIP_TYPES: Set[str] = SCHEMA_RELATION_TYPES

    def __init__(self, relationship_type_allowlist: Set[str] | None = None):
        self.allowed_types = (
            relationship_type_allowlist or self.ALLOWED_RELATIONSHIP_TYPES
        )

    @staticmethod
    def _norm_id(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", "_", s)
        return s

    @staticmethod
    def _norm_label(s: str) -> str:
        s = (s or "").strip()
        return s or "Entity"

    def validate(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        entities: List[Dict[str, Any]] = extracted.get("entities") or []
        relationships: List[Dict[str, Any]] = extracted.get("relationships") or []

        # Normalize entities + de-dupe by id
        seen_entity_ids: Set[str] = set()
        norm_entities: List[Dict[str, Any]] = []

        for e in entities:
            eid = self._norm_id(e.get("id") or "")
            if not eid:
                continue
            if eid in seen_entity_ids:
                continue
            seen_entity_ids.add(eid)

            props = e.get("properties") or {}
            # Ensure properties has name when possible
            if "name" not in props and e.get("properties", {}).get("name"):
                props["name"] = e.get("properties").get("name")

            norm_entities.append(
                {
                    "id": eid,
                    "label": self._norm_label(e.get("label")),
                    "properties": props,
                }
            )

        # Validate relationships refer to existing entities
        entity_id_set = set(seen_entity_ids)
        seen_rel_keys: Set[Tuple[str, str, str, str]] = set()
        norm_relationships: List[Dict[str, Any]] = []

        for r in relationships:
            r_type = (r.get("type") or "").strip()
            source = self._norm_id(r.get("source") or "")
            target = self._norm_id(r.get("target") or "")
            rid = self._norm_id(r.get("id") or "")
            props = r.get("properties") or {}

            if not r_type or not source or not target:
                continue
            if source not in entity_id_set or target not in entity_id_set:
                continue

            if self.allowed_types and r_type not in self.allowed_types:
                continue

            key = (r_type, source, target, rid)
            if key in seen_rel_keys:
                continue
            seen_rel_keys.add(key)

            # Default confidence to preserve consistent retrieval scoring
            if "confidence" not in props:
                # If absent, allow retrieval but with neutral confidence.
                props["confidence"] = 0.0

            norm_relationships.append(
                {
                    "id": rid or f"{r_type}:{source}->{target}",
                    "type": r_type,
                    "source": source,
                    "target": target,
                    "properties": props,
                }
            )

        return {"entities": norm_entities, "relationships": norm_relationships}

