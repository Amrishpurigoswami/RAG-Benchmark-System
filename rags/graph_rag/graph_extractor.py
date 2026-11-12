import json
import logging
from typing import Any, Dict, List

from rags.graph_rag.graph_schema import (
    VALID_ENTITY_TYPES,
    VALID_RELATIONSHIP_TYPES,
)
from rags.graph_rag.llm_config import get_construction_client, get_construction_models

logger = logging.getLogger(__name__)


class GraphExtractor:
    """LLM extraction layer (NO DB ops).

    Output (strict JSON):
      {
        "entities": [
          {
            "id": "Employee::...",
            "type": "Employee",
            "name": "...",
            "attributes": { ... }
          }
        ],
        "relationships": [
          {
            "type": "REPORTS_TO",
            "source": "Employee::...",
            "target": "Manager::...",
            "properties": { ... }
          }
        ]
      }

    Notes
    - Only emits whitelisted entity/relationship types.
    - Scalar values (money, ids, dates, clause numbers, etc.) become entity.attributes.
    - Relationship IDs are not required (store MERGEs by endpoints + type).
    """

    VALID_ENTITY_TYPES = VALID_ENTITY_TYPES
    VALID_RELATION_TYPES = VALID_RELATIONSHIP_TYPES

    def __init__(self):
        self.client = get_construction_client()
        self.model_fallbacks = get_construction_models()
        print(f"[GraphExtractor] Provider: {self.client.base_url}")
        print(f"[GraphExtractor] Fallback models: {self.model_fallbacks}")

    def build_prompt(self, extraction_instructions: str, chunk: Dict[str, Any]) -> str:
        return f"""
{extraction_instructions}

You are building a document-grounded Knowledge Graph.

CRITICAL RULES
1) Output ONLY valid JSON. No markdown. No commentary.
2) Only emit entities of type in this list:
   {sorted(self.VALID_ENTITY_TYPES)}
3) Only emit relationships of type in this list:
   {sorted(self.VALID_RELATION_TYPES)}
4) Stable IDs must use this exact format:
   <Type>::<Human-readable name or key>
   Examples:
     Employee::Hemant Sharma
     Manager::Rakesh Malhotra
     Organization::NexGen Tech Solutions Pvt. Ltd.
     Clause::6.2
     Policy::Leave Policy
5) Do NOT create nodes for scalar values.
   Convert these into attributes instead:
     - Employee ID strings (becomes Employee.attributes.employee_id)
     - Salary/CTC/bonus amounts (becomes Employee.attributes.monthly_ctc, take_home_salary, etc.)
     - Dates (becomes appropriate entity attribute)
     - Clause numbers like 6.2 (create Clause node ONLY if it is part of policy/leave rules; otherwise use as attribute)
     - Monetary strings (Rs. 12,300 / 61,500) -> attributes.amount fields
6) Relationships must connect entity IDs (source/target) and must reference entities present in THIS JSON.
7) Evidence grounding: if you create a relationship or entity, include an "evidence" snippet in properties (and/or attribute evidence if helpful).
8) If the chunk doesn’t contain enough information to create a high-quality entity/edge, return empty lists.

CHUNK CONTEXT
- page: {chunk.get('page')}
- chunk: {chunk.get('chunk')}
- heading: {chunk.get('heading')}

CHUNK TEXT
{chunk.get('text','')}

OUTPUT JSON SCHEMA (STRICT)
{{
  "entities": [
    {{
      "id": "<Type>::<key>",
      "type": "<Type>",
      "name": "<human name>",
      "attributes": {{
        "document": "<source pdf name if available>",
        "page": <int>,
        "chunk": <int>,
        "source_pdf": "<source pdf name if available>",
        "evidence": "<verbatim snippet>",
        "...": "additional extracted attributes"
      }}
    }}
  ],
  "relationships": [
    {{
      "type": "<RELATION_TYPE>",
      "source": "<entity-id-from-entities>",
      "target": "<entity-id-from-entities>",
      "properties": {{
        "page": <int>,
        "chunk": <int>,
        "evidence": "<verbatim snippet>"
      }}
    }}
  ]
}}
""".strip()

    def _safe_json_parse(self, content: str) -> Dict[str, Any]:
        content = (content or "").strip()
        if content.startswith("```"):
            content = content.split("```", 1)[1]
            if "```" in content:
                content = content.rsplit("```", 1)[0]

        content = content.strip()
        first = content.find("{")
        last = content.rfind("}")
        if first != -1 and last != -1:
            content = content[first : last + 1]

        return json.loads(content)

    def extract(self, extraction_instructions: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.build_prompt(extraction_instructions, chunk)

        last_err: Exception | None = None
        data: Dict[str, Any] | None = None

        # Try models in order (primary -> secondary -> tertiary -> quaternary)
        for model in self.model_fallbacks:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content or ""
                data = self._safe_json_parse(content)
                break
            except Exception as e:
                last_err = e
                continue

        if data is None:
            raise last_err or RuntimeError("Graph extraction failed; no model produced output")

        entities = data.get("entities") or []
        relationships = data.get("relationships") or []

        # Normalize to our expected keys
        norm_entities: List[Dict[str, Any]] = []
        for e in entities:
            if not isinstance(e, dict):
                continue
            etype = e.get("type")
            eid = e.get("id")
            name = e.get("name")
            attrs = e.get("attributes") or {}
            if not eid or not etype:
                continue

            # Convert type->label expected by validator/store
            norm_entities.append(
                {
                    "id": str(eid),
                    "label": str(etype),
                    "properties": {**attrs, "name": name},
                }
            )

        norm_relationships: List[Dict[str, Any]] = []
        for r in relationships:
            if not isinstance(r, dict):
                continue
            rtype = r.get("type")
            source = r.get("source")
            target = r.get("target")
            props = r.get("properties") or {}
            if not rtype or not source or not target:
                continue

            norm_relationships.append(
                {
                    "type": str(rtype),
                    "source": str(source),
                    "target": str(target),
                    "properties": props,
                }
            )

        return {"entities": norm_entities, "relationships": norm_relationships}

