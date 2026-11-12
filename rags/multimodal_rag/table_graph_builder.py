"""Table-to-Graph Builder — Convert structured table rows into graph JSON.

Takes output from TableParser and produces entities + relationships
that are compatible with graph_validator.py and graph_store.py.

Heuristics:
• Each row becomes an entity (type derived from context or "TableRow").
• Column values can become either attributes or linked entities.
• Relationships connect row entities to column value entities.

Supports common table patterns:
  - Employee | Salary | Manager → Employee HAS_SALARY, Employee REPORTS_TO Manager
  - Policy | Clause | Description → Policy HAS_CLAUSE Clause
  - Attendance | Date | Status → Attendance HAS_DATE, Attendance HAS_STATUS
"""

import re
from typing import Any, Dict, List, Optional, Set

from rags.graph_rag.graph_store import GraphStore


class TableGraphBuilder:
    """Convert table data into graph JSON for Neo4j storage."""

    # Entity types that should be created as separate nodes (not attributes)
    NODE_COLUMN_PATTERNS: Dict[str, str] = {
        "employee": "Employee",
        "manager": "Manager",
        "department": "Department",
        "organization": "Organization",
        "policy": "Policy",
        "leave": "Leave",
        "attendance": "Attendance",
        "kpi": "KPI",
        "project": "Project",
        "team": "Team",
        "designation": "Designation",
        "role": "Designation",
    }

    # Relationship type inference based on column name pairs
    RELATIONSHIP_PATTERNS: Dict[str, str] = {
        "manager": "REPORTS_TO",
        "reports_to": "REPORTS_TO",
        "reporting_manager": "REPORTS_TO",
        "supervisor": "REPORTS_TO",
        "salary": "HAS_SALARY",
        "ctc": "HAS_SALARY",
        "monthly_ctc": "HAS_SALARY",
        "department": "BELONGS_TO",
        "organization": "WORKS_AT",
        "policy": "HAS_POLICY",
        "leave_type": "HAS_LEAVE_TYPE",
        "leave_balance": "HAS_LEAVE_BALANCE",
        "status": "HAS_STATUS",
    }

    def __init__(self, source_pdf: str = ""):
        self.source_pdf = source_pdf

    def build_graph(
        self,
        table_data: Dict[str, Any],
        entity_type: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Convert a single table into graph JSON.

        Args:
            table_data: Table dict from TableParser (page, headers, rows, etc.).
            entity_type: Optional override for the primary entity type.

        Returns:
            Graph JSON with "entities" and "relationships" lists.
        """
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []

        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        page = table_data.get("page", 0)
        table_idx = table_data.get("table_index", 0)

        if not headers or not rows:
            return {"entities": entities, "relationships": relationships}

        # Infer primary entity type from headers or use provided type
        primary_type = entity_type or self._infer_entity_type(headers)

        seen_entity_ids: Set[str] = set()

        for row_idx, row in enumerate(rows):
            # --- Create primary row entity ---
            row_name = self._extract_row_name(row, headers, primary_type)
            row_id = f"{primary_type}::{row_name}"
            row_entity = {
                "id": row_id,
                "label": primary_type,
                "properties": {
                    "name": row_name,
                    "page": page,
                    "table_index": table_idx,
                    "row_index": row_idx,
                    "source_pdf": self.source_pdf,
                    "evidence": self._row_to_text(row, headers),
                },
            }
            # Add all column values as attributes
            for h in headers:
                val = row.get(h, "")
                if val:
                    row_entity["properties"][self._safe_attr_name(h)] = val

            if row_id not in seen_entity_ids:
                entities.append(row_entity)
                seen_entity_ids.add(row_id)

            # --- Process each column ---
            for h in headers:
                val = row.get(h, "")
                if not val:
                    continue

                col_lower = h.lower().strip()
                val_lower = val.lower().strip()

                # Check if this column should become a separate node
                node_type = self._match_node_pattern(col_lower)

                # Skip the primary identifier column (avoid self-referencing)
                if self._is_identifier_column(col_lower, primary_type):
                    continue

                if node_type:
                    # Create a linked entity node
                    node_name = val
                    node_id = f"{node_type}::{node_name}"

                    if node_id not in seen_entity_ids:
                        node_entity = {
                            "id": node_id,
                            "label": node_type,
                            "properties": {
                                "name": node_name,
                                "page": page,
                                "source_pdf": self.source_pdf,
                                "evidence": val,
                            },
                        }
                        # Add extra attributes from row for context
                        for extra_h in headers:
                            extra_val = row.get(extra_h, "")
                            if extra_val and extra_h.lower() != h.lower():
                                node_entity["properties"][
                                    self._safe_attr_name(extra_h)
                                ] = extra_val

                        entities.append(node_entity)
                        seen_entity_ids.add(node_id)

                    # Determine relationship type
                    rel_type = self._infer_relationship_type(
                        col_lower, primary_type, node_type
                    )

                    relationships.append({
                        "type": rel_type,
                        "source": row_id,
                        "target": node_id,
                        "properties": {
                            "page": page,
                            "table_index": table_idx,
                            "evidence": val,
                        },
                    })

                # Even for attribute columns, check if there's a numeric value
                # that should be linked via a relationship
                elif self._is_numeric_relationship(col_lower):
                    # Create a value node for numeric data
                    value_type = f"{primary_type}Value"
                    value_name = f"{h}_{val}"
                    value_id = f"{value_type}::{value_name}"

                    if value_id not in seen_entity_ids:
                        value_entity = {
                            "id": value_id,
                            "label": value_type,
                            "properties": {
                                "name": value_name,
                                "value": val,
                                "column": h,
                                "page": page,
                                "source_pdf": self.source_pdf,
                            },
                        }
                        entities.append(value_entity)
                        seen_entity_ids.add(value_id)

                    rel_type = self.RELATIONSHIP_PATTERNS.get(
                        col_lower, f"HAS_{self._safe_attr_name(h).upper()}"
                    )
                    relationships.append({
                        "type": rel_type,
                        "source": row_id,
                        "target": value_id,
                        "properties": {
                            "page": page,
                            "evidence": val,
                        },
                    })

        return {"entities": entities, "relationships": relationships}

    def _infer_entity_type(self, headers: List[str]) -> str:
        """Guess the primary entity type from column headers."""
        header_text = " ".join(h.lower() for h in headers)

        for keyword, etype in [
            ("employee", "Employee"),
            ("manager", "Manager"),
            ("kpi", "KPI"),
            ("policy", "Policy"),
            ("attendance", "Attendance"),
            ("leave", "Leave"),
            ("department", "Department"),
            ("project", "Project"),
        ]:
            if keyword in header_text:
                return etype

        return "TableRow"

    def _extract_row_name(
        self, row: Dict[str, str], headers: List[str], primary_type: str
    ) -> str:
        """Extract a human-readable name for this row."""
        # Prefer name/id columns
        for keyword in ["name", "id", "employee", "employee_id", "code", "title"]:
            for h in headers:
                if keyword in h.lower():
                    val = row.get(h, "").strip()
                    if val:
                        return val

        # Use first non-empty column
        for h in headers:
            val = row.get(h, "").strip()
            if val:
                return val

        return f"{primary_type}_{row}"

    def _match_node_pattern(self, col_lower: str) -> Optional[str]:
        """Check if a column name matches a known node pattern."""
        for pattern, node_type in self.NODE_COLUMN_PATTERNS.items():
            if pattern in col_lower:
                return node_type
        return None

    def _is_identifier_column(self, col_lower: str, primary_type: str) -> bool:
        """Check if a column is the primary identifier (e.g., 'employee' for Employee)."""
        primary_lower = primary_type.lower()
        if primary_lower in col_lower:
            # Only skip if it's exactly the type name or "name" variant
            if col_lower == primary_lower or "name" in col_lower or "id" in col_lower:
                return True
        return False

    def _is_numeric_relationship(self, col_lower: str) -> bool:
        """Check if a column typically has numeric values that become relationships."""
        numeric_keywords = [
            "salary", "ctc", "amount", "balance", "count", "percentage",
            "score", "value", "rating", "budget", "revenue", "profit",
        ]
        return any(kw in col_lower for kw in numeric_keywords)

    def _infer_relationship_type(
        self, col_lower: str, source_type: str, target_type: str
    ) -> str:
        """Determine the relationship type between source and target."""
        # Check known patterns first
        for pattern, rel_type in self.RELATIONSHIP_PATTERNS.items():
            if pattern in col_lower:
                return rel_type

        # Fall back to type-based relationship
        if target_type == "Manager":
            return "REPORTS_TO"
        if target_type == "Department":
            return "BELONGS_TO"
        if target_type == "Organization":
            return "WORKS_AT"
        if target_type == "Policy":
            return "HAS_POLICY"
        if target_type == "Salary":
            return "HAS_SALARY"

        return f"HAS_{target_type.upper()}"

    @staticmethod
    def _safe_attr_name(name: str) -> str:
        """Convert a column header to a safe attribute name."""
        name = name.lower().strip()
        name = re.sub(r"[^a-z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")

    @staticmethod
    def _row_to_text(row: Dict[str, str], headers: List[str]) -> str:
        """Convert a row to evidence text."""
        parts = [f"{h}: {row.get(h, '')}" for h in headers]
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_table = {
        "page": 1,
        "table_index": 0,
        "headers": ["Employee", "Salary", "Manager", "Department"],
        "rows": [
            {"Employee": "Hemant Sharma", "Salary": "72000", "Manager": "Rakesh Malhotra", "Department": "Engineering"},
            {"Employee": "Priya Singh", "Salary": "85000", "Manager": "Rakesh Malhotra", "Department": "Engineering"},
        ],
    }

    builder = TableGraphBuilder(source_pdf="test.pdf")
    graph = builder.build_graph(sample_table, entity_type="Employee")
    print(f"Entities: {len(graph['entities'])}")
    print(f"Relationships: {len(graph['relationships'])}")
    for e in graph["entities"]:
        print(f"  Entity: {e['id']} ({e['label']})")
    for r in graph["relationships"]:
        print(f"  Rel: {r['source']} -[{r['type']}]-> {r['target']}")

