"""Single source of truth for Graph RAG entity and relationship type schemas.

All modules (extractor, validator, store, retriever) MUST import from here
instead of maintaining their own copies.
"""

from typing import Final, Set

# ======================================================================
# Entity Types
# ======================================================================
VALID_ENTITY_TYPES: Final[Set[str]] = {
    "Employee",
    "Manager",
    "Organization",
    "Department",
    "Designation",
    "Salary",
    "Policy",
    "Clause",
    "LeaveRequest",
    "Leave",
    "MedicalRecord",
    "Document",
    "TableRow",
    "Image",
    "Attendance",
    "KPI",
    "Project",
    "Team",
}

# ======================================================================
# Relationship Types
# ======================================================================
VALID_RELATIONSHIP_TYPES: Final[Set[str]] = {
    "WORKS_AT",
    "REPORTS_TO",
    "MANAGES",
    "BELONGS_TO",
    "LOCATED_AT",
    "HAS_EMPLOYEE_ID",
    "HAS_SALARY",
    "HAS_POLICY",
    "HAS_CLAUSE",
    "HAS_LEAVE_TYPE",
    "HAS_LEAVE_BALANCE",
    "HAS_STATUS",
    "APPROVED_BY",
    "REQUESTED",
    "HAS_MEDICAL_RECORD",
}

# ======================================================================
# Validation helpers
# ======================================================================
ENTITY_ID_SEPARATOR: Final[str] = "::"


def is_valid_entity_id(entity_id: str) -> bool:
    """Check that an entity ID follows the '<Type>::<key>' pattern
    and uses a known entity type."""
    if not entity_id or not isinstance(entity_id, str):
        return False
    if ENTITY_ID_SEPARATOR not in entity_id:
        return False
    prefix = entity_id.split(ENTITY_ID_SEPARATOR, 1)[0]
    return prefix in VALID_ENTITY_TYPES


def normalize_label(label: str) -> str:
    """Normalize an entity label to a valid type, defaulting to 'Entity'."""
    label = (label or "").strip()
    if label in VALID_ENTITY_TYPES:
        return label
    # Attempt case-insensitive match
    for valid in VALID_ENTITY_TYPES:
        if label.lower() == valid.lower():
            return valid
    return label if label else "Entity"

