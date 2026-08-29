import logging
import os
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Neo4j-only retrieval.

    Synchronous reads using Neo4j GraphDatabase driver.

    NOTE: Neo4j driver is created lazily on first retrieve() call.
          If Neo4j is unavailable, retrieve() returns an empty list
          instead of crashing.
    """

    def __init__(self):
        self._neo_uri = os.getenv("NEO4J_URI")
        self._neo_user = os.getenv("NEO4J_USERNAME")
        self._neo_pass = os.getenv("NEO4J_PASSWORD")
        self._driver = None

    def _ensure_driver(self) -> bool:
        """Lazily create and verify Neo4j driver. Returns True if connected."""
        if self._driver is not None:
            return True
        if not self._neo_uri or not self._neo_user or not self._neo_pass:
            logger.warning("Missing Neo4j env vars — retrieval disabled")
            return False
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self._neo_uri, auth=(self._neo_user, self._neo_pass))
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning("Neo4j unavailable (%s) — retrieval will return empty results", e)
            self._driver = None
            return False

    def close(self):
        if self._driver:
            self._driver.close()
        self._driver = None

    def retrieve(self, question: str, limit: int = 40) -> List[str]:
        try:
            q = (question or "").lower()
            if not q.strip():
                return []
        except Exception:
            return []

        if not self._ensure_driver():
            return []

        anchor_name = "Hemant Sharma"

        facts: List[str] = []

        with self._driver.session() as session:
            # 1) employee id
            if "employee" in q and ("id" in q or "employee id" in q):
                res = session.run(
                    """
                    MATCH (e:Employee {name:$name})
                    RETURN e.employee_id AS employee_id
                    LIMIT 1
                    """,
                    name=anchor_name,
                )
                row = res.single()
                if row and row.get("employee_id"):
                    facts.append(f"Employee ID: {row['employee_id']}")
                return facts[:limit]

            # 2) reporting manager
            if "reporting manager" in q or ("reports" in q and "manager" in q):
                res = session.run(
                    """
                    MATCH (:Employee {name:$name})-[:REPORTS_TO]->(m:Manager)
                    RETURN m.name AS manager_name
                    LIMIT 5
                    """,
                    name=anchor_name,
                )
                for row in res:
                    if row.get("manager_name"):
                        facts.append(f"Reporting manager: {row['manager_name']}")
                return facts[:limit]

            # 3) monthly salary / monthly ctc
            if "monthly" in q and ("salary" in q or "ctc" in q):
                res = session.run(
                    """
                    MATCH (e:Employee {name:$name})
                    RETURN e.monthly_ctc AS monthly_ctc
                    LIMIT 1
                    """,
                    name=anchor_name,
                )
                row = res.single()
                if row:
                    val = row.get("monthly_ctc")
                    if val:
                        facts.append(f"Monthly salary/CTC: {val}")
                return facts[:limit]

            # 4) bonus withheld
            if "bonus" in q and ("withheld" in q or "with hold" in q or "withhold" in q):
                res = session.run(
                    """
                    MATCH (e:Employee {name:$name})-[r]->(x)
                    WHERE type(r) IN ['HAS_SALARY','APPROVED_BY','REQUESTED']
                    RETURN r.evidence AS evidence
                    LIMIT 25
                    """,
                    name=anchor_name,
                )
                for row in res:
                    evidence = row.get("evidence")
                    if evidence and "bonus" in evidence.lower() and (
                        "withheld" in evidence.lower() or "withhold" in evidence.lower()
                    ):
                        facts.append(f"Bonus withheld: {evidence}")

                if not facts:
                    return facts[:limit]

                try:
                    resolution = session.run(
                        """
                        MATCH (:Employee {name:$name})-[:REQUESTED]->(ber:Document)
                        OPTIONAL MATCH (ber)-[:HAS_CLAUSE]->(clause:Clause)
                        OPTIONAL MATCH (ber)-[:APPROVED_BY]->(hr:Department)
                        RETURN ber.name AS request_name,
                               ber.amount AS amount,
                               ber.approval_status AS approval_status,
                               ber.credit_date AS credit_date,
                               clause.name AS clause_name,
                               hr.name AS approver
                        LIMIT 1
                        """,
                        name=anchor_name,
                    ).single()
                    if resolution and resolution.get("request_name"):
                        parts = [f"Resolution: {resolution['request_name']}"]
                        if resolution.get("clause_name"):
                            parts.append(f"under Clause {resolution['clause_name']}")
                        if resolution.get("approval_status"):
                            parts.append(f"was {resolution['approval_status']}")
                        if resolution.get("approver"):
                            parts.append(f"by {resolution['approver']}")
                        if resolution.get("amount"):
                            parts.append(f"for {resolution['amount']}")
                        if resolution.get("credit_date"):
                            parts.append(f"and credited on {resolution['credit_date']}")
                        facts.append(" ".join(parts) + ".")
                except Exception:
                    pass

                return facts[:limit]

            # 5) Named-entity overview
            try:
                entity_rows = session.run(
                    """
                    MATCH (n)
                    WHERE n.name IS NOT NULL
                      AND size(n.name) > 3
                      AND toLower($question) CONTAINS toLower(n.name)
                    WITH n
                    ORDER BY CASE WHEN n.summary IS NOT NULL THEN 0 ELSE 1 END,
                             size(coalesce(n.summary, n.evidence, '')) DESC
                    RETURN n.name AS name, n.summary AS summary, n.evidence AS evidence
                    LIMIT 5
                    """,
                    question=question,
                )
                seen_facts: set[str] = set()
                for row in entity_rows:
                    summary = (row.get("summary") or "").strip()
                    evidence = (row.get("evidence") or "").strip()
                    source_text = summary or evidence
                    for item in source_text.splitlines():
                        fact = item.strip()
                        if fact and fact not in seen_facts:
                            facts.append(fact)
                            seen_facts.add(fact)
                        if len(facts) >= limit:
                            break
                    if len(facts) >= limit:
                        break
                if facts:
                    return facts[:limit]
            except Exception:
                pass

        return facts[:limit]