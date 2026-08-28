"""Streamlit chat and graph explorer for the RAG Benchmark System.

Run from the repository root:
    streamlit run outputs/chat/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rags.graph_rag.graph_rag import GraphQuery
from rags.graph_rag.graph_retriever import GraphRetriever


st.set_page_config(
    page_title="RAG Benchmark Chat",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def graph_query() -> GraphQuery:
    return GraphQuery()


def run_cypher(query: str, **parameters: Any) -> list[dict[str, Any]]:
    """Run a read-only query through the project's configured Neo4j driver."""
    retriever = GraphRetriever()
    try:
        with retriever._driver.session() as session:
            return [record.data() for record in session.run(query, **parameters)]
    finally:
        retriever.close()


def database_metrics() -> dict[str, int]:
    rows = run_cypher(
        """
        MATCH (n)
        WITH count(n) AS nodes
        MATCH ()-[r]->()
        RETURN nodes, count(r) AS relationships
        """
    )
    return rows[0] if rows else {"nodes": 0, "relationships": 0}


def source_files() -> list[str]:
    rows = run_cypher(
        """
        MATCH ()-[r]->()
        WHERE r.source_pdf IS NOT NULL
        RETURN DISTINCT r.source_pdf AS source_pdf
        ORDER BY source_pdf
        """
    )
    return [row["source_pdf"] for row in rows if row.get("source_pdf")]


def relationship_rows(source_pdf: str | None) -> list[dict[str, Any]]:
    return run_cypher(
        """
        MATCH (a)-[r]->(b)
        WHERE ($source_pdf = '' OR r.source_pdf = $source_pdf)
        RETURN coalesce(a.name, elementId(a)) AS source,
               type(r) AS relationship,
               coalesce(b.name, elementId(b)) AS target,
               r.source_pdf AS source_pdf,
               r.evidence AS evidence,
               r.fact AS fact
        ORDER BY source, relationship, target
        LIMIT 200
        """,
        source_pdf=source_pdf or "",
    )


def dot_graph(rows: list[dict[str, Any]]) -> str:
    """Create a compact Graphviz view without exposing raw Neo4j IDs."""
    lines = ["digraph G {", "rankdir=LR;", "node [shape=box style=rounded];"]
    seen_nodes: set[str] = set()
    for row in rows[:40]:
        source = str(row["source"]).replace('"', "'")
        target = str(row["target"]).replace('"', "'")
        relation = str(row["relationship"]).replace('"', "'")
        for node in (source, target):
            if node not in seen_nodes:
                lines.append(f'"{node}";')
                seen_nodes.add(node)
        lines.append(f'"{source}" -> "{target}" [label="{relation}"];')
    lines.append("}")
    return "\n".join(lines)


st.title("🧠 RAG Benchmark System")
st.caption("Ask grounded questions, inspect retrieved facts, and explore the Neo4j knowledge graph.")

try:
    metrics = database_metrics()
    sources = source_files()
except Exception as error:
    st.error(f"Neo4j is unavailable: {error}")
    st.stop()

with st.sidebar:
    st.header("Project overview")
    st.metric("Knowledge-graph nodes", metrics["nodes"])
    st.metric("Knowledge-graph relationships", metrics["relationships"])
    st.metric("Source PDFs", len(sources))
    st.divider()
    st.markdown(
        """
        **Graph RAG pipeline**

        PDF → Profile → Chunks → Extraction → Validation → Neo4j → Retrieval → Cerebras answer

        The app always shows the retrieved facts used to ground an answer.
        """
    )

chat_tab, graph_tab, architecture_tab = st.tabs(
    ["💬 Chat", "🕸️ Graph explorer", "🏗️ Architecture"]
)

with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("facts"):
                with st.expander("Retrieved graph facts"):
                    for fact in message["facts"]:
                        st.markdown(f"- {fact}")

    prompt = st.chat_input("Ask about Hemant, policies, salary, leave, or the knowledge graph…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving graph facts and generating a grounded answer…"):
                try:
                    result = graph_query().query(prompt)
                    answer = result["answer"]
                    facts = result.get("facts", [])
                except Exception as error:
                    answer = f"I could not complete the graph query: {error}"
                    facts = []

            st.markdown(answer)
            if facts:
                with st.expander("Retrieved graph facts", expanded=True):
                    for fact in facts:
                        st.markdown(f"- {fact}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "facts": facts}
        )

with graph_tab:
    selected_source = st.selectbox(
        "Filter relationships by source PDF",
        options=["All sources", *sources],
    )
    selected_source = "" if selected_source == "All sources" else selected_source

    try:
        rows = relationship_rows(selected_source)
    except Exception as error:
        st.error(f"Could not load graph relationships: {error}")
        rows = []

    st.caption(f"Showing {len(rows)} relationships")
    if rows:
        st.graphviz_chart(dot_graph(rows), use_container_width=True)
        table = pd.DataFrame(rows)
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No relationships match this filter.")

with architecture_tab:
    st.graphviz_chart(
        """
        digraph pipeline {
          rankdir=LR;
          node [shape=box style=rounded];
          PDF -> Profile -> Chunking -> Extraction -> Validation -> Neo4j;
          Neo4j -> Retrieval -> Cerebras -> Answer;
        }
        """,
        use_container_width=True,
    )
    st.markdown(
        """
        - **OpenRouter**: document profiling and structured graph extraction.
        - **Neo4j**: stores source-linked entities and semantic relationships.
        - **Cerebras**: synthesizes an answer strictly from retrieved graph facts.
        - **Fallback**: if synthesis fails, the application returns the retrieved fact rather than claiming no answer exists.
        """
    )
