"""Table Parser — Detect and extract tables from PDFs.

Uses pdfplumber to locate tables on each page and
return structured rows with column headers.

Output format (per table):
{
    "page": int,
    "table_index": int,
    "headers": [str, ...],
    "rows": [
        { "header_1": "value_1", "header_2": "value_2", ... },
        ...
    ],
    "bbox": (x0, y0, x1, y1) | None
}
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber


class TableParser:
    """Extract structured tables from PDF files using pdfplumber."""

    def __init__(self, min_table_words: int = 3):
        """
        Args:
            min_table_words: Minimum words per row to consider valid.
        """
        self.min_table_words = min_table_words

    def extract_tables_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract all tables from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of table dicts with keys: page, table_index, headers, rows, bbox.
        """
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        all_tables: List[Dict[str, Any]] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()

                if not tables:
                    continue

                for table_idx, table in enumerate(tables):
                    parsed = self._parse_table(table, page_idx, table_idx)
                    if parsed and parsed["rows"]:
                        all_tables.append(parsed)

        return all_tables

    def extract_tables_from_page(
        self, page, page_number: int
    ) -> List[Dict[str, Any]]:
        """Extract tables from a single pdfplumber Page object."""
        tables = page.extract_tables()
        if not tables:
            return []

        results = []
        for idx, table in enumerate(tables):
            parsed = self._parse_table(table, page_number, idx)
            if parsed and parsed["rows"]:
                results.append(parsed)

        return results

    def _parse_table(
        self,
        table: List[List[Optional[str]]],
        page_number: int,
        table_index: int,
    ) -> Optional[Dict[str, Any]]:
        """Convert a raw pdfplumber table into structured dict.

        Args:
            table: Raw table data from pdfplumber (list of rows, each row is list of cells).
            page_number: Page number.
            table_index: Table index on page.

        Returns:
            Structured table dict, or None if the table is empty.
        """
        if not table or len(table) < 2:
            return None

        # First row = headers
        raw_headers = table[0]
        headers = [
            (h.strip() if h else f"column_{i}")
            for i, h in enumerate(raw_headers)
        ]

        # Ensure unique headers
        header_counts: Dict[str, int] = {}
        unique_headers: List[str] = []
        for h in headers:
            if not h:
                h = "column"
            if h in header_counts:
                header_counts[h] += 1
                unique_headers.append(f"{h}_{header_counts[h]}")
            else:
                header_counts[h] = 0
                unique_headers.append(h)
        headers = unique_headers

        # Parse data rows
        rows: List[Dict[str, str]] = []
        for row in table[1:]:
            # Skip completely empty rows
            if all(cell is None or not cell.strip() for cell in row):
                continue

            # Skip rows with too few words (likely artifacts)
            word_count = sum(len((cell or "").split()) for cell in row)
            if word_count < self.min_table_words:
                continue

            row_dict = {}
            for i, cell in enumerate(row):
                key = headers[i] if i < len(headers) else f"column_{i}"
                row_dict[key] = (cell or "").strip()

            rows.append(row_dict)

        if not rows:
            return None

        return {
            "page": page_number,
            "table_index": table_index,
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
        }

    @staticmethod
    def table_to_text(table_data: Dict[str, Any]) -> str:
        """Convert structured table data back to readable text.

        Useful for feeding table content into the existing text-based
        graph_extractor.
        """
        lines = [f"[Table on page {table_data['page']}]"]
        lines.append(" | ".join(table_data["headers"]))
        lines.append("-" * len(" | ".join(table_data["headers"])))
        for row in table_data["rows"]:
            vals = [row.get(h, "") for h in table_data["headers"]]
            lines.append(" | ".join(vals))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    parser = TableParser()
    test_pdf = Path("data/hemant_story.pdf")
    if test_pdf.is_file():
        tables = parser.extract_tables_from_pdf(test_pdf)
        print(f"\nFound {len(tables)} table(s) in {test_pdf.name}")
        for t in tables:
            print(json.dumps(t, indent=2, ensure_ascii=False))
    else:
        print(f"No test PDF found at {test_pdf}")

