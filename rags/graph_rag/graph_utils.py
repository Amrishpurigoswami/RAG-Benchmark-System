from pathlib import Path
from hashlib import md5
from typing import Any, Dict, List, Optional
from pypdf import PdfReader
import re


class GraphUtils:

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""

        text = text.replace("\x00", " ")

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text)

        text = re.sub(r"\s+\n", "\n", text)

        return text.strip()

    @staticmethod
    def remove_duplicates(chunks):
        seen = set()
        unique = []
        for chunk in chunks:
            h = md5(chunk.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(chunk)
        return unique

    def process_pdf(
        self,
        pdf_path: Path,
        chunk_size: int = 400,
        overlap: int = 80,
        profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Read a PDF and return a list of text chunks with metadata.

        Each chunk dict contains:
            page    : int   — 1-based page number
            chunk   : int   — 0-based chunk index within the page
            heading : str   — detected heading line (first non-empty line)
            text    : str   — cleaned chunk text
        """
        reader = PdfReader(pdf_path)
        heading_pattern = re.compile(
            r"^(?:(?:\d+\.)+\d*\s+)?[A-Z][A-Za-z0-9 &\-:,/]{2,80}$"
        )

        all_chunks: List[Dict[str, Any]] = []

        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            text = self.clean_text(raw)
            if not text:
                continue

            # Detect heading: first non-empty line that looks like a title
            lines = text.splitlines()
            heading = ""
            for line in lines:
                line = line.strip()
                if line and heading_pattern.match(line):
                    heading = line
                    break

            # Split page text into overlapping word-level chunks
            words = text.split()
            if not words:
                continue

            chunk_idx = 0
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)

                all_chunks.append(
                    {
                        "page": page_num,
                        "chunk": chunk_idx,
                        "heading": heading,
                        "text": chunk_text,
                    }
                )

                chunk_idx += 1
                start += chunk_size - overlap

        return all_chunks