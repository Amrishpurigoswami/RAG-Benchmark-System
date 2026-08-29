import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dotenv import load_dotenv
from pypdf import PdfReader

from rags.graph_rag.graph_profile import GraphProfile
from rags.graph_rag.graph_utils import GraphUtils
from rags.graph_rag.graph_extractor import GraphExtractor
from rags.graph_rag.graph_validator import GraphValidator
from rags.graph_rag.graph_store import GraphStore

load_dotenv()

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Orchestrator ONLY: PDF -> Profile -> Chunk -> Extract -> Validate -> Store."""

    def __init__(self, pdf_paths: Optional[Iterable[Path]] = None):
        self.data_folder = Path("data")
        self.pdf_paths = [Path(path) for path in pdf_paths] if pdf_paths else None
        self.utils = GraphUtils()
        self.profile_builder = GraphProfile()
        self.extractor = GraphExtractor()
        self.validator = GraphValidator()
        self.store = GraphStore()

    def _load_extraction_prompt_for_pdf(self, pdf_path: Path) -> str:
        stem = pdf_path.stem
        prompt_file = Path("profiles") / f"{stem}_extraction_prompt.txt"
        if not prompt_file.exists():
            return ""
        return prompt_file.read_text(encoding="utf-8")

    def _pdf_text_preview_for_profile(self, pdf_path: Path, max_chars: int = 120000) -> str:
        reader = PdfReader(pdf_path)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
            if sum(len(p) for p in parts) > max_chars:
                break
        return "\n".join(parts)[:max_chars]

    def build(self):
        self.store.initialize()

        pdfs = self.pdf_paths or list(self.data_folder.glob("*.pdf"))
        if not pdfs:
            raise RuntimeError("No PDF found in ./data")

        missing_pdfs = [pdf for pdf in pdfs if not pdf.is_file()]
        if missing_pdfs:
            raise FileNotFoundError(f"PDF not found: {missing_pdfs[0]}")

        for pdf in pdfs:
            logger.info("PDF: %s", pdf.name)

            # 1) Profile (one-time per PDF)
            profile_json_path = Path("profiles") / f"{pdf.stem}_profile.json"
            if not profile_json_path.exists():
                document_text = self._pdf_text_preview_for_profile(pdf)
                self.profile_builder.build_profile(pdf.name, document_text)

            profile: Optional[Dict[str, Any]] = None
            if profile_json_path.exists():
                profile = json.loads(profile_json_path.read_text(encoding="utf-8"))

            extraction_prompt = self._load_extraction_prompt_for_pdf(pdf)
            if not extraction_prompt:
                raise RuntimeError(f"Missing extraction prompt for {pdf.name}")

            # 2) Chunk
            chunks = self.utils.process_pdf(pdf, profile=profile)
            logger.info("Chunks created: %d", len(chunks))

            # 3-5) Extract -> Validate -> Store
            total_graph_payloads = 0
            for idx, chunk in enumerate(chunks, start=1):
                text = (chunk.get("text") or "").strip()
                if len(text.split()) < 25:
                    continue

                logger.debug(
                    "[%s] (%d/%d) page=%s chunk=%s heading=%r",
                    pdf.name, idx, len(chunks),
                    chunk.get('page'), chunk.get('chunk'), chunk.get('heading'),
                )

                extracted = self.extractor.extract(extraction_prompt, chunk)
                validated = self.validator.validate(extracted)

                graph_payload = {
                    "entities": validated.get("entities") or [],
                    "relationships": validated.get("relationships") or [],
                }

                self.store.upsert_graph(graph_payload)
                total_graph_payloads += 1

            logger.info("Done %s. Stored payloads: %d", pdf.name, total_graph_payloads)

        self.store.close()


if __name__ == "__main__":
    GraphBuilder().build()

