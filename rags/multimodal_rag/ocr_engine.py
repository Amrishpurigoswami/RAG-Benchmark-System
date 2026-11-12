"""OCR Engine — Extract text from scanned PDFs and images.

Pipeline:
  Image (page screenshot or extracted image)
    → OCR
    → Text
    → Existing text pipeline (graph_utils -> graph_extractor -> etc.)

Supports multiple OCR backends:
1. EasyOCR (primary, good accuracy, no separate install)
2. Tesseract (fallback, requires system install)
3. PyMuPDF text extraction (for digital PDFs, no OCR needed)

Output format:
{
    "page": int,
    "text": str,
    "confidence": float,
    "blocks": [  # Optional: per-word bounding boxes
        {
            "text": str,
            "bbox": (x0, y0, x1, y1),
            "confidence": float
        }
    ]
}
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rags.multimodal_rag.image_extractor import ImageExtractor

logger = logging.getLogger(__name__)


class OCREngine:
    """Extract text from images and scanned PDFs using OCR."""

    def __init__(self, backend: str = "easyocr", lang: List[str] | None = None):
        """
        Args:
            backend: OCR backend to use ("easyocr", "tesseract", or "none").
            lang: Language codes (e.g., ["en"]). Defaults to English.
        """
        self.backend = backend
        self.lang = lang or ["en"]
        self._reader = None  # Lazy-loaded OCR reader

    def _lazy_init(self) -> None:
        """Initialize the OCR reader on first use."""
        if self._reader is not None:
            return

        if self.backend == "easyocr":
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    self.lang,
                    gpu=False,  # Use CPU for broad compatibility
                )
                logger.info("[OCREngine] Initialized EasyOCR reader")
            except ImportError:
                logger.warning(
                    "EasyOCR not installed. "
                    "Install with: pip install easyocr"
                )
                self._reader = None
        elif self.backend == "tesseract":
            try:
                import pytesseract
                # Verify tesseract is installed
                pytesseract.get_tesseract_version()
                self._reader = pytesseract
                logger.info("[OCREngine] Initialized Tesseract OCR")
            except (ImportError, Exception):
                logger.warning(
                    "Tesseract not available. "
                    "Install tesseract-ocr system package and: pip install pytesseract"
                )
                self._reader = None
        else:
            logger.warning(f"Unknown OCR backend: {self.backend}")
            self._reader = None

    def ocr_page(
        self, pdf_path: Path, page_number: int, dpi: int = 200
    ) -> Dict[str, Any]:
        """OCR a single PDF page rendered as an image.

        Args:
            pdf_path: Path to the PDF.
            page_number: 1-indexed page number.
            dpi: Resolution for rendering (higher = better OCR, slower).

        Returns:
            OCR result with extracted text.
        """
        # Render page to image
        image_bytes = ImageExtractor.get_page_screenshot(pdf_path, page_number, dpi)

        if not image_bytes:
            return {
                "page": page_number,
                "text": "",
                "confidence": 0.0,
                "blocks": [],
                "error": "Failed to render page",
            }

        return self.ocr_image_bytes(image_bytes, page_number)

    def ocr_pdf(self, pdf_path: Path, dpi: int = 200) -> List[Dict[str, Any]]:
        """OCR all pages of a PDF.

        Args:
            pdf_path: Path to the PDF.
            dpi: Resolution for rendering.

        Returns:
            List of OCR results, one per page.
        """
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        import fitz

        doc = fitz.open(pdf_path)
        try:
            total_pages = len(doc)
            results = []

            for page_num in range(1, total_pages + 1):
                result = self.ocr_page(pdf_path, page_num, dpi)
                results.append(result)
                logger.info(
                    f"OCR page {page_num}/{total_pages}: "
                    f"{len(result['text'])} chars, "
                    f"confidence={result['confidence']:.2f}"
                )

            return results
        finally:
            doc.close()

    def ocr_image_file(self, image_path: Path, page_number: int = 0) -> Dict[str, Any]:
        """OCR a single image file.

        Args:
            image_path: Path to the image file (PNG, JPG, etc.).
            page_number: Logical page number (default 0).

        Returns:
            OCR result.
        """
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_bytes = image_path.read_bytes()
        return self.ocr_image_bytes(image_bytes, page_number)

    def ocr_image_bytes(
        self, image_bytes: bytes, page_number: int
    ) -> Dict[str, Any]:
        """OCR an image from raw bytes.

        Args:
            image_bytes: PNG/JPEG image bytes.
            page_number: Logical page number.

        Returns:
            OCR result dict.
        """
        self._lazy_init()

        if self._reader is None:
            return {
                "page": page_number,
                "text": "",
                "confidence": 0.0,
                "blocks": [],
                "error": "No OCR backend available",
            }

        if self.backend == "easyocr":
            return self._ocr_easyocr(image_bytes, page_number)
        elif self.backend == "tesseract":
            return self._ocr_tesseract(image_bytes, page_number)
        else:
            return {
                "page": page_number,
                "text": "",
                "confidence": 0.0,
                "blocks": [],
                "error": f"Unknown backend: {self.backend}",
            }

    def _ocr_easyocr(self, image_bytes: bytes, page_number: int) -> Dict[str, Any]:
        """Run EasyOCR on image bytes."""
        import numpy as np
        from PIL import Image
        import io

        # Convert bytes to PIL Image to numpy array
        pil_image = Image.open(io.BytesIO(image_bytes))
        np_image = np.array(pil_image)

        # Run OCR
        results = self._reader.readtext(np_image)

        # Process results
        text_parts: List[str] = []
        blocks: List[Dict[str, Any]] = []
        total_confidence = 0.0
        valid_count = 0

        for (bbox, text, confidence) in results:
            if not text or not text.strip():
                continue

            text_parts.append(text.strip())

            blocks.append({
                "text": text.strip(),
                "bbox": tuple(bbox[0]) if bbox else None,
                "confidence": float(confidence),
            })

            total_confidence += confidence
            valid_count += 1

        confidence = total_confidence / max(valid_count, 1)

        result_text = "\n".join(text_parts)

        return {
            "page": page_number,
            "text": result_text,
            "confidence": confidence,
            "blocks": blocks,
            "error": None,
        }

    def _ocr_tesseract(self, image_bytes: bytes, page_number: int) -> Dict[str, Any]:
        """Run Tesseract OCR on image bytes."""
        from PIL import Image
        import io

        pil_image = Image.open(io.BytesIO(image_bytes))

        # Run OCR
        data = self._reader.image_to_data(
            pil_image,
            lang="+".join(self.lang),
            output_type="dict",
        )

        text_parts: List[str] = []
        blocks: List[Dict[str, Any]] = []
        total_confidence = 0.0
        valid_count = 0

        for i in range(len(data["text"])):
            text = (data["text"][i] or "").strip()
            conf = data["conf"][i]

            if not text or conf == -1:
                continue

            text_parts.append(text)

            blocks.append({
                "text": text,
                "bbox": (
                    data["left"][i],
                    data["top"][i],
                    data["left"][i] + data["width"][i],
                    data["top"][i] + data["height"][i],
                ),
                "confidence": conf / 100.0,
            })

            try:
                total_confidence += conf / 100.0
                valid_count += 1
            except (ValueError, TypeError):
                pass

        confidence = total_confidence / max(valid_count, 1)

        return {
            "page": page_number,
            "text": "\n".join(text_parts),
            "confidence": confidence,
            "blocks": blocks,
            "error": None,
        }

    @staticmethod
    def needs_ocr(pdf_path: Path) -> bool:
        """Check if a PDF likely needs OCR (no extractable text).

        Args:
            pdf_path: Path to the PDF.

        Returns:
            True if the PDF has no extractable text (likely scanned).
        """
        import fitz

        doc = fitz.open(pdf_path)
        try:
            total_text = ""
            for page in doc:
                total_text += page.get_text() or ""
            return len(total_text.strip()) < 50
        finally:
            doc.close()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ocr = OCREngine(backend="easyocr")

    test_pdf = Path("data/hemant_story.pdf")
    if test_pdf.is_file():
        needs = ocr.needs_ocr(test_pdf)
        print(f"\nPDF needs OCR: {needs}")

        if needs:
            result = ocr.ocr_page(test_pdf, page_number=1)
            print(f"OCR page 1: {len(result['text'])} chars, confidence={result['confidence']:.2f}")
            print(f"Text preview: {result['text'][:200]}")
        else:
            print("PDF has extractable text, no OCR needed")

        with open(test_pdf, "rb") as f:
            img = ImageExtractor.get_page_screenshot(test_pdf, 1)
            if img:
                result = ocr.ocr_image_bytes(img, page_number=1)
                print(f"\nOCR from screenshot: {len(result['text'])} chars")
                print(f"Text: {result['text'][:200]}")
    else:
        print(f"No test PDF found at {test_pdf}")

