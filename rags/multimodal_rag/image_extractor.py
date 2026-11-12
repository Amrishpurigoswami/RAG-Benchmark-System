"""Image Extractor — Extract embedded images from PDF pages.

Uses PyMuPDF (fitz) to locate and extract images from PDF documents.
Saves images to a temporary directory and returns metadata for captioning.

Output format (per image):
{
    "page": int,
    "image_index": int,
    "path": str | None,          # Path to saved image file (if saved)
    "width": int,
    "height": int,
    "bbox": (x0, y0, x1, y1),   # Position on page
    "image_bytes": bytes | None, # Raw image bytes (if not saved to disk)
    "metadata": { ... }          # Any extraction metadata
}
"""

import io
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF


class ImageExtractor:
    """Extract embedded images from PDF pages using PyMuPDF."""

    # Minimum image dimension to consider meaningful (avoids icons, bullets, etc.)
    MIN_IMAGE_SIZE: int = 64

    # Maximum number of images to extract per page (prevents spamming)
    MAX_IMAGES_PER_PAGE: int = 20

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Args:
            output_dir: Directory to save extracted images. If None, images
                        are kept in memory as bytes.
        """
        self.output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_pdf(
        self, pdf_path: Path, save_images: bool = True
    ) -> List[Dict[str, Any]]:
        """Extract all images from a PDF.

        Args:
            pdf_path: Path to the PDF file.
            save_images: If True, save images to output_dir. If False,
                         keep image bytes in memory.

        Returns:
            List of image metadata dicts.
        """
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        all_images: List[Dict[str, Any]] = []

        doc = fitz.open(pdf_path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                images = self._extract_from_page(page, page_num + 1, save_images)
                all_images.extend(images)

                # Stop if we have too many images total (safety limit)
                if len(all_images) > 200:
                    print(f"[ImageExtractor] Reached image limit, stopping.")
                    break

        finally:
            doc.close()

        print(f"[ImageExtractor] Extracted {len(all_images)} images from {pdf_path.name}")
        return all_images

    def extract_from_page_bytes(self, page_bytes: bytes, page_number: int) -> List[Dict[str, Any]]:
        """Extract images from a single page given as raw PDF bytes."""
        doc = fitz.open(stream=page_bytes, filetype="pdf")
        try:
            if len(doc) == 0:
                return []
            page = doc[0]
            return self._extract_from_page(page, page_number, save_images=False)
        finally:
            doc.close()

    def _extract_from_page(
        self, page, page_number: int, save_images: bool
    ) -> List[Dict[str, Any]]:
        """Extract images from a single PyMuPDF Page object."""
        images: List[Dict[str, Any]] = []
        image_list = page.get_images(full=True)

        if not image_list:
            return images

        for img_idx, img_info in enumerate(image_list):
            if img_idx >= self.MAX_IMAGES_PER_PAGE:
                break

            xref = img_info[0]
            base_image = page.parent.extract_image(xref)

            if not base_image:
                continue

            image_bytes = base_image.get("image")
            if not image_bytes:
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            # Skip small images (icons, bullets, decorations)
            if width < self.MIN_IMAGE_SIZE or height < self.MIN_IMAGE_SIZE:
                continue

            ext = base_image.get("ext", "png")

            # Try to get bbox on page
            bbox = self._get_image_bbox(page, xref)

            image_entry: Dict[str, Any] = {
                "page": page_number,
                "image_index": img_idx,
                "width": width,
                "height": height,
                "bbox": bbox,
                "ext": ext,
                "image_bytes": None,
                "path": None,
                "metadata": {
                    "xref": xref,
                    "source_page": page_number,
                },
            }

            if save_images and self.output_dir:
                img_filename = f"page{page_number:04d}_img{img_idx:03d}_{uuid.uuid4().hex[:8]}.{ext}"
                img_path = self.output_dir / img_filename
                img_path.write_bytes(image_bytes)
                image_entry["path"] = str(img_path)
            else:
                image_entry["image_bytes"] = image_bytes

            images.append(image_entry)

        return images

    @staticmethod
    def _get_image_bbox(page, xref: int) -> Optional[Tuple[float, float, float, float]]:
        """Get the bounding box of an image on a page."""
        try:
            # Iterate through page drawings/images to find bbox
            for img_block in page.get_image_info():
                if img_block.get("xref") == xref:
                    bbox = img_block.get("bbox")
                    if bbox and len(bbox) == 4:
                        return tuple(float(v) for v in bbox)
        except Exception:
            pass
        return None

    @staticmethod
    def get_page_screenshot(pdf_path: Path, page_number: int, dpi: int = 150) -> Optional[bytes]:
        """Render a full page as an image (useful for OCR or captioning).

        Args:
            pdf_path: Path to the PDF.
            page_number: 1-indexed page number.
            dpi: Resolution for rendering.

        Returns:
            PNG image bytes, or None on failure.
        """
        doc = fitz.open(pdf_path)
        try:
            if page_number < 1 or page_number > len(doc):
                return None
            page = doc[page_number - 1]
            zoom = dpi / 72  # Default PDF DPI is 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            return pix.tobytes("png")
        except Exception:
            return None
        finally:
            doc.close()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        extractor = ImageExtractor(output_dir=Path(tmp_dir))
        test_pdf = Path("data/hemant_story.pdf")
        if test_pdf.is_file():
            images = extractor.extract_from_pdf(test_pdf, save_images=True)
            print(f"\nExtracted {len(images)} images:")
            for img in images[:5]:
                print(f"  Page {img['page']}: {img['width']}x{img['height']} -> {img['path']}")
            if len(images) > 5:
                print(f"  ... and {len(images) - 5} more")
        else:
            print(f"No test PDF found at {test_pdf}")

