"""Image Caption — Generate descriptions for extracted images.

Converts images into text descriptions, then into graph JSON
compatible with graph_validator.py and graph_store.py.

Supports multiple captioning strategies:
1. Vision-Language Model (via OpenRouter vision-capable models)
2. Local BLIP-2 (via transformers, CPU-compatible)
3. Metadata-based fallback (dimensions, page position)

Output format (per image):
{
    "caption": "Human-readable description",
    "graph": {
        "entities": [...],
        "relationships": [...]
    },
    "page": int,
    "image_index": int
}
"""

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rags.graph_rag.llm_config import get_construction_client, get_construction_models

logger = logging.getLogger(__name__)

# Known vision-capable OpenRouter models (tried in order)
VISION_MODELS: List[str] = [
    "openai/gpt-4o-mini",
    "qwen/qwen-vl-plus:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "google/gemini-2.0-flash-exp:free",
]


class ImageCaption:
    """Generate captions for images and convert to graph facts.

    Uses the existing OpenRouter LLM config (via llm_config.py) so no
    additional API keys are needed. Falls back through multiple vision
    models, then a local BLIP-2 model, and finally a metadata-based
    description.
    """

    def __init__(self):
        self.client = get_construction_client()
        self.model_fallbacks = get_construction_models()
        self._blip_processor = None  # Lazy-loaded local BLIP-2
        self._blip_model = None
        logger.info(f"[ImageCaption] Provider: {self.client.base_url}")
        logger.info(f"[ImageCaption] Vision models: {VISION_MODELS}")

    def caption_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a caption + graph facts for a single image.

        Args:
            image_data: Image metadata dict from ImageExtractor.

        Returns:
            Dict with "caption", "graph", "page", "image_index".
        """
        page = image_data.get("page", 0)
        img_idx = image_data.get("image_index", 0)
        width = image_data.get("width", 0)
        height = image_data.get("height", 0)
        image_path = image_data.get("path")
        image_bytes = image_data.get("image_bytes")

        # 1) Try vision model captioning (if we have image bytes or path)
        caption = self._generate_caption(image_path, image_bytes, width, height)

        # 2) Convert caption into graph JSON
        graph = self._caption_to_graph(caption, page, img_idx)

        return {
            "caption": caption,
            "graph": graph,
            "page": page,
            "image_index": img_idx,
        }

    def _get_image_bytes(
        self,
        image_path: Optional[str],
        image_bytes: Optional[bytes],
    ) -> Optional[bytes]:
        """Resolve image bytes from either path or bytes field."""
        if image_bytes:
            return image_bytes
        if image_path and Path(image_path).is_file():
            return Path(image_path).read_bytes()
        return None

    def _generate_caption(
        self,
        image_path: Optional[str],
        image_bytes: Optional[bytes],
        width: int,
        height: int,
    ) -> str:
        """Generate a text description of the image.

        Strategy:
        1. Try vision-capable OpenRouter models (VISION_MODELS list).
        2. Try local BLIP-2 model (CPU, no API key needed).
        3. Fall back to metadata-based description.
        """
        bytes_data = self._get_image_bytes(image_path, image_bytes)

        # --- 1) Try vision-capable API models ---
        if bytes_data:
            for vision_model in VISION_MODELS:
                try:
                    logger.info(f"Trying vision model: {vision_model}")
                    return self._caption_with_vision_api(bytes_data, vision_model)
                except Exception as e:
                    logger.warning(f"Vision model {vision_model} failed: {e}")
                    continue

        # --- 2) Try local BLIP-2 (transformers, CPU) ---
        if bytes_data:
            try:
                local_caption = self._caption_with_local_model(bytes_data)
                if local_caption:
                    return local_caption
            except Exception as e:
                logger.warning(f"Local vision model failed: {e}")

        # --- 3) Fallback: metadata description ---
        return self._metadata_caption(width, height)

    def _caption_with_vision_api(self, image_bytes: bytes, model: str) -> str:
        """Send image to a vision-language model via OpenRouter.

        Args:
            image_bytes: Raw image bytes.
            model: OpenRouter model ID that supports vision.

        Returns:
            Caption text.
        """
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = """Describe this image in detail.

Focus on:

1. What type of image is this? (photograph, chart, diagram, table, logo, etc.)
2. What entities are present? (people, objects, text, data points)
3. What relationships exist between entities?
4. What text is visible in the image?
5. Is this an organization chart, flow diagram, or data visualization?

Be precise and structured. Your description will be used to build
a Knowledge Graph, so include specific names, titles, values,
and hierarchical relationships where visible."""

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )

        return response.choices[0].message.content or ""

    def _caption_with_local_model(self, image_bytes: bytes) -> Optional[str]:
        """Caption image using a local BLIP-2 model via transformers.

        This runs on CPU and requires: pip install transformers torch Pillow

        Returns:
            Caption string, or None if the model is not available.
        """
        try:
            from PIL import Image as PILImage
            import io

            if self._blip_processor is None or self._blip_model is None:
                try:
                    from transformers import BlipProcessor, BlipForConditionalGeneration
                    logger.info("Loading BLIP-2 processor and model (CPU)...")
                    self._blip_processor = BlipProcessor.from_pretrained(
                        "Salesforce/blip-image-captioning-base"
                    )
                    self._blip_model = BlipForConditionalGeneration.from_pretrained(
                        "Salesforce/blip-image-captioning-base"
                    )
                    self._blip_model.eval()
                    logger.info("BLIP-2 model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load BLIP-2 model: {e}")
                    # Try tiny BLIP as lighter fallback
                    try:
                        from transformers import BlipProcessor, BlipForConditionalGeneration
                        self._blip_processor = BlipProcessor.from_pretrained(
                            "Salesforce/blip-image-captioning-tiny"
                        )
                        self._blip_model = BlipForConditionalGeneration.from_pretrained(
                            "Salesforce/blip-image-captioning-tiny"
                        )
                        self._blip_model.eval()
                        logger.info("BLIP-tiny model loaded successfully")
                    except Exception as e2:
                        logger.warning(f"Failed to load BLIP-tiny model: {e2}")
                        self._blip_model = None

            if self._blip_processor is None or self._blip_model is None:
                return None

            pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self._blip_processor(pil_image, return_tensors="pt")
            out = self._blip_model.generate(**inputs, max_new_tokens=100)
            caption = self._blip_processor.decode(out[0], skip_special_tokens=True)
            return caption.strip()

        except ImportError:
            logger.warning(
                "transformers/torch not installed for local BLIP-2. "
                "Install with: pip install transformers torch Pillow"
            )
            return None
        except Exception as e:
            logger.warning(f"Local BLIP-2 caption failed: {e}")
            return None

    @staticmethod
    def _metadata_caption(width: int, height: int) -> str:
        """Generate a basic description from image metadata."""
        aspect_ratio = width / height if height > 0 else 1
        size_label = "small"
        if width > 1000 and height > 1000:
            size_label = "large"
        elif width > 500 and height > 500:
            size_label = "medium"

        orientation = "landscape" if aspect_ratio > 1.3 else "portrait" if aspect_ratio < 0.77 else "square"

        return (
            f"A {size_label} {orientation} image ({width}x{height} pixels). "
            f"Content could not be automatically described. "
            f"This may be a photograph, illustration, logo, or diagram."
        )

    def _caption_to_graph(
        self, caption: str, page: int, image_index: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract graph entities and relationships from a caption.

        Uses the existing graph extraction LLM to parse the caption
        text into structured graph JSON.
        """
        # Use the extraction LLM to parse caption into graph
        prompt = f"""Extract entities and relationships from this image description.

IMAGE DESCRIPTION:
{caption}

OUTPUT STRICT JSON:
{{
  "entities": [
    {{
      "id": "<Type>::<name>",
      "type": "<Type>",
      "name": "<name>",
      "attributes": {{
        "page": {page},
        "image_index": {image_index},
        "source": "image_caption",
        "evidence": "<supporting text from caption>"
      }}
    }}
  ],
  "relationships": [
    {{
      "type": "<RELATION_TYPE>",
      "source": "<entity-id>",
      "target": "<entity-id>",
      "properties": {{
        "page": {page},
        "image_index": {image_index},
        "evidence": "<supporting text>"
      }}
    }}
  ]
}}

Rules:
- Only create entities/relationships explicitly mentioned in the caption.
- If no clear entities or relationships exist, return empty lists.
- Use meaningful types like Person, Organization, Document, Chart, Table, etc."""

        last_err: Optional[Exception] = None
        for model in self.model_fallbacks:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content or ""
                # Parse JSON from response
                import json
                import re

                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```", 1)[1]
                    if "```" in content:
                        content = content.rsplit("```", 1)[0]

                first = content.find("{")
                last = content.rfind("}")
                if first != -1 and last != -1:
                    content = content[first : last + 1]

                data = json.loads(content)
                entities = data.get("entities") or []
                relationships = data.get("relationships") or []

                # Normalize to graph_validator format
                norm_entities = []
                for e in entities:
                    if not isinstance(e, dict):
                        continue
                    eid = e.get("id")
                    etype = e.get("type")
                    if not eid or not etype:
                        continue
                    norm_entities.append({
                        "id": str(eid),
                        "label": str(etype),
                        "properties": {
                            **(e.get("attributes") or {}),
                            "name": e.get("name", ""),
                        },
                    })

                norm_relationships = []
                for r in relationships:
                    if not isinstance(r, dict):
                        continue
                    rtype = r.get("type")
                    source = r.get("source")
                    target = r.get("target")
                    if not rtype or not source or not target:
                        continue
                    norm_relationships.append({
                        "type": str(rtype),
                        "source": str(source),
                        "target": str(target),
                        "properties": r.get("properties") or {},
                    })

                return {"entities": norm_entities, "relationships": norm_relationships}

            except Exception as e:
                last_err = e
                continue

        # Fallback: create a single "Image" entity from the caption
        return {
            "entities": [
                {
                    "id": f"Image::page{page}_img{image_index}",
                    "label": "Image",
                    "properties": {
                        "name": f"Image on page {page}",
                        "page": page,
                        "image_index": image_index,
                        "description": caption[:500],
                        "source": "image_caption",
                    },
                }
            ],
            "relationships": [],
        }

    def caption_images_batch(
        self, images: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate captions and graphs for multiple images."""
        results = []
        for img in images:
            try:
                result = self.caption_image(img)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to caption image on page {img.get('page')}: {e}")
                results.append({
                    "caption": f"Failed to process: {e}",
                    "graph": {"entities": [], "relationships": []},
                    "page": img.get("page", 0),
                    "image_index": img.get("image_index", 0),
                })
        return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    captioner = ImageCaption()

    # Test with a sample image entry
    sample_image = {
        "page": 1,
        "image_index": 0,
        "width": 800,
        "height": 600,
        "path": None,
        "image_bytes": None,
    }

    result = captioner.caption_image(sample_image)
    print(f"\nCaption: {result['caption']}")
    print(f"Entities: {len(result['graph']['entities'])}")
    print(f"Relationships: {len(result['graph']['relationships'])}")

