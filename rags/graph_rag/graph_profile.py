import json
from pathlib import Path

from rags.graph_rag.llm_config import get_construction_client, get_construction_models


class GraphProfile:
    """
    ---------------------------------------------------------
    Analyze a PDF document before graph construction.

    Outputs:
        profiles/
            <document>_profile.json
            <document>_summary.txt
            <document>_extraction_prompt.txt

    This is executed ONLY ONCE for every PDF.
    ---------------------------------------------------------
    """

    def __init__(self):
        self.client = get_construction_client()
        self.model_fallbacks = get_construction_models()
        print(f"[GraphProfile] Provider: {self.client.base_url}")
        print(f"[GraphProfile] Fallback models: {self.model_fallbacks}")

        self.output_folder = Path("profiles")
        self.output_folder.mkdir(exist_ok=True)

    # -------------------------------------------------------
    # Build prompt for document understanding
    # -------------------------------------------------------

    def _prompt(self, document_text):

        return f"""
You are an expert Knowledge Graph Architect.

Your task is NOT to answer questions.

Your task is to understand the entire document and prepare
instructions for another LLM that will later construct a
Knowledge Graph.

Analyze the document carefully.

Return ONLY valid JSON.

JSON format:

{{
    "document_type":"",
    "summary":"",
    "main_entities":[],
    "important_attributes":[],
    "important_relationships":[],
    "ignore":[],
    "graph_focus":[]
}}

Rules

1. Detect document type.

2. Summarize document.

3. List only important entities.

4. List attributes users will ask.

5. List useful relationships.

6. Ignore greetings,
fillers,
small talk,
duplicate information.

7. Graph focus means
what information should become nodes.

Document

----------------------------

{document_text[:120000]}

----------------------------
"""

    # -------------------------------------------------------
    # Generate profile
    # -------------------------------------------------------

    def build_profile(self, pdf_name, document_text):

        prompt = self._prompt(document_text)

        # Try models in order (primary -> secondary -> tertiary -> quaternary)
        last_err: Exception | None = None
        content = None

        for model in self.model_fallbacks:
            for attempt in range(1, 4):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content_raw = response.choices[0].message.content
                    if content_raw is None:
                        raise RuntimeError(f"GraphProfile: model {model} returned message.content=None")

                    content = content_raw.strip()
                    if not content:
                        raise RuntimeError(f"GraphProfile: model {model} returned empty profile JSON")

                    break  # success
                except Exception as e:
                    last_err = e
                    import time
                    time.sleep(min(2 ** attempt, 4))

            if content:
                break  # got content, no need to try next model
        else:
            raise RuntimeError(f"GraphProfile: all models failed. Last error: {last_err}")



        if content.startswith("```"):
            content = content.split("```")[1]
            content = content.replace("json", "").strip()

        profile = json.loads(content)

        stem = Path(pdf_name).stem

        profile_file = self.output_folder / f"{stem}_profile.json"

        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=4)

        summary_file = self.output_folder / f"{stem}_summary.txt"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(profile["summary"])

        extraction_prompt = self.build_extraction_prompt(profile)

        prompt_file = self.output_folder / f"{stem}_extraction_prompt.txt"

        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(extraction_prompt)

        print(f"Created profile for {pdf_name}")

        return profile

    # -------------------------------------------------------
    # Build document-specific extraction prompt
    # -------------------------------------------------------

    def build_extraction_prompt(self, profile):

        prompt = f"""
You are building a Knowledge Graph.

Document Type

{profile['document_type']}

Focus on these entities:

{', '.join(profile['main_entities'])}

Extract these attributes:

{', '.join(profile['important_attributes'])}

Create relationships such as:

{', '.join(profile['important_relationships'])}

Ignore:

{', '.join(profile['ignore'])}

Graph should prioritize:

{', '.join(profile['graph_focus'])}

Never create entities from greetings,
small talk,
duplicate sentences,
or conversational fillers.

Every node must be directly supported
by the document.

Every relationship must be explicitly
mentioned in the document.

Output a clean and accurate graph.
"""

        return prompt