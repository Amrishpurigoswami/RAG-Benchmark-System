"""Provider-specific LLM configuration for the Graph RAG pipeline.

Graph construction (profile and extraction) always uses OpenRouter, while
answer synthesis uses Cerebras.  Keeping credentials separate prevents a key
for one provider being sent to the other provider's endpoint.
"""

import os
from typing import Final

from openai import OpenAI


OPENROUTER_BASE_URL: Final = "https://openrouter.ai/api/v1"
CEREBRAS_BASE_URL: Final = "https://api.cerebras.ai/v1"


def _required_env(name: str) -> str:
    """Return a required environment value with a clear configuration error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _models(*names: str, defaults: tuple[str, ...]) -> list[str]:
    """Collect configured models in order, removing duplicate entries."""
    configured = [os.getenv(name, "").strip() for name in names]
    result: list[str] = []
    for model in [*configured, *defaults]:
        if model and model not in result:
            result.append(model)
    return result


def get_construction_client() -> OpenAI:
    """Create the OpenRouter client used for profile and graph extraction."""
    return OpenAI(
        api_key=_required_env("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
    )


def get_construction_models() -> list[str]:
    """Return OpenRouter graph-construction models in configured fallback order."""
    return _models(
        "GRAPH_PRIMARY_MODEL",
        "GRAPH_SECONDARY_MODEL",
        "GRAPH_TERTIARY_MODEL",
        "GRAPH_QUATERNARY_MODEL",
        # Legacy names are retained temporarily for existing deployments.
        "PRIMARY_MODEL",
        "GRAPH_MODEL",
        "SECONDARY_MODEL",
        "TERTIARY_MODEL",
        "QUATERNARY_MODEL",
        defaults=("meta-llama/llama-3.3-70b-instruct:free",),
    )


def get_answer_client() -> OpenAI:
    """Create the Cerebras client used only for final answer generation."""
    return OpenAI(
        api_key=_required_env("CEREBRAS_API_KEY"),
        base_url=os.getenv("CEREBRAS_BASE_URL", CEREBRAS_BASE_URL),
    )


def get_answer_models() -> list[str]:
    """Return Cerebras answer-synthesis models in configured fallback order."""
    return _models(
        "ANSWER_PRIMARY_MODEL",
        "ANSWER_SECONDARY_MODEL",
        "ANSWER_TERTIARY_MODEL",
        "ANSWER_MODEL",
        "FALLBACK_MODEL",
        "SECOND_FALLBACK_MODEL",
        defaults=("gpt-oss-120b",),
    )


def get_answer_model() -> str:
    """Compatibility helper returning the preferred Cerebras answer model."""
    return get_answer_models()[0]
