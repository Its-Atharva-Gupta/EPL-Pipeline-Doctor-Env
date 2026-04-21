import hashlib
import logging
import time
from typing import Any

import ollama

from .constants import OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

_cache: dict[str, Any] = {}


def _cache_key(*parts: str) -> str:
    combined = "\x00".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: float = OLLAMA_TIMEOUT,
    cache_key: str | None = None,
) -> str:
    """Call Ollama with timeout. Returns raw text response."""
    key = cache_key or _cache_key(system_prompt, user_prompt, model)
    if key in _cache:
        logger.debug("LLM cache hit for key=%s", key[:8])
        return _cache[key]

    start = time.monotonic()
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"num_predict": 256},
            think=False,  # gemma4 is a thinking model; suppress CoT to get content in message.content
        )
        elapsed = time.monotonic() - start
        text = response.message.content or ""
        logger.info("Ollama call model=%s elapsed=%.2fs len_out=%d", model, elapsed, len(text))
        _cache[key] = text
        return text
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.warning("Ollama call failed after %.2fs: %s", elapsed, exc)
        raise
