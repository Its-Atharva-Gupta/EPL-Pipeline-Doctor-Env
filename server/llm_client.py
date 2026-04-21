import hashlib
import logging
from typing import Any

from .constants import OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

_cache: dict[str, Any] = {}


def _cache_key(*parts: str) -> str:
    combined = "\x00".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def get_llm_response(
    user_prompt: str,
    system_prompt: str,
    config: Any = None,  # ProviderConfig | None
    timeout: float = OLLAMA_TIMEOUT,
    cache_key: str | None = None,
) -> str:
    """Call the configured LLM provider. Falls back to ollama if config is None."""
    from .llm_providers import call_llm

    if config is None:
        from models import ProviderConfig

        config = ProviderConfig(provider="ollama", model=OLLAMA_MODEL)

    key = cache_key or _cache_key(config.provider, config.model, system_prompt, user_prompt)
    if key in _cache:
        logger.debug("LLM cache hit for key=%s", key[:8])
        return _cache[key]

    result = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
        timeout=timeout,
    )
    _cache[key] = result
    return result


def call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: float = OLLAMA_TIMEOUT,
    cache_key: str | None = None,
) -> str:
    """Backward-compatible wrapper — delegates to get_llm_response with ollama config."""
    from models import ProviderConfig

    cfg = ProviderConfig(provider="ollama", model=model)
    return get_llm_response(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        config=cfg,
        timeout=timeout,
        cache_key=cache_key,
    )
