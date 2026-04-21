import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    config: Any,  # ProviderConfig — avoid circular import at module level
    timeout: float = 8.0,
) -> str:
    """Dispatch to the appropriate LLM provider and return raw text."""
    match config.provider:
        case "anthropic":
            return _call_anthropic(system_prompt, user_prompt, config, timeout)
        case "openai":
            return _call_openai_compat(system_prompt, user_prompt, config, timeout)
        case "groq":
            return _call_openai_compat(
                system_prompt,
                user_prompt,
                config,
                timeout,
                base_url="https://api.groq.com/openai/v1",
            )
        case "openrouter":
            return _call_openai_compat(
                system_prompt,
                user_prompt,
                config,
                timeout,
                base_url="https://openrouter.ai/api/v1",
                extra_headers={"HTTP-Referer": "https://github.com/etl-pipeline-doctor"},
            )
        case "ollama":
            return _call_ollama(system_prompt, user_prompt, config, timeout)
        case _:
            raise ValueError(f"Unknown provider: {config.provider}")


def _call_anthropic(system_prompt: str, user_prompt: str, config: Any, timeout: float) -> str:
    import anthropic

    start = time.monotonic()
    client = anthropic.Anthropic(api_key=config.api_key, timeout=timeout)
    message = client.messages.create(
        model=config.model,
        max_tokens=256,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    elapsed = time.monotonic() - start
    text = message.content[0].text if message.content else ""
    logger.info(
        "Anthropic call model=%s elapsed=%.2fs in=%d out=%d",
        config.model,
        elapsed,
        message.usage.input_tokens,
        message.usage.output_tokens,
    )
    return text


def _call_openai_compat(
    system_prompt: str,
    user_prompt: str,
    config: Any,
    timeout: float,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> str:
    import openai

    start = time.monotonic()
    kwargs: dict[str, Any] = {"api_key": config.api_key, "timeout": timeout}
    effective_base_url = base_url or config.base_url or None
    if effective_base_url:
        kwargs["base_url"] = effective_base_url
    if extra_headers:
        kwargs["default_headers"] = extra_headers

    client = openai.OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=256,
    )
    elapsed = time.monotonic() - start
    text = response.choices[0].message.content or ""
    usage = response.usage
    logger.info(
        "OpenAI-compat call provider=%s model=%s elapsed=%.2fs in=%d out=%d",
        config.provider,
        config.model,
        elapsed,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )
    return text


def _call_ollama(system_prompt: str, user_prompt: str, config: Any, timeout: float) -> str:
    import ollama

    start = time.monotonic()
    response = ollama.chat(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"num_predict": 256},
        think=False,
    )
    elapsed = time.monotonic() - start
    text = response.message.content or ""
    logger.info("Ollama call model=%s elapsed=%.2fs len_out=%d", config.model, elapsed, len(text))
    return text
