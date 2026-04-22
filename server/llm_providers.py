import logging
import os
import random
import threading
import time
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_MIN_INTERVAL_S = 0.25  # 250ms between outbound API calls (global)
_DEFAULT_MAX_CONCURRENCY = 2
_DEFAULT_MAX_RETRIES = 6
_DEFAULT_BACKOFF_BASE_S = 0.5
_DEFAULT_JITTER_MAX_S = 0.25
_DEFAULT_MAX_BACKOFF_S = 30.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = float(raw)
        return v if v >= 0 else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = int(raw)
        return v if v >= 0 else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Global throttling + concurrency control (process-wide)
# ---------------------------------------------------------------------------

_throttle_lock = threading.Lock()
_next_allowed_request_time: float = 0.0
_concurrency_sem = threading.BoundedSemaphore(
    max(1, _env_int("LLM_MAX_CONCURRENCY", _DEFAULT_MAX_CONCURRENCY))
)


def _throttle_global() -> None:
    """Ensure a minimum delay between outbound API calls (global across threads)."""
    global _next_allowed_request_time
    min_interval = _env_float("LLM_MIN_INTERVAL_S", _DEFAULT_MIN_INTERVAL_S)
    if min_interval <= 0:
        return

    sleep_for = 0.0
    with _throttle_lock:
        now = time.monotonic()
        if now < _next_allowed_request_time:
            sleep_for = _next_allowed_request_time - now
            _next_allowed_request_time = _next_allowed_request_time + min_interval
        else:
            _next_allowed_request_time = now + min_interval

    if sleep_for > 0:
        time.sleep(sleep_for)


def _extract_retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort Retry-After parsing from common exception shapes (OpenAI/Anthropic/httpx)."""
    headers = None
    response = getattr(exc, "response", None) or getattr(exc, "http_response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None

    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if not raw:
        return None

    # Header may be integer seconds, float seconds, or HTTP-date.
    try:
        return max(0.0, float(str(raw).strip()))
    except Exception:
        pass

    try:
        import email.utils
        from datetime import timezone

        dt = email.utils.parsedate_to_datetime(str(raw).strip())
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = time.time()
        return max(0.0, dt.timestamp() - now)
    except Exception:
        return None


def _is_rate_limit_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 429:
        return True

    response = getattr(exc, "response", None) or getattr(exc, "http_response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    # Fall back to string match to cover provider-specific exceptions.
    msg = str(exc).lower()
    return "429" in msg and ("too many requests" in msg or "rate limit" in msg)


def call_llm_with_retry(
    request_fn: Callable[[], T],
    *,
    provider: str,
    model: str,
    max_retries: int | None = None,
) -> T:
    """Execute an LLM API call with global throttling and 429 backoff."""
    retries = _env_int("LLM_MAX_RETRIES", _DEFAULT_MAX_RETRIES) if max_retries is None else max_retries
    base = _env_float("LLM_BACKOFF_BASE_S", _DEFAULT_BACKOFF_BASE_S)
    jitter_max = _env_float("LLM_BACKOFF_JITTER_MAX_S", _DEFAULT_JITTER_MAX_S)
    max_backoff = _env_float("LLM_MAX_BACKOFF_S", _DEFAULT_MAX_BACKOFF_S)

    attempt = 0
    while True:
        _concurrency_sem.acquire()
        try:
            _throttle_global()
            return request_fn()
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise

            if attempt >= retries:
                raise

            retry_after = _extract_retry_after_seconds(exc)
            jitter = random.uniform(0.0, max(0.0, jitter_max))
            backoff = max(0.0, base) * (2**attempt)
            wait = backoff + jitter
            if retry_after is not None:
                wait = max(wait, retry_after + jitter)
            if max_backoff > 0:
                wait = min(wait, max_backoff)

            logger.warning(
                "Rate limited (HTTP 429) provider=%s model=%s retry=%d/%d wait=%.2fs retry_after=%s",
                provider,
                model,
                attempt + 1,
                retries,
                wait,
                f"{retry_after:.2f}s" if retry_after is not None else "none",
            )
        finally:
            _concurrency_sem.release()

        time.sleep(wait)
        attempt += 1


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

    def _make_request():
        # Avoid duplicate retries in SDKs; prefer our centralized backoff.
        kwargs: dict[str, Any] = {"api_key": config.api_key, "timeout": timeout}
        try:
            client = anthropic.Anthropic(**kwargs, max_retries=0)
        except TypeError:
            client = anthropic.Anthropic(**kwargs)
        return client.messages.create(
            model=config.model,
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

    start = time.monotonic()
    message = call_llm_with_retry(
        cast(Callable[[], Any], _make_request),
        provider=config.provider,
        model=config.model,
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
    *,
    n: int = 1,
) -> str | list[str]:
    import openai

    start = time.monotonic()
    kwargs: dict[str, Any] = {
        "api_key": config.api_key,
        "timeout": timeout,
        # Avoid duplicate retries in SDKs; prefer our centralized backoff.
        "max_retries": 0,
    }
    effective_base_url = base_url or config.base_url or None
    if effective_base_url:
        kwargs["base_url"] = effective_base_url
    if extra_headers:
        kwargs["default_headers"] = extra_headers

    def _make_request():
        client = openai.OpenAI(**kwargs)
        return client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
            n=max(1, int(n)),
        )

    response = call_llm_with_retry(
        cast(Callable[[], Any], _make_request),
        provider=config.provider,
        model=config.model,
    )
    elapsed = time.monotonic() - start
    choices = getattr(response, "choices", []) or []
    texts = [(c.message.content or "") for c in choices]
    usage = response.usage
    logger.info(
        "OpenAI-compat call provider=%s model=%s elapsed=%.2fs in=%d out=%d",
        config.provider,
        config.model,
        elapsed,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )
    return texts[0] if (max(1, int(n)) == 1) else texts


def _call_ollama(system_prompt: str, user_prompt: str, config: Any, timeout: float) -> str:
    import ollama

    def _make_request():
        return ollama.chat(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"num_predict": 256},
            think=False,
        )

    start = time.monotonic()
    response = call_llm_with_retry(
        cast(Callable[[], Any], _make_request),
        provider=config.provider,
        model=config.model,
        max_retries=0,  # local; treat failures as fatal
    )
    elapsed = time.monotonic() - start
    text = response.message.content or ""
    logger.info("Ollama call model=%s elapsed=%.2fs len_out=%d", config.model, elapsed, len(text))
    return text
