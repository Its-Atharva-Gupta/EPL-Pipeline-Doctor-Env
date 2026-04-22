"""
Process-wide provider configuration storage.

OpenEnv creates one Environment instance per WebSocket session. The /configure
endpoint lives at the FastAPI app layer, so this module provides a small,
cycle-free bridge for sharing provider config with env instances.
"""

from __future__ import annotations

from typing import Any

_provider_config: Any | None = None  # ProviderConfig | None (kept as Any to avoid import cycles)


def set_provider_config(config: Any | None) -> None:
    global _provider_config
    _provider_config = config


def get_provider_config() -> Any | None:
    return _provider_config

