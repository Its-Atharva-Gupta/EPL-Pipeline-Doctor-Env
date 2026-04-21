"""Sync HTTP client for the ETL Pipeline Doctor environment server."""

import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent))

from models import ETLAction, ETLObservation, ETLState


class ETLPipelineDoctorEnv:
    """Sync HTTP client wrapping the /reset, /step, /state endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(self, seed: int | None = None, **kwargs) -> ETLObservation:
        payload: dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = seed
        payload.update(kwargs)
        resp = self._session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return ETLObservation(**data.get("observation", data))

    def step(self, action: ETLAction) -> ETLObservation:
        payload = action.model_dump()
        resp = self._session.post(f"{self.base_url}/step", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return ETLObservation(**data.get("observation", data))

    def state(self) -> ETLState:
        resp = self._session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        data.pop("judge", None)  # server adds "judge" key; ETLState doesn't have it
        return ETLState(**data)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "ETLPipelineDoctorEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
