import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ETLAction, ETLObservation, ETLState, ProviderConfig

from .constants import OLLAMA_MODEL
from .etl_pipeline_doctor_environment import EtlPipelineDoctorEnvironment
from .llm_providers import call_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton — persists config across HTTP requests
# ---------------------------------------------------------------------------
_env: EtlPipelineDoctorEnvironment = EtlPipelineDoctorEnvironment()
_provider_config: ProviderConfig | None = None


def _auto_configure() -> ProviderConfig | None:
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key=api_key,
        )
    if os.environ.get("JUDGE_BACKEND") == "ollama":
        return ProviderConfig(provider="ollama", model=OLLAMA_MODEL)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provider_config
    auto = _auto_configure()
    if auto:
        _provider_config = auto
        _env.set_provider(auto)
        logger.info(
            "Auto-configured judge: provider=%s model=%s", auto.provider, auto.model
        )
    else:
        logger.info(
            "No judge configured at startup. POST /configure or set "
            "ANTHROPIC_API_KEY / JUDGE_BACKEND=ollama."
        )
    yield


app = FastAPI(title="ETL Pipeline Doctor", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _require_judge() -> None:
    if _provider_config is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Judge not configured. POST /configure with a ProviderConfig, "
                "or set the ANTHROPIC_API_KEY or JUDGE_BACKEND=ollama environment variable."
            ),
        )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    seed: int | None = None
    episode_id: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/configure")
def configure(config: ProviderConfig) -> dict[str, str]:
    """Validate provider config by smoke-testing the LLM, then store it."""
    global _provider_config
    try:
        call_llm(
            system_prompt="You are a test assistant.",
            user_prompt="Reply with the single word: ready",
            config=config,
            timeout=10.0,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"LLM smoke test failed: {exc}")

    _provider_config = config
    _env.set_provider(config)
    logger.info("Configured judge: provider=%s model=%s", config.provider, config.model)
    return {"status": "ok", "provider": config.provider, "model": config.model}


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()) -> dict[str, Any]:
    _require_judge()
    obs = _env.reset(seed=body.seed, episode_id=body.episode_id)
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(action: ETLAction) -> dict[str, Any]:
    _require_judge()
    obs = _env.step(action)
    return {"observation": obs.model_dump()}


@app.get("/state")
def state() -> dict[str, Any]:
    s = _env.get_state().model_dump()
    if _provider_config:
        s["judge"] = {"provider": _provider_config.provider, "model": _provider_config.model}
    else:
        s["judge"] = None
    return s


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": ETLAction.model_json_schema(),
        "observation": ETLObservation.model_json_schema(),
        "state": ETLState.model_json_schema(),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
