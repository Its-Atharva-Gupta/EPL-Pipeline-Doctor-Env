#!/usr/bin/env python3
"""
Debug script: transparent trace of "training-like" reward evaluation.

What it shows:
- Server startup + health checks
- (Optional) POST /configure payload + response
- WebSocket connection URL
- Every WS JSON message sent/received (raw + parsed)
- For each "episode" (batch):
  - reset() response
  - sequential scoring of N candidate completions (like train.py reward_fn)
  - state() snapshots to show the server-side step counter evolving

This does NOT run TRL/GRPO; it reproduces the env interaction pattern that
train.py uses for reward evaluation so you can see exactly what's happening.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import requests

from client import ETLPipelineDoctorEnvAsync
from models import ETLAction, ProviderConfig


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _pp(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True)


def _print_kv(title: str, value: Any) -> None:
    print(f"[{_ts()}] {title}: {value}")


def _print_json(title: str, payload: Any) -> None:
    print(f"[{_ts()}] {title}:\n{_pp(payload)}")


def _wait_for_health(base_url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/healthz", timeout=2)
            if r.status_code == 200:
                _print_json("HEALTHZ OK", r.json())
                return
            last_err = f"status={r.status_code} body={r.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"Server not healthy after {timeout_s:.1f}s. Last error: {last_err}")


def _start_server_background(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    import uvicorn

    from server.app import app

    def _run() -> None:
        uvicorn.run(app, host=host, port=port, log_level="info")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


class VerboseETLPipelineDoctorEnvAsync(ETLPipelineDoctorEnvAsync):
    async def connect(self) -> "VerboseETLPipelineDoctorEnvAsync":
        _print_kv("WS CONNECT", getattr(self, "_ws_url", "<unknown>"))
        return await super().connect()  # type: ignore[return-value]

    async def disconnect(self) -> None:
        _print_kv("WS DISCONNECT", "closing")
        await super().disconnect()

    async def _send(self, message: Dict[str, Any]) -> None:  # type: ignore[override]
        raw = json.dumps(message, ensure_ascii=True)
        _print_kv("WS SEND", raw)
        await super()._send(message)

    async def _receive(self) -> Dict[str, Any]:  # type: ignore[override]
        # Re-implement base method so we can log the raw frame too.
        ws = getattr(self, "_ws", None)
        if ws is None:
            raise RuntimeError("WebSocket is not connected")
        raw = await asyncio.wait_for(ws.recv(), timeout=getattr(self, "_message_timeout", 60.0))
        _print_kv("WS RECV", raw)
        parsed = json.loads(raw)
        _print_json("WS RECV PARSED", parsed)
        return parsed


def _default_candidates() -> List[str]:
    # Include one intentional repeat to show repeat-penalties and state coupling.
    return [
        "TRACE LINEAGE gold.kpi_daily_revenue",
        "TRACE LINEAGE gold.kpi_daily_revenue",
        "INSPECT TABLE silver.orders_enriched",
        "SELECT * FROM gold_kpi_daily_revenue LIMIT 5",
        "VERIFY",
    ]


async def _score_candidates_sequential(
    env: VerboseETLPipelineDoctorEnvAsync,
    candidates: Iterable[str],
) -> None:
    """
    Mimics train.py: one reset(), then env.step() once per candidate sequentially.
    """
    for i, cmd in enumerate(candidates):
        _print_kv("CANDIDATE", f"{i}: {cmd}")
        obs = await env.step(ETLAction(command=cmd))
        # Observation is a pydantic model; model_dump() is stable.
        obs_dict = obs.observation.model_dump()
        _print_json("OBS", obs_dict)

async def _score_candidates_independent(
    env: VerboseETLPipelineDoctorEnvAsync,
    candidates: Iterable[str],
    seed: Optional[int],
) -> None:
    """
    Scores each candidate from a fresh reset() so rewards are comparable.

    This is NOT what train.py currently does; it's here as a diagnostic contrast.
    """
    for i, cmd in enumerate(candidates):
        _print_kv("CANDIDATE", f"{i}: {cmd}")
        _print_kv("RESET (per-candidate) seed", seed)
        reset_res = await env.reset(seed=seed)
        _print_json("RESET OBS (per-candidate)", reset_res.observation.model_dump())
        obs = await env.step(ETLAction(command=cmd))
        obs_dict = obs.observation.model_dump()
        _print_json("OBS", obs_dict)


async def main_async(args: argparse.Namespace) -> int:
    # -------------------------
    # 0) Start server
    # -------------------------
    base_url = args.base_url.rstrip("/")
    _print_kv("BASE_URL", base_url)
    _print_kv("START_SERVER", args.start_server)

    if args.start_server:
        _start_server_background(host=args.host, port=args.port)
        _wait_for_health(base_url, timeout_s=args.health_timeout_s)

    # -------------------------
    # 1) Configure judge (optional)
    # -------------------------
    if args.configure:
        cfg = ProviderConfig(
            provider=args.provider,
            model=args.model,
            api_key=os.environ.get(args.api_key_env, ""),
            base_url=args.provider_base_url or "",
        )
        _print_kv("CONFIGURE api_key_env", args.api_key_env)
        _print_json("CONFIGURE payload", cfg.model_dump())
        r = requests.post(f"{base_url}/configure", json=cfg.model_dump(), timeout=15)
        _print_kv("CONFIGURE status", r.status_code)
        try:
            _print_json("CONFIGURE response", r.json())
        except Exception:  # noqa: BLE001
            _print_kv("CONFIGURE response", r.text[:500])

    # -------------------------
    # 2) WebSocket episodes
    # -------------------------
    env = VerboseETLPipelineDoctorEnvAsync(base_url=base_url, message_timeout_s=args.ws_timeout_s)
    await env.connect()
    try:
        candidates = _default_candidates()
        if args.candidates_json:
            candidates = json.loads(args.candidates_json)
            if not isinstance(candidates, list) or not all(isinstance(x, str) for x in candidates):
                raise ValueError("--candidates-json must be a JSON array of strings")

        _print_json("CANDIDATES", candidates)

        for ep in range(args.episodes):
            _print_kv("EPISODE", ep)
            seed = args.seed + ep if args.seed is not None else None
            _print_kv("RESET seed", seed)
            reset_res = await env.reset(seed=seed)
            _print_json("RESET OBS", reset_res.observation.model_dump())

            _print_kv("STATE (pre)", "request")
            st = await env.state()
            _print_json("STATE (pre) data", st.model_dump())

            if args.evaluation_mode == "sequential":
                await _score_candidates_sequential(env, candidates[: args.num_candidates])
            else:
                await _score_candidates_independent(
                    env,
                    candidates[: args.num_candidates],
                    seed=seed,
                )

            _print_kv("STATE (post)", "request")
            st2 = await env.state()
            _print_json("STATE (post) data", st2.model_dump())

    finally:
        await env.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Transparent ETL env training trace")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--num-candidates", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--evaluation-mode",
        default="sequential",
        choices=["sequential", "independent"],
        help="sequential mimics train.py; independent resets per candidate",
    )

    p.add_argument("--start-server", action="store_true", help="Start uvicorn in a background thread")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--health-timeout-s", type=float, default=20.0)
    p.add_argument("--ws-timeout-s", type=float, default=120.0)

    p.add_argument("--configure", action="store_true", help="POST /configure before running episodes")
    p.add_argument("--provider", default="groq", choices=["anthropic", "openai", "groq", "openrouter", "ollama"])
    p.add_argument("--model", default="llama-3.3-70b-versatile")
    p.add_argument("--api-key-env", default="GROQ_API_KEY", help="Env var name holding provider API key")
    p.add_argument("--provider-base-url", default="", help="Optional base_url for openai-compatible providers")

    p.add_argument(
        "--candidates-json",
        default="",
        help="Override candidates: JSON array of command strings (e.g. '[\"VERIFY\",\"TRACE LINEAGE ...\"]')",
    )

    args = p.parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        _print_kv("INTERRUPT", "exiting")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
