import json
import logging
import pathlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ETLAction, ToolResult

from .constants import OLLAMA_TIMEOUT
from .llm_client import _cache_key, get_llm_response

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (pathlib.Path(__file__).parent / "prompts" / "judge_system.md").read_text()


class LLMJudge:
    """Scores per-step reasoning quality using a configurable LLM provider."""

    def score(
        self,
        alert: str,
        compact_history: list[str],
        action: ETLAction,
        tool_result: ToolResult,
        config: Any = None,  # ProviderConfig | None
    ) -> float:
        score, _ = self.score_and_debug(
            alert=alert,
            compact_history=compact_history,
            action=action,
            tool_result=tool_result,
            config=config,
            include_prompts=False,
            include_raw=False,
        )
        return score

    def score_and_debug(
        self,
        alert: str,
        compact_history: list[str],
        action: ETLAction,
        tool_result: ToolResult,
        config: Any = None,  # ProviderConfig | None
        *,
        include_prompts: bool = True,
        include_raw: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        """Return judge score plus optional raw/provider debug info."""
        from .llm_client import cache_has

        user_prompt = _build_user_prompt(alert, compact_history, action, tool_result)
        cache_key = _cache_key(_SYSTEM_PROMPT, user_prompt)

        debug: dict[str, Any] = {
            "cache_key": cache_key,
            "cache_hit": cache_has(cache_key),
        }
        if include_prompts:
            debug["system_prompt"] = _SYSTEM_PROMPT
            debug["user_prompt"] = user_prompt

        try:
            raw = get_llm_response(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config=config,
                timeout=OLLAMA_TIMEOUT,
                cache_key=cache_key,
            )
            if include_raw:
                debug["raw"] = raw
            score, brief = _parse_score_and_brief(raw)
            debug["brief"] = brief
            return score, debug
        except Exception as exc:
            logger.warning("Judge call failed: %s — defaulting to 0.0", exc)
            debug["error"] = str(exc)
            return 0.0, debug


def _build_user_prompt(
    alert: str,
    compact_history: list[str],
    action: ETLAction,
    tool_result: ToolResult,
) -> str:
    history_text = "\n".join(compact_history) if compact_history else "(none)"
    cmd_short = action.command if len(action.command) <= 100 else action.command[:97] + "..."
    return (
        f"ALERT: {alert}\n"
        f"PRIOR ACTIONS:\n{history_text}\n"
        f"CURRENT ACTION:\n"
        f"  Command: {cmd_short}\n"
        f"RESULT: {tool_result.output[:500]}\n"
        f"Success: {'Yes' if tool_result.success else 'No'}\n\n"
        "Score the command quality on a scale from -1.0 (irrational or breaks things) "
        "to +1.0 (smart diagnostic step or correct fix). Return exactly this JSON:\n"
        '{"score": <float>, "brief": "<one-sentence justification>"}'
    )


def _parse_score_and_brief(raw: str) -> tuple[float, str]:
    try:
        # Find JSON in the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        data = json.loads(raw[start:end])
        score = float(data["score"])
        brief = str(data.get("brief", ""))
        return max(-1.0, min(1.0, score)), brief
    except Exception as exc:
        logger.warning("Failed to parse judge response '%s': %s", raw[:100], exc)
        return 0.0, ""
