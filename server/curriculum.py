import logging
import random
from collections import deque
from typing import Any

from .adversarial_designer import AdversarialDesigner
from .constants import (
    DEMOTION_THRESHOLD,
    FAULT_TYPES,
    PROMOTION_THRESHOLD,
    TIERS,
)
from .fault_catalogue import FaultCatalogue, FaultSpec

logger = logging.getLogger(__name__)


class CurriculumController:
    """Tracks per-fault-type mastery and picks scenarios for each episode."""

    def __init__(
        self,
        catalogue: FaultCatalogue,
        designer: AdversarialDesigner,
        seed: int = 0,
    ) -> None:
        self._cat = catalogue
        self._designer = designer
        self._rng = random.Random(seed)

        # EMA of success per fault type over last 10 episodes
        self._ema: dict[str, float] = {ft: 0.5 for ft in FAULT_TYPES}
        self._tier_idx: dict[str, int] = {ft: 0 for ft in FAULT_TYPES}

        # Recent episode history for designer
        self._history: deque[dict] = deque(maxlen=20)
        self._episode_count = 0
        self._last_designer_episode = 0

    def pick_fault(self, config: Any = None) -> FaultSpec:
        """Sample a fault type (biased toward un-mastered) and return a scenario."""
        self._episode_count += 1
        self._maybe_run_designer(config)

        # Bias sampling: weight by (1 - mastery)
        weights = [max(0.05, 1.0 - self._ema[ft]) for ft in FAULT_TYPES]
        fault_type = self._rng.choices(FAULT_TYPES, weights=weights, k=1)[0]
        tier = TIERS[self._tier_idx[fault_type]]
        spec = self._cat.pick(fault_type, tier, self._rng)
        logger.info(
            "Curriculum picked fault=%s tier=%s mastery=%.2f",
            fault_type,
            tier,
            self._ema[fault_type],
        )
        return spec

    def record_outcome(self, fault_type: str, resolved: bool, steps: int) -> None:
        """Update mastery EMA after an episode ends."""
        alpha = 0.2  # EMA weight for most recent episode (window ~10)
        success = 1.0 if resolved else 0.0
        old = self._ema[fault_type]
        self._ema[fault_type] = (1 - alpha) * old + alpha * success

        # Promote / demote tier
        mastery = self._ema[fault_type]
        idx = self._tier_idx[fault_type]
        if mastery >= PROMOTION_THRESHOLD and idx < len(TIERS) - 1:
            self._tier_idx[fault_type] = idx + 1
            logger.info("Promoted %s → %s", fault_type, TIERS[idx + 1])
        elif mastery <= DEMOTION_THRESHOLD and idx > 0:
            self._tier_idx[fault_type] = idx - 1
            logger.info("Demoted %s → %s", fault_type, TIERS[idx - 1])

        self._history.append(
            {
                "fault_type": fault_type,
                "resolved": resolved,
                "steps": steps,
            }
        )

    def state_summary(self) -> dict:
        return {
            "episode": self._episode_count,
            "mastery": dict(self._ema),
            "tiers": {ft: TIERS[idx] for ft, idx in self._tier_idx.items()},
        }

    def _maybe_run_designer(self, config: Any = None) -> None:
        """Run the adversarial designer every 20 episodes."""
        if self._episode_count - self._last_designer_episode < 20:
            return
        if len(self._history) < 5:
            return

        self._last_designer_episode = self._episode_count
        worst_ft = min(self._ema, key=lambda k: self._ema[k])
        failure_rate = 1.0 - self._ema[worst_ft]
        tier = TIERS[self._tier_idx[worst_ft]]

        spec = self._designer.design(
            worst_fault_type=worst_ft,
            failure_rate=failure_rate,
            underused_tools=[],
            common_wrong_fix="unknown",
            current_tier=tier,
            config=config,
        )
        if spec:
            self._cat.add_scenario(spec)
            logger.info("Adversarial designer added scenario: %s", spec)
