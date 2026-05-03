"""
CrossDomainOrchestrator — Inter-model communication backbone.

Architecture:
  Event → DomainDetector.score(event)
       → cross-domain boost from RiskBus
       → final score published to RiskBus
       → ConsensusEngine monitors multi-domain flags

Components:
  RiskSignal: Signal unit between models
  RiskBus: Thread-safe shared memory with subscribers
  CrossDomainOrchestrator: Detection coordination and consensus
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Signal routing: domains listen to relevant peer signals
SIGNAL_ROUTES: Dict[str, List[str]] = {
    "procurement": ["contract", "finance", "approval"],
    "finance":     ["payroll",  "budget",  "approval"],
    "payroll":     ["finance",  "access"],
    "access":      ["approval", "finance"],
    "approval":    ["procurement", "contract"],
    "contract":    ["procurement", "budget"],
    "budget":      ["finance",  "approval"],
}


@dataclass
class RiskSignal:
    agent_id:       str
    domain:         str
    score:          float
    timestamp:      datetime
    event_metadata: dict = field(default_factory=dict)


class RiskBus:
    """Thread-safe shared memory for risk signals across models.
    Models publish scores after detecting events and query for cross-domain
    information about the same agent.
    """

    def __init__(self, decay_factor: float = 0.92, max_history: int = 10):
        self._lock = threading.Lock()
        self._memory: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.decay_factor = decay_factor
        self.max_history = max_history
        self._callbacks: List[Callable[[RiskSignal], None]] = []

    def publish(self, signal: RiskSignal) -> None:
        with self._lock:
            history = self._memory[signal.agent_id][signal.domain]
            history.append(signal.score)
            if len(history) > self.max_history:
                history.pop(0)
        for cb in self._callbacks:
            try:
                cb(signal)
            except Exception as e:
                logger.debug(f"RiskBus callback error: {e}")

    def get_cross_domain_score(
        self,
        agent_id: str,
        exclude_domain: str,
        route_filter: Optional[List[str]] = None,
    ) -> float:
        """Compute weighted aggregate score from peer domains.
        Uses exponential decay weighting (recent signals weighted higher).

        Args:
            agent_id: Agent ID to check
            exclude_domain: Domain to exclude (the querying domain)
            route_filter: Only consider domains in this list

        Returns:
            float 0.0-1.0: highest cross-domain score with decay weighting
        """
        with self._lock:
            agent_mem = dict(self._memory.get(agent_id, {}))

        relevant = {
            d: scores
            for d, scores in agent_mem.items()
            if d != exclude_domain
            and (route_filter is None or d in route_filter)
        }

        if not relevant:
            return 0.0

        domain_scores = []
        for _domain, history in relevant.items():
            if not history:
                continue
            # Recent score (last index) has highest weight
            n = len(history)
            weights = [self.decay_factor ** (n - 1 - i) for i in range(n)]
            w_sum = sum(weights)
            weighted_score = sum(w * s for w, s in zip(weights, history)) / w_sum
            domain_scores.append(weighted_score)

        return float(max(domain_scores)) if domain_scores else 0.0

    def get_consensus_alert(
        self, agent_id: str, threshold: float = 0.5
    ) -> dict:
        """Cek apakah ≥N domain menandai agen yang sama sebagai berisiko."""
        with self._lock:
            agent_mem = dict(self._memory.get(agent_id, {}))

        flagged = {
            d: scores[-1]
            for d, scores in agent_mem.items()
            if scores and scores[-1] >= threshold
        }
        return {
            "agent_id":        agent_id,
            "domains_flagged": list(flagged.keys()),
            "domain_scores":   flagged,
            "consensus_count": len(flagged),
            "max_score":       max(flagged.values(), default=0.0),
        }

    def subscribe(self, callback: Callable[[RiskSignal], None]) -> None:
        """Subscribe to all signals for logging/alerting."""
        self._callbacks.append(callback)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return latest score for each agent across all domains."""
        with self._lock:
            return {
                agent: {
                    domain: round(scores[-1], 4) if scores else 0.0
                    for domain, scores in domains.items()
                }
                for agent, domains in self._memory.items()
            }

    def agent_count(self) -> int:
        with self._lock:
            return len(self._memory)


# ─── CrossDomainOrchestrator ──────────────────────────────────────────────────
class CrossDomainOrchestrator:
    """Main coordinator for all domain detectors.
    Enables inter-model communication via RiskBus.

    Per-event flow:
      1. Find matching domain detector
      2. Score event with detector -> base_score
      3. Query RiskBus for signals from other domains -> cross_boost
      4. final_score = min(1.0, base_score + cross_boost)
      5. Publish final_score to RiskBus
      6. Check consensus: are >=2 domains flagging the same agent?
      7. Return complete result

    Cross-domain interaction matrix (see SIGNAL_ROUTES above):
      - Payroll flagged -> procurement & finance alert
      - Access anomaly -> approval & finance increase score
      - etc.

    Models can be hot-swapped while running without stopping simulation.
    """

    # Cross-domain signals contribute this much to final score
    CROSS_DOMAIN_BOOST_WEIGHT = 0.25

    # Alert if >=N domains flag the same agent
    CONSENSUS_MIN_DOMAINS = 2
    CONSENSUS_ALERT_THRESHOLD = 0.5

    def __init__(self, detectors: Dict[str, object]):
        """Args:
            detectors: {domain_id: fitted_detector_instance}
                       Detector must have .score(df) -> pd.Series method
        """
        self.detectors = dict(detectors)
        self.bus = RiskBus()
        self._alert_log: List[dict] = []
        self._event_count = 0
        self._lock = threading.Lock()

        # Wire logging callbacks
        self.bus.subscribe(self._on_high_risk_signal)

    # ── Main entry point ──────────────────────────────────────────────────────
    def process_event(self, event: dict) -> dict:
        """Process one simulation event in real-time.

        Args:
            event: dict from CorporateFraudSimulator.event_log

        Returns:
            dict containing:
              - base_score, cross_boost, final_score (float 0-1)
              - alert (bool)
              - consensus (dict)
              - domain, agent_id, timestamp
        """
        with self._lock:
            self._event_count += 1
            event_id = self._event_count

        domain    = event.get("domain", "")
        agent_id  = str(event.get("agent_id", ""))
        timestamp = event.get("timestamp", datetime.now())

        result = {
            "event_id":    event_id,
            "domain":      domain,
            "agent_id":    agent_id,
            "timestamp":   timestamp,
            "base_score":  0.0,
            "cross_boost": 0.0,
            "final_score": 0.0,
            "alert":       False,
            "consensus":   None,
        }

        # 1. Base detection ────────────────────────────────────────────────
        base_score = self._score_with_detector(domain, event)

        # 2. Cross-domain boost ────────────────────────────────────────────
        route_filter = SIGNAL_ROUTES.get(domain)
        cross_raw = self.bus.get_cross_domain_score(
            agent_id, exclude_domain=domain, route_filter=route_filter
        )
        cross_boost = self.CROSS_DOMAIN_BOOST_WEIGHT * cross_raw
        final_score = float(min(1.0, base_score + cross_boost))

        # 3. Publish ke RiskBus ────────────────────────────────────────────
        self.bus.publish(RiskSignal(
            agent_id=agent_id,
            domain=domain,
            score=final_score,
            timestamp=timestamp,
            event_metadata={
                "action": event.get("action_type"),
                "is_corrupt_gt": event.get("is_corrupt"),
            },
        ))

        # 4. Consensus check ───────────────────────────────────────────────
        consensus = self.bus.get_consensus_alert(
            agent_id, threshold=self.CONSENSUS_ALERT_THRESHOLD
        )
        is_alert = (
            final_score >= 0.5
            or consensus["consensus_count"] >= self.CONSENSUS_MIN_DOMAINS
        )

        result.update({
            "base_score":  round(base_score, 4),
            "cross_boost": round(cross_boost, 4),
            "final_score": round(final_score, 4),
            "alert":       is_alert,
            "consensus":   consensus,
        })

        if is_alert:
            with self._lock:
                self._alert_log.append({
                    k: v for k, v in result.items()
                    if k != "consensus"
                })
                self._alert_log[-1]["consensus_domains"] = ",".join(
                    consensus.get("domains_flagged", [])
                )
                self._alert_log[-1]["consensus_count"] = consensus.get(
                    "consensus_count", 0
                )

        return result

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _score_with_detector(self, domain: str, event: dict) -> float:
        detector = self.detectors.get(domain)
        if detector is None:
            return 0.0
        try:
            row_df = pd.DataFrame([event])
            scores = detector.score(row_df)
            return float(scores.iloc[0])
        except Exception as e:
            logger.debug(f"Detector '{domain}' scoring error: {e}")
            return 0.0

    def _on_high_risk_signal(self, signal: RiskSignal) -> None:
        """Callback: log warning if signal score is very high."""
        if signal.score >= 0.75:
            logger.warning(
                f"HIGH RISK | agent={signal.agent_id} | "
                f"domain={signal.domain} | score={signal.score:.3f}"
            )

    # ── Public API ────────────────────────────────────────────────────────────
    def update_detectors(self, new_detectors: Dict[str, object]) -> None:
        """Hot-swap new detectors into orchestrator without stopping simulation.
        Thread-safe.
        """
        with self._lock:
            self.detectors.update(new_detectors)
        logger.info(
            f"Detectors hot-swapped: {list(new_detectors.keys())}"
        )

    def get_alerts(self) -> pd.DataFrame:
        """Return all logged alerts."""
        with self._lock:
            log = list(self._alert_log)
        return pd.DataFrame(log) if log else pd.DataFrame()

    def get_top_agents(self, n: int = 10) -> pd.DataFrame:
        """Return N agen dengan skor tertinggi lintas semua domain."""
        snap = self.bus.snapshot()
        if not snap:
            return pd.DataFrame()
        rows = []
        for agent_id, domains in snap.items():
            scores_list = list(domains.values())
            rows.append({
                "agent_id":       agent_id,
                "max_score":      max(scores_list) if scores_list else 0.0,
                "domains_active": len(domains),
                "mean_score":     round(np.mean(scores_list), 4) if scores_list else 0.0,
                **{f"score_{d}": s for d, s in domains.items()},
            })
        df = pd.DataFrame(rows)
        return df.sort_values("max_score", ascending=False).head(n).reset_index(drop=True)

    def summary_stats(self) -> dict:
        with self._lock:
            n_alerts = len(self._alert_log)
            n_events = self._event_count
        return {
            "total_events":    n_events,
            "total_alerts":    n_alerts,
            "alert_rate":      round(n_alerts / max(n_events, 1), 4),
            "agents_tracked":  self.bus.agent_count(),
            "domains_active":  list(self.detectors.keys()),
        }