"""
run_pipeline.py
===============
Pipeline 2 — Run System (Live Detection + Cross-Domain Interaction)

Features:
  - Simulation continues from day 731+ (after training ends on day 730)
  - Each simulated event is immediately scored by the model
  - Models interact via RiskBus + CrossDomainOrchestrator
  - Every 360 simulation days: retrain all models with cumulative data
  - New models hot-swap without stopping simulation

Usage:
  python run_pipeline.py --start-day 730 --run-days 360
  python run_pipeline.py --start-day 730 --run-days 720 --retrain-every 360
  python run_pipeline.py --start-day 730 --run-days 360 --model-suffix initial
"""

from __future__ import annotations

import argparse
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from selene.orchestrator import CrossDomainOrchestrator

from train_pipeline import (
    DATA_DIR,
    MODEL_DIR,
    CORPUS_PATH,
    _import_detectors,
    _import_simulation,
    load_detectors,
    load_latest_detectors,
    save_detectors,
    train_all_detectors,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALERT_DIR = DATA_DIR / "alerts"
ALERT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ObservableList - key to "model embedded in simulation"
# ─────────────────────────────────────────────────────────────────────────────
class ObservableList(list):
    """List wrapper that invokes a callback on each append.
    Enables models to run synchronously within the simulation loop without
    modifying simulator code.
    """

    def __init__(self, callback: Optional[Callable] = None):
        super().__init__()
        self.callback = callback
        self.detection_results: List[dict] = []

    def append(self, item: dict) -> None:
        super().append(item)
        if self.callback is not None:
            try:
                result = self.callback(item)
                self.detection_results.append(result)
                item["rt_score"]       = result.get("final_score", 0.0)
                item["rt_base_score"]  = result.get("base_score", 0.0)
                item["rt_cross_boost"] = result.get("cross_boost", 0.0)
                item["rt_alert"]       = result.get("alert", False)
                item["rt_consensus"]   = result.get("consensus", {}).get(
                    "consensus_count", 0
                )
            except Exception as e:
                logger.debug(f"ObservableList callback error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# RetrainWorker - background retraining every 360 simulation days
# ─────────────────────────────────────────────────────────────────────────────
class RetrainWorker:
    """Manages periodic retraining.

    Strategy:
    - Retraining uses CUMULATIVE DATA (initial corpus + all new data)
    - After completion, new models hot-swap into orchestrator
    - Orchestrator continues running during retrain (no downtime)
    - Old models remain active until new models are ready
    """

    def __init__(
        self,
        orchestrator: CrossDomainOrchestrator,
        detector_classes: Dict[str, type],
    ):
        self.orchestrator = orchestrator
        self.detector_classes = detector_classes
        self._lock = threading.Lock()
        self._retrain_count = 0

    def retrain(self, combined_df: pd.DataFrame, cycle: int) -> None:
        """Perform retrain with cumulative data.
        Thread-safe: can be called from separate thread.

        Args:
            combined_df: initial corpus + all accumulated new data
            cycle: retrain cycle number (for file naming)
        """
        logger.info(f"\n{'─'*55}")
        logger.info(f"RETRAIN CYCLE {cycle} — {len(combined_df):,} rows")
        logger.info(f"{'─'*55}")

        with self._lock:
            try:
                new_fitted = train_all_detectors(
                    combined_df,
                    self.detector_classes,
                    eval_on_test=True,
                )
                save_detectors(new_fitted, suffix=f"cycle{cycle}")
                # Hot-swap: new models replace old ones in orchestrator
                self.orchestrator.update_detectors(new_fitted)
                self._retrain_count += 1
                logger.info(f"Cycle {cycle} complete — {len(new_fitted)} models hot-swapped")
            except Exception as e:
                logger.error(f"Retrain cycle {cycle} failed: {e}")
                import traceback
                traceback.print_exc()

    @property
    def retrain_count(self) -> int:
        return self._retrain_count


# ─────────────────────────────────────────────────────────────────────────────
# StreamingSimulator — main run pipeline
# ─────────────────────────────────────────────────────────────────────────────
class StreamingSimulator:
    """Run extended simulation with embedded real-time detection.

    Embedding mechanism:
    1. CorporateFraudSimulator uses self.event_log.append(event) for each event
    2. We inject ObservableList as event_log
    3. ObservableList calls orchestrator.process_event(event) on each append
    4. Scores are returned and enriched into event dict in-place
    5. Simulator doesn't know model is running — simulation proceeds normally

    Retraining cycle:
    - Every retrain_every simulation days, pause briefly for retrain
    - Retraining uses cumulative data (initial corpus + all new events)
    - After retrain completes, new models hot-swap into orchestrator
    - Simulation continues with fresher models

    Inter-model interaction:
    - Each score is sent to RiskBus via orchestrator.process_event()
    - Next detector of same agent gets signal boost from other domains
      (cross-domain amplification)
    - If >=2 domains flag the same agent -> consensus alert
    """

    def __init__(
        self,
        orchestrator: CrossDomainOrchestrator,
        retrain_worker: RetrainWorker,
        retrain_every: int = 360,
        alert_threshold: float = 0.5,
    ):
        self.orchestrator    = orchestrator
        self.retrain_worker  = retrain_worker
        self.retrain_every   = retrain_every
        self.alert_threshold = alert_threshold

        # Accumulate all raw events (for retraining)
        self._raw_events: List[dict]   = []
        # Accumulate events with scores (for final output)
        self._scored_events: List[dict] = []

        self.stats = {
            "total_events": 0,
            "total_alerts": 0,
            "retrains":     0,
            "days_run":     0,
        }

    # ── Event callback (dipanggil dari ObservableList) ─────────────────────
    def _on_event(self, event: dict) -> dict:
        """Called synchronously for each event generated by simulator.
        Runs detection and updates statistics.
        """
        self._raw_events.append(event)
        self.stats["total_events"] += 1

        result = self.orchestrator.process_event(event)

        if result.get("alert"):
            self.stats["total_alerts"] += 1

        # Combine event + detection result for output
        combined = {**event, **{
            f"det_{k}": v for k, v in result.items()
            if k not in ("consensus",)
        }}
        combined["det_consensus_count"]   = result.get("consensus", {}).get("consensus_count", 0)
        combined["det_domains_flagged"]   = ",".join(
            result.get("consensus", {}).get("domains_flagged", [])
        )
        self._scored_events.append(combined)

        return result

    # ── Main run ──────────────────────────────────────────────────────────────
    def run(
        self,
        start_day: int       = 720,
        run_days: int        = 720,
        initial_corpus_path: Optional[Path] = None,
        config_override: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Continue simulation from start_day for run_days.

        Args:
            start_day: Starting day (typically 730 = after training)
            run_days: Number of simulation days to run
            initial_corpus_path: Path to corpus for cumulative retraining
            config_override: Override simulator CONFIG

        Returns:
            DataFrame with events and detection scores
        """
        logger.info("\n" + "=" * 55)
        logger.info("SELENE - RUN PIPELINE")
        logger.info(f"   Start day : {start_day}")
        logger.info(f"   Run days  : {run_days}")
        logger.info(f"   Retrain   : every {self.retrain_every} days")
        logger.info("=" * 55)

        # Load initial corpus for cumulative retraining
        base_df: Optional[pd.DataFrame] = None
        if initial_corpus_path and initial_corpus_path.exists():
            logger.info(f"Loading initial corpus from {initial_corpus_path}")
            base_df = pd.read_parquet(initial_corpus_path)
            logger.info(f"   {len(base_df):,} rows loaded")

        # Import simulator
        SimClass, base_config, _ = _import_simulation()
        config = config_override or base_config

        # Adjust start date simulasi
        from datetime import timedelta
        import pandas as _pd
        orig_start = _pd.Timestamp(
            config.get("simulation", {}).get("start_date", "2024-01-01")
        )
        new_start = orig_start + timedelta(days=start_day)

        config_run = dict(config)
        if "simulation" not in config_run:
            config_run["simulation"] = {}
        config_run["simulation"]["start_date"] = new_start.strftime("%Y-%m-%d")

        # Retrain cycle counter
        retrain_cycle = 1
        days_run = 0

        while days_run < run_days:
            block = min(self.retrain_every, run_days - days_run)
            day_start = start_day + days_run
            day_end   = day_start + block - 1

            logger.info(f"\nDay block {day_start} -> {day_end} ({block} days)")

            # Create new simulator for this block
            config_block = dict(config_run)
            config_block["simulation"] = dict(config_run.get("simulation", {}))
            block_start = orig_start + timedelta(days=days_run + start_day)
            config_block["simulation"]["start_date"] = block_start.strftime("%Y-%m-%d")

            sim = SimClass(config_block)

            # ── KUNCI: inject ObservableList sebagai event_log ──────────────
            # Setiap sim.event_log.append(event) → _on_event(event) → model score
            sim.event_log = ObservableList(callback=self._on_event)

            logger.info("🎮 Simulasi berjalan dengan deteksi real-time tertanam...")
            sim.run(days=block)

            self.stats["days_run"] += block
            days_run += block

            # Log progress blok
            n_new = len(sim.event_log)
            logger.info(
                f"   {n_new:,} events diproses | "
                f"alerts: {self.stats['total_alerts']:,} total"
            )

            # Flush alerts ke disk
            self._flush_alerts(cycle=retrain_cycle)

            # ── Retrain dengan data kumulatif ──────────────────────────────
            new_events_df = pd.DataFrame(list(sim.event_log))

            if base_df is not None:
                combined_df = pd.concat([base_df, new_events_df], ignore_index=True)
            else:
                combined_df = new_events_df

            logger.info(f"\n🔁 Memulai retrain cycle {retrain_cycle} "
                        f"({len(combined_df):,} rows kumulatif)...")

            # Retrain di thread utama (bisa diubah ke background thread jika perlu)
            self.retrain_worker.retrain(combined_df, cycle=retrain_cycle)
            self.stats["retrains"] += 1
            retrain_cycle += 1

            # Akumulasi corpus untuk siklus berikutnya
            base_df = combined_df

        # Final flush + summary
        self._flush_alerts(cycle="final")
        self._print_summary()

        return self._build_output()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _flush_alerts(self, cycle) -> None:
        """Simpan alert ke CSV."""
        alerts_df = self.orchestrator.get_alerts()
        if alerts_df.empty:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ALERT_DIR / f"alerts_cycle{cycle}_{ts}.csv"
        alerts_df.to_csv(path, index=False)
        logger.info(f"💾 {len(alerts_df)} alerts → {path.name}")

    def _build_output(self) -> pd.DataFrame:
        """Bangun DataFrame output dari semua scored events."""
        if not self._scored_events:
            return pd.DataFrame()
        df = pd.DataFrame(self._scored_events)
        # Konversi timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def _print_summary(self) -> None:
        stats = self.orchestrator.summary_stats()
        logger.info("\n" + "=" * 55)
        logger.info("📊 SELENE RUN PIPELINE — RINGKASAN")
        logger.info("=" * 55)
        logger.info(f"   Hari dijalankan    : {self.stats['days_run']}")
        logger.info(f"   Total events       : {self.stats['total_events']:,}")
        logger.info(f"   Total alerts       : {self.stats['total_alerts']:,}")
        logger.info(f"   Alert rate         : {stats['alert_rate']*100:.2f}%")
        logger.info(f"   Agen dipantau      : {stats['agents_tracked']:,}")
        logger.info(f"   Retrain cycles     : {self.stats['retrains']}")

        top = self.orchestrator.get_top_agents(n=5)
        if not top.empty:
            logger.info("\n🎯 TOP 5 HIGH-RISK AGENTS:")
            logger.info(
                top[["agent_id", "max_score", "domains_active", "mean_score"]].to_string(index=False)
            )
        logger.info("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Main Run Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    start_day:     int  = 720,
    run_days:      int  = 720,
    model_suffix:  str  = "initial",
    retrain_every: int  = 360,
    use_latest:    bool = False,
) -> pd.DataFrame:
    """
    Entry point Pipeline 2 — Run System.

    Args:
        start_day:     Hari ke berapa simulasi dilanjutkan (umumnya 720)
        run_days:      Berapa hari yang akan dijalankan
        model_suffix:  Suffix model yang dimuat (default: "initial")
        retrain_every: Interval retrain dalam hari simulasi
        use_latest:    Jika True, muat model versi terbaru secara otomatis

    Returns:
        DataFrame event + skor deteksi real-time
    """
    start_time = datetime.now()

    # 1. Load detectors
    logger.info("📦 Memuat detector models...")
    if use_latest:
        fitted = load_latest_detectors()
    else:
        fitted = load_detectors(suffix=model_suffix)

    # 2. Setup orchestrator + retrain worker
    orchestrator     = CrossDomainOrchestrator(detectors=fitted)
    detector_classes = _import_detectors()
    retrain_worker   = RetrainWorker(orchestrator, detector_classes)

    # 3. Setup streaming simulator
    streamer = StreamingSimulator(
        orchestrator=orchestrator,
        retrain_worker=retrain_worker,
        retrain_every=retrain_every,
    )

    # 4. Run!
    result_df = streamer.run(
        start_day=start_day,
        run_days=run_days,
        initial_corpus_path=CORPUS_PATH,
    )

    # 5. Simpan hasil lengkap
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"run_results_{ts}.parquet"
    if not result_df.empty:
        result_df.to_parquet(out_path, index=False, compression="zstd")
        logger.info(f"\n✨ Hasil disimpan → {out_path}")
        logger.info(f"   {len(result_df):,} rows | {len(result_df.columns)} kolom")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"   Waktu total: {elapsed:.1f}s")

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Selene Run Pipeline — simulasi lanjutan dengan deteksi real-time"
    )
    parser.add_argument(
        "--start-day", type=int, default=730,
        help="Starting day for continued simulation (default: 730)"
    )
    parser.add_argument(
        "--run-days", type=int, default=360,
        help="Number of simulation days to run (default: 360)"
    )
    parser.add_argument(
        "--model-suffix", default="initial",
        help="Suffix of model to load (default: 'initial')"
    )
    parser.add_argument(
        "--retrain-every", type=int, default=360,
        help="Retraining interval in simulation days (default: 360)"
    )
    parser.add_argument(
        "--use-latest", action="store_true",
        help="Load latest model version automatically"
    )
    args = parser.parse_args()

    run_pipeline(
        start_day=args.start_day,
        run_days=args.run_days,
        model_suffix=args.model_suffix,
        retrain_every=args.retrain_every,
        use_latest=args.use_latest,
    )