"""
train_pipeline.py
=================
Pipeline 1 — Train System (Initial Model Training & Persistence)

Responsibilities:
  - Import all domain detector classes from selene-*.py files
  - Load and split corpus data per domain
  - Train all detectors in parallel
  - Evaluate and log metrics (AUC, precision, recall)
  - Save/load models to/from disk with version suffixes

Usage:
  python train_pipeline.py                          # train and save as 'initial'
  python train_pipeline.py --suffix v2              # save with suffix 'v2'
  python train_pipeline.py --eval-only --suffix v2  # evaluate existing model

Exports for run_pipeline.py:
  DATA_DIR, MODEL_DIR, CORPUS_PATH
  _import_detectors(), _import_simulation()
  train_all_detectors()
  save_detectors(), load_detectors(), load_latest_detectors()
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import threading
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


_HERE = Path(__file__).parent.resolve()

DATA_DIR    = _HERE / "data"
MODEL_DIR   = _HERE / "model_creation" / "model"
CORPUS_PATH = _HERE / "simulated_corpus.parquet"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_corrupt"

DOMAIN_REGISTRY: Dict[str, Tuple[str, str]] = {
    "access":      ("selene-access",      "AccessDetector"),
    "approval":    ("selene-approval",    "ApprovalDetector"),
    "budget":      ("selene-budget",      "BudgetDetector"),
    "contract":    ("selene-contract",    "ContractDetector"),
    "finance":     ("selene-finance",     "FinanceDetector"),
    "payroll":     ("selene-payroll",     "PayrollDetector"),
    "procurement": ("selene-procurement", "ProcurementDetector"),
}




def _load_module_from_file(module_name: str, file_path: Path):
    """Load Python module from file path.
    Needed because filenames contain hyphens (non-importable).
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak bisa load module dari {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_detectors() -> Dict[str, type]:
    """
    Import semua detector class dari file selene-*.py.

    Returns:
        Dict[domain_name, DetectorClass]

    Contoh:
        {"access": AccessDetector, "approval": ApprovalDetector, ...}
    """
    detector_classes: Dict[str, type] = {}

    for domain, (file_stem, class_name) in DOMAIN_REGISTRY.items():
        file_path = _HERE / "model_creation" / f"{file_stem}.py"
        if not file_path.exists():
            logger.warning(f"File tidak ditemukan: {file_path} — domain '{domain}' dilewati")
            continue

        try:
            mod = _load_module_from_file(f"selene_{domain}", file_path)
            cls = getattr(mod, class_name)
            detector_classes[domain] = cls
            logger.debug(f"   ✔ {domain} → {class_name}")
        except Exception as e:
            logger.error(f"Gagal import {class_name} dari {file_path.name}: {e}")

    logger.info(f"📦 {len(detector_classes)} detector class berhasil di-import: "
                f"{list(detector_classes.keys())}")
    return detector_classes


def _import_simulation() -> Tuple[type, dict, None]:
    """Import simulator class and its base configuration.

    Searches for simulator in order:
      1. selene.simulation.CorporateFraudSimulator (official package)
      2. simulation.CorporateFraudSimulator        (local)
      3. CorpusReplaySimulator                     (fallback corpus-based)

    Returns:
        (SimClass, base_config, None)
    """
    # Try importing from available package/module
    for mod_path, cls_name in [
        ("selene.simulation", "CorporateFraudSimulator"),
        ("simulation",        "CorporateFraudSimulator"),
    ]:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            config = getattr(mod, "DEFAULT_CONFIG", _default_sim_config())
            logger.info(f"✅ Simulator ditemukan: {mod_path}.{cls_name}")
            return cls, config, None
        except (ImportError, AttributeError):
            continue

    # Fallback: CorpusReplaySimulator (replay events dari corpus)
    logger.warning(
        "CorporateFraudSimulator tidak ditemukan — "
        "menggunakan CorpusReplaySimulator (replay corpus)"
    )
    return CorpusReplaySimulator, _default_sim_config(), None


def _default_sim_config() -> dict:
    return {
        "simulation": {
            "start_date":    "2024-01-01",
            "events_per_day": 50,
            "corrupt_ratio":  0.04,
        },
        "domains": list(DOMAIN_REGISTRY.keys()),
    }



class CorpusReplaySimulator:
    """
    Simulator fallback yang me-replay events dari corpus yang sudah ada.

    Cara kerja:
    - Pada saat init, load simulated_corpus.parquet
    - Pada saat run(days=N), sample events secara proporsional per hari
    - Setiap event di-append ke self.event_log (memicu ObservableList callback)

    Ini memungkinkan run_pipeline.py berjalan bahkan tanpa
    CorporateFraudSimulator yang sesungguhnya.
    """

    def __init__(self, config: dict):
        self.config     = config
        self.event_log  = []
        self._corpus_df: Optional[pd.DataFrame] = None

        sim_cfg = config.get("simulation", {})
        self._start_date    = pd.Timestamp(sim_cfg.get("start_date", "2024-01-01"))
        self._events_per_day = int(sim_cfg.get("events_per_day", 50))
        self._rng           = np.random.default_rng(seed=42)

        self._load_corpus()

    def _load_corpus(self) -> None:
        if CORPUS_PATH.exists():
            try:
                self._corpus_df = pd.read_parquet(CORPUS_PATH)
                logger.debug(f"   Corpus dimuat: {len(self._corpus_df):,} rows")
            except Exception as e:
                logger.warning(f"   Gagal load corpus: {e} — menggunakan synthetic events")
                self._corpus_df = None
        else:
            self._corpus_df = None

    def run(self, days: int = 730) -> None:
        """Run simulation for `days` days.
        Each event is appended to self.event_log one by one.
        """
        if self._corpus_df is not None and len(self._corpus_df) > 0:
            self._replay_corpus(days)
        else:
            self._generate_synthetic(days)

    def _replay_corpus(self, days: int) -> None:
        """Sample rows dari corpus dan replay sebagai event stream."""
        df = self._corpus_df
        domains = df["domain"].unique().tolist() if "domain" in df.columns else list(DOMAIN_REGISTRY.keys())

        for day_offset in range(days):
            current_date = self._start_date + timedelta(days=day_offset)
            n_events     = max(1, self._rng.poisson(self._events_per_day))

            # Sample acak dari corpus per hari
            sample = df.sample(
                n=min(n_events, len(df)),
                replace=True,
                random_state=int(day_offset),
            )

            for _, row in sample.iterrows():
                event = row.to_dict()
                # Override timestamp agar sesuai dengan hari simulasi
                event["timestamp"] = current_date + timedelta(
                    hours=self._rng.integers(0, 24),
                    minutes=self._rng.integers(0, 60),
                )
                # Pastikan domain & agent_id ada
                if "domain" not in event:
                    event["domain"] = self._rng.choice(domains)
                if "agent_id" not in event:
                    event["agent_id"] = f"agent_{self._rng.integers(1, 200)}"

                self.event_log.append(event)

    def _generate_synthetic(self, days: int) -> None:
        """Generate event sintetik sederhana jika corpus tidak tersedia."""
        domains    = list(DOMAIN_REGISTRY.keys())
        n_agents   = 100
        corrupt_r  = self.config.get("simulation", {}).get("corrupt_ratio", 0.04)

        for day_offset in range(days):
            current_date = self._start_date + timedelta(days=day_offset)
            n_events     = max(1, self._rng.poisson(self._events_per_day))

            for _ in range(n_events):
                domain     = self._rng.choice(domains)
                is_corrupt = self._rng.random() < corrupt_r
                agent_id   = f"agent_{self._rng.integers(1, n_agents + 1)}"

                event = {
                    "domain":      domain,
                    "agent_id":    agent_id,
                    "is_corrupt":  int(is_corrupt),
                    "timestamp":   current_date + timedelta(
                        hours=self._rng.integers(0, 24)
                    ),
                    "action_type": self._rng.choice(
                        ["submit", "approve", "execute", "review"]
                    ),
                }

                # Tambahkan fitur numerik sintetik
                for feat in [
                    "feat_undocumented_spend_pct",
                    "feat_salary_change_backdated_days",
                    "feat_submitter_rejection_rate",
                    "feat_night_step_count",
                    "feat_conformance_fitness",
                    "feat_skipped_steps",
                    "feat_failed_auth_count",
                    "feat_salary_vs_peer_ratio",
                ]:
                    val = self._rng.normal(0.5 if is_corrupt else 0.1, 0.15)
                    event[feat] = float(np.clip(val, 0.0, 1.0))

                self.event_log.append(event)



def _prepare_domain_data(
    df: pd.DataFrame,
    domain: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Filter corpus per domain, split train/test.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if "domain" in df.columns:
        domain_df = df[df["domain"] == domain].copy()
    else:
        domain_df = df.copy()

    if domain_df.empty:
        raise ValueError(f"Tidak ada data untuk domain '{domain}'")

    if TARGET not in domain_df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ada di data domain '{domain}'")

    # Fill NaN pada kolom fitur numerik
    feat_cols = [c for c in domain_df.columns if c.startswith("feat_")]
    domain_df[feat_cols] = domain_df[feat_cols].fillna(0)

    X = domain_df.drop(columns=[TARGET])
    y = domain_df[TARGET].astype(int)

    # Cek cukup sample untuk stratified split
    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        logger.warning(
            f"Domain '{domain}': kelas minoritas < 2 sample, "
            f"split tanpa stratify"
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

    return X_train, X_test, y_train, y_test


def _evaluate_detector(
    detector,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    domain: str,
) -> dict:
    """
    Evaluasi detector: hitung AUC-ROC, AP, dan classification report.
    Return dict metrics.
    """
    metrics = {"domain": domain, "auc_roc": None, "avg_precision": None}
    try:
        scores  = detector.score(X_test)
        y_pred  = (scores >= 0.5).astype(int)

        if y_test.nunique() > 1:
            metrics["auc_roc"]       = round(roc_auc_score(y_test, scores), 4)
            metrics["avg_precision"] = round(average_precision_score(y_test, scores), 4)
        else:
            logger.warning(f"Domain '{domain}': hanya 1 kelas di test set — AUC tidak dihitung")

        logger.info(f"\n{'─'*50}")
        logger.info(f"Evaluasi: {domain.upper()}")
        logger.info(f"{'─'*50}")
        logger.info(
            classification_report(
                y_test, y_pred,
                target_names=["Normal", "Corrupt"],
                zero_division=0,
            )
        )
        if metrics["auc_roc"] is not None:
            logger.info(f"   AUC-ROC        : {metrics['auc_roc']:.4f}")
            logger.info(f"   Avg Precision  : {metrics['avg_precision']:.4f}")

    except Exception as e:
        logger.warning(f"Evaluasi domain '{domain}' gagal: {e}")

    return metrics


def train_all_detectors(
    df: pd.DataFrame,
    detector_classes: Dict[str, type],
    eval_on_test:  bool = True,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Train semua detector pada data yang diberikan.

    Setiap detector di-train pada slice domain-nya sendiri dari df.
    Bisa dijalankan secara paralel (n_jobs > 1) atau sekuensial.

    Args:
        df:               DataFrame korpus (harus punya kolom 'domain' & 'is_corrupt')
        detector_classes: Dict dari _import_detectors()
        eval_on_test:     Jika True, evaluasi di test split dan log metrics
        n_jobs:           Jumlah thread paralel (1 = sekuensial)

    Returns:
        Dict[domain, fitted_detector_instance]
    """
    fitted: Dict[str, Any] = {}
    all_metrics: List[dict] = []
    lock = threading.Lock()

    def _train_one(domain: str, DetectorClass: type) -> None:
        logger.info(f"\n{'='*50}")
        logger.info(f"   🏋️  Training: {domain.upper()}")
        logger.info(f"{'='*50}")

        try:
            X_train, X_test, y_train, y_test = _prepare_domain_data(df, domain)
            logger.info(
                f"   Train: {len(X_train):,} | "
                f"Test: {len(X_test):,} | "
                f"Corrupt: {y_train.sum():,} ({y_train.mean()*100:.1f}%)"
            )

            detector = DetectorClass()
            detector.fit(X_train, y_train)

            metrics = {}
            if eval_on_test:
                metrics = _evaluate_detector(detector, X_test, y_test, domain)

            with lock:
                fitted[domain] = detector
                all_metrics.append(metrics)

            logger.info(f"    {domain} selesai di-train")

        except Exception as e:
            logger.error(f"    Training '{domain}' gagal: {e}")
            import traceback
            traceback.print_exc()

    if n_jobs > 1:
        threads = []
        for domain, cls in detector_classes.items():
            t = threading.Thread(target=_train_one, args=(domain, cls), daemon=True)
            threads.append(t)
            t.start()
            # Batasi konkurensi
            if len(threads) >= n_jobs:
                for t in threads:
                    t.join()
                threads = []
        for t in threads:
            t.join()
    else:
        for domain, cls in detector_classes.items():
            _train_one(domain, cls)

    # Ringkasan metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        logger.info("\n" + "=" * 55)
        logger.info("RINGKASAN TRAINING")
        logger.info("=" * 55)
        logger.info(metrics_df.to_string(index=False))
        logger.info("=" * 55)

    logger.info(f"\n{len(fitted)}/{len(detector_classes)} detector berhasil di-train")
    return fitted




def save_detectors(
    fitted: Dict[str, Any],
    suffix: str = "initial",
) -> Dict[str, Path]:
    """
    Simpan setiap fitted detector ke file joblib terpisah.

    Format nama file: model/{domain}_{suffix}.pkl

    Args:
        fitted: Dict dari train_all_detectors()
        suffix: Versi / label (contoh: 'initial', 'cycle1', 'v2')

    Returns:
        Dict[domain, saved_path]
    """
    saved_paths: Dict[str, Path] = {}

    for domain, detector in fitted.items():
        fname = MODEL_DIR / f"{domain}_{suffix}.pkl"
        try:
            joblib.dump(detector, fname, compress=3)
            saved_paths[domain] = fname
            logger.info(f"   💾 Saved: {fname.name}")
        except Exception as e:
            logger.error(f"   ❌ Gagal simpan '{domain}': {e}")

    logger.info(f"✅ {len(saved_paths)} model disimpan → {MODEL_DIR}")
    return saved_paths


def load_detectors(
    suffix: str = "initial",
    domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Muat detector dari disk berdasarkan suffix.

    Args:
        suffix:  Versi model (contoh: 'initial', 'cycle2')
        domains: Daftar domain yang dimuat (None = semua yang ada)

    Returns:
        Dict[domain, fitted_detector_instance]
    """
    target_domains = domains or list(DOMAIN_REGISTRY.keys())
    fitted: Dict[str, Any] = {}

    for domain in target_domains:
        fname = MODEL_DIR / f"{domain}_{suffix}.pkl"
        if not fname.exists():
            logger.warning(f"   ⚠️  Model tidak ditemukan: {fname.name}")
            continue
        try:
            detector = joblib.load(fname)
            fitted[domain] = detector
            logger.info(f"   📦 Loaded: {fname.name}")
        except Exception as e:
            logger.error(f"   ❌ Gagal load '{domain}_{suffix}': {e}")

    logger.info(f"✅ {len(fitted)} model dimuat (suffix='{suffix}')")
    return fitted


def load_latest_detectors(
    domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Muat detector versi terbaru per domain secara otomatis.

    Logika: untuk setiap domain, pilih file *.pkl yang paling baru diubah.

    Args:
        domains: Daftar domain yang dimuat (None = semua)

    Returns:
        Dict[domain, fitted_detector_instance]
    """
    target_domains = domains or list(DOMAIN_REGISTRY.keys())
    fitted: Dict[str, Any] = {}

    for domain in target_domains:
        candidates = sorted(
            MODEL_DIR.glob(f"{domain}_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            logger.warning(f"   ⚠️  Tidak ada model untuk domain '{domain}'")
            continue

        latest = candidates[0]
        try:
            detector = joblib.load(latest)
            fitted[domain] = detector
            logger.info(f"   📦 Latest loaded: {latest.name}")
        except Exception as e:
            logger.error(f"   ❌ Gagal load {latest.name}: {e}")

    logger.info(f"✅ {len(fitted)} model (latest) dimuat")
    return fitted




def train_pipeline(
    corpus_path: Optional[Path] = None,
    suffix:      str             = "initial",
    eval_on_test: bool           = True,
    n_jobs:      int             = 1,
) -> Dict[str, Any]:
    """
    Entry point Pipeline 1 — Train System.

    1. Load corpus dari parquet
    2. Import semua detector classes
    3. Train semua detector
    4. Simpan ke disk

    Args:
        corpus_path:  Path ke corpus parquet (default: CORPUS_PATH)
        suffix:       Suffix untuk penamaan file model
        eval_on_test: Evaluasi setelah training
        n_jobs:       Paralel training

    Returns:
        Dict[domain, fitted_detector]
    """
    start_time  = datetime.now()
    corpus_path = corpus_path or CORPUS_PATH

    logger.info("\n" + "=" * 55)
    logger.info("🚀 SELENE — TRAIN PIPELINE")
    logger.info(f"   Corpus : {corpus_path}")
    logger.info(f"   Suffix : {suffix}")
    logger.info("=" * 55)

    # 1. Load corpus
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus tidak ditemukan: {corpus_path}\n"
            f"Jalankan dataset generation terlebih dahulu."
        )

    logger.info(f"\n📂 Memuat corpus dari {corpus_path.name}...")
    df = pd.read_parquet(corpus_path)
    logger.info(f"   {len(df):,} rows | {len(df.columns)} kolom")

    if "domain" in df.columns:
        logger.info(f"   Domain distribution:\n{df['domain'].value_counts().to_string()}")
    if TARGET in df.columns:
        corrupt_pct = df[TARGET].mean() * 100
        logger.info(f"   Corrupt rate: {corrupt_pct:.2f}%")

    # 2. Import detector classes
    logger.info("\n📦 Mengimpor detector classes...")
    detector_classes = _import_detectors()

    if not detector_classes:
        raise RuntimeError("Tidak ada detector class yang berhasil diimport!")

    # 3. Train
    fitted = train_all_detectors(
        df,
        detector_classes,
        eval_on_test=eval_on_test,
        n_jobs=n_jobs,
    )

    # 4. Save
    logger.info(f"\n💾 Menyimpan model dengan suffix='{suffix}'...")
    save_detectors(fitted, suffix=suffix)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n✨ Train pipeline selesai dalam {elapsed:.1f}s")
    return fitted


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Selene Train Pipeline — training awal semua domain detector"
    )
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_PATH,
        help=f"Path ke corpus parquet (default: {CORPUS_PATH})"
    )
    parser.add_argument(
        "--suffix", default="initial",
        help="Suffix untuk penamaan file model (default: 'initial')"
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluasi setelah training"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Jumlah thread paralel untuk training (default: 1)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Hanya evaluasi model yang sudah ada, tanpa retrain"
    )
    args = parser.parse_args()

    if args.eval_only:
        # Load model yang ada, evaluasi di corpus
        logger.info("📊 Mode evaluasi saja (--eval-only)")
        fitted = load_detectors(suffix=args.suffix)
        if not fitted:
            logger.error("Tidak ada model ditemukan. Jalankan training terlebih dahulu.")
            sys.exit(1)

        df = pd.read_parquet(args.corpus)
        for domain, detector in fitted.items():
            try:
                _, X_test, _, y_test = _prepare_domain_data(df, domain)
                _evaluate_detector(detector, X_test, y_test, domain)
            except Exception as e:
                logger.error(f"Evaluasi '{domain}' gagal: {e}")
    else:
        train_pipeline(
            corpus_path=args.corpus,
            suffix=args.suffix,
            eval_on_test=not args.no_eval,
            n_jobs=args.n_jobs,
        )