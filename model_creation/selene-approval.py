"""
Domain: Approval - Process Conformance Detection
Records: 1,002 | Corrupt: 145 (14.5%) | Features: 43 | Mechanisms: 3
Models: Process Mining (PM4Py conformance) + Graph cycle detection (NetworkX) + LightGBM
"""

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Feature Groups
FEATURES_CIRCULAR = [
    "feat_circular_approval_flag",
    "feat_approver_requestor_cooccurrence",
    "feat_actor_count_distinct",
]

FEATURES_BYPASS = [
    "feat_skipped_steps",
    "feat_execution_before_approval_flag",
    "feat_conformance_fitness",
    "feat_duration_percentile",
]

FEATURES_RUBBER = [
    "feat_approver_avg_review_time",
    "feat_approver_rejection_rate",
    "feat_night_step_count",
    "feat_total_cycle_time_hours",
    "feat_same_actor_consecutive_flag",
]

ALL_TABULAR_FEATURES = FEATURES_CIRCULAR + FEATURES_BYPASS + FEATURES_RUBBER

APPROVER_COL   = "feat_approver_id"
REQUESTOR_COL  = "feat_submitter_id"
ACTIVITY_SEQ   = "feat_activity_sequence"

TARGET = "is_corrupt"

IDEAL_PROCESS = [
    "submit",
    "review_l1",
    "review_l2",
    "approve",
    "execute",
    "close",
]

class ProcessConformanceScorer:
    def __init__(self, ideal_process: list[str] = None):
        self.ideal_process = ideal_process or IDEAL_PROCESS
        self._ideal_set = set(self.ideal_process)
        self._ideal_order = {step: i for i, step in enumerate(self.ideal_process)}

    def _parse_sequence(self, seq_raw) -> list[str]:
        if isinstance(seq_raw, list):
            return [str(s) for s in seq_raw]
        if isinstance(seq_raw, str):
            return [s.strip().strip("'\"") for s in seq_raw.strip("[]").split(",") if s.strip()]
        return []

    def fitness_score(self, sequence_raw) -> float:
        seq = self._parse_sequence(sequence_raw)
        if not seq: return 0.0
        seq_set = set(seq)
        missing = self._ideal_set - seq_set
        actual_ideal = [s for s in seq if s in self._ideal_set]
        inversions = 0
        for i in range(len(actual_ideal)):
            for j in range(i + 1, len(actual_ideal)):
                if self._ideal_order.get(actual_ideal[i], 999) > self._ideal_order.get(actual_ideal[j], 999):
                    inversions += 1
        max_inv = len(actual_ideal) * (len(actual_ideal) - 1) / 2 + 1
        fitness = 1.0 - (0.5 * (len(missing) / len(self.ideal_process)) + 0.3 * (inversions / max_inv))
        return max(0.0, fitness)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        if ACTIVITY_SEQ not in df.columns: return df
        df = df.copy()
        df["feat_conformance_fitness"] = df[ACTIVITY_SEQ].map(self.fitness_score)
        df["feat_skipped_steps"] = df[ACTIVITY_SEQ].map(lambda x: len(self._ideal_set - set(self._parse_sequence(x))))
        df["feat_execution_before_approval_flag"] = df[ACTIVITY_SEQ].map(lambda x: int("execute" in x and "approve" in x and x.index("execute") < x.index("approve")))
        return df

class CircularApprovalDetector:
    def __init__(self):
        self._circular_pairs = set()
        self._graph = nx.DiGraph()

    def fit(self, df: pd.DataFrame) -> "CircularApprovalDetector":
        for _, row in df.iterrows():
            a, r = str(row.get(APPROVER_COL, "")), str(row.get(REQUESTOR_COL, ""))
            if a and r and a != r:
                self._graph.add_edge(a, r)
        for cycle in nx.simple_cycles(self._graph):
            if len(cycle) >= 2:
                for i in range(len(cycle)):
                    self._circular_pairs.add((cycle[i], cycle[(i + 1) % len(cycle)]))
        return self

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["feat_circular_approval_flag"] = df.apply(lambda r: int((str(r.get(APPROVER_COL)), str(r.get(REQUESTOR_COL))) in self._circular_pairs), axis=1)
        return df

class ApprovalMLScorer:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1
        )
        self._features = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ApprovalMLScorer":
        self._features = [f for f in X.columns if f.startswith("feat_") and X[f].dtype in [np.float64, np.int64, np.int32, "Int8"]]
        self.model.fit(X[self._features].astype(float), y.astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self._features].astype(float))[:, 1]

class ApprovalDetector:
    def __init__(self):
        self.process_scorer = ProcessConformanceScorer()
        self.circular_detect = CircularApprovalDetector()
        self.ml_scorer = ApprovalMLScorer()

    def _enrich(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.process_scorer.enrich(X)
        X = self.circular_detect.enrich(X)
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ApprovalDetector":
        self.circular_detect.fit(X)
        X_enriched = self._enrich(X)
        self.ml_scorer.fit(X_enriched, y)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        X_enriched = self._enrich(X)
        return pd.Series(self.ml_scorer.predict_proba(X_enriched), index=X.index, name="approval_risk_score")

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        return (self.score(X) >= threshold).astype(int).rename("is_corrupt_pred")

def train(df: pd.DataFrame, test_size: float = 0.2) -> ApprovalDetector:
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    detector = ApprovalDetector().fit(X_train, y_train)
    y_pred = detector.predict(X_test)
    print("=" * 55)
    print("  Approval Conformance Detection (LightGBM) Report")
    print("=" * 55)
    print(classification_report(y_test.astype(int), y_pred))
    return detector

# Load the dataset
path = '/content/simulated_corpus.parquet'
df_corpus = pd.read_parquet(path)

# Data Cleaning
critical_cols = [ACTIVITY_SEQ, TARGET]
df_corpus = df_corpus.dropna(subset=critical_cols)

# Run the training process with restored feature
detector_model = train(df_corpus)

# Save model
model_path = "selene-approval-model.pkl"
try:
    with open(model_path, "wb") as f:
        pickle.dump(detector_model, f)
    print(f"Successfully saved {type(detector_model).__name__} to {model_path}")
except NameError:
    print("Error: 'detector_model' variable not found. Please ensure the training cell has been executed.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Display risk scores sample
risk_scores = detector_model.score(df_corpus.drop(columns=[TARGET]))
display(risk_scores.to_frame().head())


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc

# Prepare evaluation data
X_eval = df_corpus.drop(columns=[TARGET])
y_eval = df_corpus[TARGET]

# Get Scores and Predictions
test_scores = detector_model.score(X_eval)
y_pred = detector_model.predict(X_eval)

# Visualisation Dashboard
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Feature Importance
model_internal = detector_model.ml_scorer.model
features_used = detector_model.ml_scorer._features
importance = model_internal.feature_importances_

imp_df = pd.DataFrame({"feature": features_used, "importance": importance}).sort_values("importance", ascending=False)
sns.barplot(x="importance", y="feature", data=imp_df.head(15), ax=axes[0, 0], palette="magma")
axes[0, 0].set_title("Approval Risk Key Indicators (Restored Cycle Time)")

# 2. Score Distribution
sns.histplot(test_scores, bins=30, kde=True, ax=axes[0, 1], color="blue")
axes[0, 1].set_title("Risk Score Distribution")

# 3. Confusion Matrix
cm = confusion_matrix(y_eval.astype(int), y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix")

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_eval.astype(int), test_scores)
axes[1, 1].plot(fpr, tpr, label=f"AUC: {auc(fpr, tpr):.4f}", color="darkorange", lw=2)
axes[1, 1].plot([0,1], [0,1], 'k--')
axes[1, 1].set_title("ROC Performance")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*40)
print("      APPROVAL CONFORMANCE SUMMARY")
print("="*40)
print(f"Total Records Analyzed : {len(df_corpus):,}")
print(f"Model AUC-ROC Score    : {roc_auc_score(y_eval.astype(int), test_scores):.4f}")
if not imp_df.empty:
    print(f"Top Indicator: {imp_df.iloc[0]['feature']}")
print("="*40)