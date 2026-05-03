"""
Domain: Finance — Expense Claim Detection
Records: ~92k | Features: 174 | Models: LightGBM + Isolation Forest
"""

import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────────────
# Feature Configuration
# ───────────────────────────────────────────────────────────────────────────

ALL_TABULAR_FEATURES = [
    "feat_undocumented_spend_pct", "feat_salary_change_backdated_days",
    "feat_submitter_rejection_rate", "feat_night_step_count",
    "feat_conformance_fitness", "feat_skipped_steps",
    "feat_failed_auth_count", "feat_salary_vs_peer_ratio"
]
TARGET = "is_corrupt"

class FinanceLightGBM:
    def __init__(self, corrupt_ratio=0.04):
        spw = (1 - corrupt_ratio) / corrupt_ratio
        self.model = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            metric="average_precision", verbose=-1, random_state=42
        )
        self.features = ALL_TABULAR_FEATURES
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.features])[:, 1]

    def feature_importance(self):
        return pd.DataFrame({"feature": self.features, "importance": self.model.feature_importances_}).sort_values("importance", ascending=False)

class FinanceAnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        self.features = ALL_TABULAR_FEATURES

    def fit(self, X, y):
        X_normal = X[y == 0][self.features].fillna(0)
        self.scaler.fit(X_normal)
        self.model.fit(self.scaler.transform(X_normal))
        return self

    def anomaly_score(self, X):
        X_scaled = self.scaler.transform(X[self.features].fillna(0))
        raw = self.model.decision_function(X_scaled)
        return 1 / (1 + np.exp(raw * 5))

class FinanceDetector:
    def __init__(self):
        self.lgbm = FinanceLightGBM()
        self.iso_forest = FinanceAnomalyDetector()

    def fit(self, X, y):
        self.lgbm.fit(X, y)
        self.iso_forest.fit(X, y)
        return self

    def score(self, X):
        lgbm_score = self.lgbm.predict_proba(X)
        if_score = self.iso_forest.anomaly_score(X)
        return pd.Series(0.65 * lgbm_score + 0.35 * if_score, index=X.index)

# ──────────────────────────────────────────────
# Main Execution with Parquet Data
# ──────────────────────────────────────────────

DATA_PATH = "/kaggle/input/datasets/mhiskabee/celine-corp-dataset/simulated_corpus.parquet"

if os.path.exists(DATA_PATH):
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Pre-processing
    df[ALL_TABULAR_FEATURES] = df[ALL_TABULAR_FEATURES].fillna(0)
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training Hybrid Detector (LGBM + Anomaly)... ")
    detector = FinanceDetector()
    detector.fit(X_train, y_train)
    
    # Save model
    model_path = "selene-finance-model.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(detector, f)
        print(f"Successfully saved {type(detector).__name__} to {model_path}")
    except NameError:
        print("Error: 'detector' variable not found. Please ensure the training cell has been executed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Evaluation
    test_scores = detector.score(X_test)
    y_pred = (test_scores >= 0.5).astype(int)

    print("=" * 55)
    print("  Finance Detection — Evaluation Report")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Corrupt"]))
    print(f"  ROC-AUC : {roc_auc_score(y_test, test_scores):.4f}")
    print("=" * 55)

    # Global Scoring
    df["risk_score"] = detector.score(df)
    print("\nScoring complete. Top 5 highest risk transactions:")
    display(df[['agent_id', 'risk_score'] + ALL_TABULAR_FEATURES].sort_values("risk_score", ascending=False).head())
else:
    print(f"Error: File {DATA_PATH} not found.")

# summary and visualization

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set plotting style
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Feature Importance (from LightGBM layer)
imp_df = detector.lgbm.feature_importance()
sns.barplot(x="importance", y="feature", data=imp_df.head(10), ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title("Top Corruption Indicators (LightGBM)")

# 2. Risk Score Distribution
sns.histplot(df, x="risk_score", hue=TARGET, element="step", stat="density", common_norm=False, ax=axes[0, 1])
axes[0, 1].set_title("Distribution of Risk Scores by Label")

# 3. Confusion Matrix (on Test Set)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, test_scores)
roc_auc = auc(fpr, tpr)
axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1, 1].set_title("Receiver Operating Characteristic (ROC)")
axes[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.show()