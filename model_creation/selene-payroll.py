"""
Step 1: Data Preparation & Model Training
"""

import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_PATH = "/kaggle/input/datasets/mhiskabee/celine-corp-dataset/simulated_corpus.parquet"
TARGET = "is_corrupt"
ALL_TABULAR_FEATURES = [
    "feat_salary_change_backdated_days",
    "feat_salary_vs_peer_ratio",
    "feat_submitter_rejection_rate",
    "feat_undocumented_spend_pct",
    "feat_night_step_count",
    "feat_conformance_fitness",
    "feat_skipped_steps",
    "feat_failed_auth_count"
]

class PayrollDetector:
    def __init__(self, corrupt_ratio=0.04):
        spw = (1 - corrupt_ratio) / corrupt_ratio
        self.model = LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            verbose=-1, random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X[ALL_TABULAR_FEATURES], y)
        return self

    def score(self, X):
        rule_trigger = X["feat_salary_change_backdated_days"] > 30
        lgbm_probs = self.model.predict_proba(X[ALL_TABULAR_FEATURES])[:, 1]
        return pd.Series(np.where(rule_trigger, 1.0, lgbm_probs), index=X.index)

# Execution
if os.path.exists(DATA_PATH):
    df = pd.read_parquet(DATA_PATH)
    df[ALL_TABULAR_FEATURES] = df[ALL_TABULAR_FEATURES].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df, df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=42
    )

    detector = PayrollDetector().fit(X_train, y_train)
    print("Model trained successfully.")
    
    # Save model
    model_path = "selene-payroll-model.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(detector, f)
        print(f"Successfully saved {type(detector).__name__} to {model_path}")
    except NameError:
        print("Error: 'detector' variable not found. Please ensure the training cell has been executed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
else:
    print(f"Error: File {DATA_PATH} not found.")


# summary and visualization

"""
Step 2: Performance Evaluation & Executive Summary
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc

# Get Predictions
test_scores = detector.score(X_test)
y_pred = (test_scores >= 0.5).astype(int)

# Visualisation Dashboard
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Feature Importance
imp_df = pd.DataFrame({"feature": ALL_TABULAR_FEATURES, "importance": detector.model.feature_importances_}).sort_values("importance", ascending=False)
sns.barplot(x="importance", y="feature", data=imp_df, ax=axes[0, 0], palette="magma")
axes[0, 0].set_title("Payroll Fraud Key Indicators")

# 2. Score Distribution
sns.histplot(test_scores, bins=30, kde=True, ax=axes[0, 1], color="blue")
axes[0, 1].set_title("Risk Score Distribution (Test Set)")

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, test_scores)
axes[1, 1].plot(fpr, tpr, label=f"AUC: {auc(fpr, tpr):.4f}", color="darkorange", lw=2)
axes[1, 1].plot([0,1], [0,1], 'k--')
axes[1, 1].set_title("ROC Performance")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Executive Summary
print("\n" + "="*40)
print("      PAYROLL FRAUD SUMMARY REPORT")
print("="*40)
total_high_risk = (detector.score(df) > 0.5).sum()
print(f"Total Records Analyzed : {len(df):,}")
print(f"High Risk Cases Found  : {total_high_risk} ({total_high_risk/len(df)*100:.2f}%)")
print(f"Model AUC-ROC Score    : {roc_auc_score(y_test, test_scores):.4f}")
print("-"*40)
print(f"Top Indicator: {imp_df.iloc[0]['feature']}")
print("="*40)