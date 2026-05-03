"""
Domain: Budget — Spending Pattern Anomaly
Records: 1,091 | Features: 17 Budget Specific
Models: Hard rules + Time-series anomaly + SMOTE + LightGBM
"""

import pickle
import numpy as np
import pandas as pd
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Feature Groups
FEATURES_SPLITTING = [
    "feat_max_sub_budget_below_threshold",
    "feat_sub_budget_created_same_day",
    "feat_sub_budget_count",
]

FEATURES_YEAREND = [
    "feat_q4_spend_vs_q1q2q3_ratio",
    "feat_spend_in_last_30d_pct",
    "feat_spend_spike_count",
    "feat_monthly_spend_cv",
]

FEATURES_FICTITIOUS = [
    "feat_undocumented_spend_pct",
    "feat_outcome_vs_spend_ratio",
    "feat_output_achievement_pct",
    "feat_spend_per_output_unit",
]

ALL_TABULAR_FEATURES = FEATURES_SPLITTING + FEATURES_YEAREND + FEATURES_FICTITIOUS
TARGET = "is_corrupt"

class BudgetHardRules:
    def __init__(self, spend_last30d_threshold=0.3, q4_ratio_threshold=1.5, undoc_spend_threshold=0.15, sub_budget_threshold=0.5):
        self.spend_last30d_threshold = spend_last30d_threshold
        self.q4_ratio_threshold      = q4_ratio_threshold
        self.undoc_spend_threshold   = undoc_spend_threshold
        self.sub_budget_threshold    = sub_budget_threshold

    def flag(self, X):
        yearend_rule = (X.get("feat_spend_in_last_30d_pct", pd.Series(0, index=X.index)) > self.spend_last30d_threshold) & (X.get("feat_q4_spend_vs_q1q2q3_ratio", pd.Series(0, index=X.index)) > self.q4_ratio_threshold)
        fictitious_rule = X.get("feat_undocumented_spend_pct", pd.Series(0, index=X.index)) > self.undoc_spend_threshold
        splitting_rule = X.get("feat_max_sub_budget_below_threshold", pd.Series(0, index=X.index)) > self.sub_budget_threshold
        return (yearend_rule | fictitious_rule | splitting_rule).rename("hard_rule_flag")

class BudgetSMOTE:
    TARGET_CORRUPT = 250
    TARGET_NORMAL  = 250
    def __init__(self, random_state=42): self.rng = np.random.RandomState(random_state)
    def _interpolate(self, X_minority, k=5):
        n, d = X_minority.shape
        synthetic = []
        X_m = X_minority.astype(np.float64)
        while len(synthetic) < self.TARGET_CORRUPT - n:
            idx = self.rng.randint(0, n)
            diffs = X_m - X_m[idx]
            dists = np.linalg.norm(diffs, axis=1)
            dists[idx] = np.inf
            k_neighbors = np.argsort(dists)[:k]
            neighbor_idx = self.rng.choice(k_neighbors)
            alpha = self.rng.rand()
            synthetic.append(X_m[idx] + alpha * (X_m[neighbor_idx] - X_m[idx]))
        return np.array(synthetic)

    def fit_resample(self, X, y):
        X_arr = X[ALL_TABULAR_FEATURES].fillna(0).astype(np.float64).values
        y_arr = y.values.astype(int)
        X_corrupt, X_normal = X_arr[y_arr == 1], X_arr[y_arr == 0]
        if len(X_corrupt) > 1:
            interp = self._interpolate(X_corrupt)
            noise = self.rng.randn(*(len(interp), X_corrupt.shape[1])) * 0.01
            X_corrupt_aug = np.vstack([X_corrupt, interp + noise])
        else: X_corrupt_aug = X_corrupt
        
        if len(X_normal) > 0:
             X_normal_aug = X_normal[self.rng.choice(len(X_normal), self.TARGET_NORMAL, replace=True)]
        else: X_normal_aug = X_normal
        
        X_final = np.vstack([X_corrupt_aug[:self.TARGET_CORRUPT], X_normal_aug])
        y_final = np.hstack([np.ones(len(X_corrupt_aug[:self.TARGET_CORRUPT]), dtype=int), np.zeros(len(X_normal_aug), dtype=int)])
        shf = self.rng.permutation(len(y_final))
        return pd.DataFrame(X_final[shf], columns=ALL_TABULAR_FEATURES), pd.Series(y_final[shf], name=TARGET)

class BudgetTimeSeriesScorer:
    def __init__(self, spike_z_threshold=2.0):
        self.spike_z_threshold = spike_z_threshold
        self._global_stats = {}
    def fit(self, X, y=None):
        df = X if y is None else X[y == 0]
        for feat in FEATURES_YEAREND:
            if feat in df.columns:
                self._global_stats[feat] = {"mean": df[feat].mean(), "std": df[feat].std() + 1e-9}
        return self
    def anomaly_score(self, X):
        scores = np.zeros(len(X))
        for feat in FEATURES_YEAREND:
            if feat in X.columns and feat in self._global_stats:
                stats = self._global_stats[feat]
                z = np.abs((X[feat].fillna(0) - stats["mean"]) / stats["std"])
                scores += (1.0/len(FEATURES_YEAREND)) * np.clip(z / self.spike_z_threshold, 0, 1)
        return np.clip(scores, 0, 1)

class BudgetLightGBM:
    def __init__(self): self.model = LGBMClassifier(n_estimators=100, max_depth=4, verbose=-1, random_state=42)
    def fit(self, X, y):
        self._features = [f for f in ALL_TABULAR_FEATURES if f in X.columns]
        self.model.fit(X[self._features], y)
        return self
    def predict_proba(self, X): return self.model.predict_proba(X[self._features])[:, 1]

class BudgetDetector:
    def __init__(self):
        self.hard_rules, self.smote, self.lgbm, self.ts_scorer = BudgetHardRules(), BudgetSMOTE(), BudgetLightGBM(), BudgetTimeSeriesScorer()
    def fit(self, X, y):
        self.ts_scorer.fit(X, y)
        X_aug, y_aug = self.smote.fit_resample(X, y)
        self.lgbm.fit(X_aug, y_aug)
        return self
    def score(self, X):
        rule_flags = self.hard_rules.flag(X).values
        lgbm_s = self.lgbm.predict_proba(X)
        ts_s = self.ts_scorer.anomaly_score(X)
        ensemble = 0.6 * lgbm_s + 0.4 * ts_s
        return pd.Series(np.where(rule_flags, np.maximum(ensemble, 0.9), ensemble), index=X.index)
    def predict(self, X, threshold=0.5): return (self.score(X) >= threshold).astype(int)

def train_budget_model(df, use_cv=True):
    X, y = df[ALL_TABULAR_FEATURES], df[TARGET]
    if use_cv:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            det = BudgetDetector().fit(X.iloc[tr_idx], y.iloc[tr_idx])
            if y.iloc[te_idx].sum() > 0:
                auc = roc_auc_score(y.iloc[te_idx], det.score(X.iloc[te_idx]))
                print(f"Fold {fold+1} ROC-AUC: {auc:.4f}")
    detector = BudgetDetector().fit(X, y)
    return detector

if __name__ == "__main__":
    path = '/content/simulated_corpus.parquet'
    if os.path.exists(path):
        df_corpus = pd.read_parquet(path)
        df_budget = df_corpus[ALL_TABULAR_FEATURES + [TARGET]].dropna()
        df_budget[TARGET] = df_budget[TARGET].astype(int)
        print(f"Processing {len(df_budget)} real budget records...")
        budget_model = train_budget_model(df_budget, use_cv=True)
        
        # Save model
        model_path = "selene-budget-model.pkl"
        try:
            with open(model_path, "wb") as f:
                pickle.dump(budget_model, f)
            print(f"Successfully saved {type(budget_model).__name__} to {model_path}")
        except NameError:
            print("Error: 'budget_model' variable not found. Please ensure the training cell has been executed.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        results = df_budget.copy()
        results['risk_score'] = budget_model.score(df_budget)
        display(results[['risk_score'] + ALL_TABULAR_FEATURES].head())




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc

# Filter only relevant budget records for evaluation
df_eval = df_corpus[ALL_TABULAR_FEATURES + [TARGET]].dropna()
X_eval = df_eval[ALL_TABULAR_FEATURES]
y_eval = df_eval[TARGET].astype(int)

# Get Scores and Predictions using budget_model from previous cell
test_scores = budget_model.score(X_eval)
y_pred = budget_model.predict(X_eval)

# Visualisation Dashboard
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Feature Importance (from LightGBM layer)
model_internal = budget_model.lgbm.model
features_used = budget_model.lgbm._features
importance = model_internal.feature_importances_

imp_df = pd.DataFrame({"feature": features_used, "importance": importance}).sort_values("importance", ascending=False)
sns.barplot(x="importance", y="feature", data=imp_df.head(15), ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title("Budget Risk Key Indicators")

# 2. Score Distribution
sns.histplot(test_scores, bins=30, kde=True, ax=axes[0, 1], color="teal")
axes[0, 1].set_title("Budget Risk Score Distribution")

# 3. Confusion Matrix
cm = confusion_matrix(y_eval, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_eval, test_scores)
axes[1, 1].plot(fpr, tpr, label=f"AUC: {auc(fpr, tpr):.4f}", color="darkorange", lw=2)
axes[1, 1].plot([0,1], [0,1], 'k--')
axes[1, 1].set_title("ROC Performance (Budget Model)")
axes[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*40)
print("      BUDGET SPENDING ANOMALY SUMMARY")
print("="*40)
print(f"Total Records Analyzed : {len(df_eval):,}")
print(f"Model AUC-ROC Score    : {roc_auc_score(y_eval, test_scores):.4f}")
if not imp_df.empty:
    print(f"Top Indicator         : {imp_df.iloc[0]['feature']}")
print("="*40)