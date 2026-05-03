"""
Domain: Contract — Legal Document Risk Scoring
Records: 56 | Corrupt: 5 (8.9%) | Features: 43 | Mechanisms: 3
Models: NLP clause analysis + Rule engine + Procurement vendor prior
"""

import pickle
import re
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

# Feature Groups with 'feat_' prefix
FEATURES_INFLATION = [
    "feat_contract_value_vs_budget_ratio",
    "feat_scope_vagueness_score",
    "feat_cumulative_amendment_value",
]

FEATURES_NON_COMPETITIVE = [
    "feat_competitive_flag",
    "feat_days_to_contract_signing",
    "feat_unusual_clause_count",
]

FEATURES_FAVORITISM = [
    "feat_vendor_contract_share_pct",
    "feat_negotiator_vendor_relation",
    "feat_penalty_asymmetry_score",
    "feat_exclusivity_clause_flag",
]

ALL_FEATURES = FEATURES_INFLATION + FEATURES_NON_COMPETITIVE + FEATURES_FAVORITISM
CONTRACT_TEXT_COL = "contract_text"
VENDOR_ID_COL = "agent_id" # Using agent_id as vendor proxy if vendor_id is missing
PROCUREMENT_RISK_SCORE_COL = "procurement_vendor_risk"
TARGET = "is_corrupt"

class ContractNLPScorer:
    UNUSUAL_CLAUSE_PATTERNS = [r"sole\s+discretion", r"without\s+cause", r"force\s+majeure.*payment"]
    VAGUENESS_PATTERNS = [r"as\s+needed", r"reasonable\s+time", r"sesuai\s+kebutuhan"]
    PENALTY_VENDOR_LIGHT = [r"penalty.*not\s+exceed", r"liquidated.*capped"]
    EXCLUSIVITY_PATTERNS = [r"exclusive.*supplier", r"vendor\s+tunggal"]

    def __init__(self):
        self._unusual_re = [re.compile(p, re.IGNORECASE) for p in self.UNUSUAL_CLAUSE_PATTERNS]
        self._vague_re = [re.compile(p, re.IGNORECASE) for p in self.VAGUENESS_PATTERNS]
        self._penalty_re = [re.compile(p, re.IGNORECASE) for p in self.PENALTY_VENDOR_LIGHT]
        self._excl_re = [re.compile(p, re.IGNORECASE) for p in self.EXCLUSIVITY_PATTERNS]

    def _count_matches(self, text, patterns): return sum(1 for p in patterns if p.search(text))

    def score_text(self, text):
        if not isinstance(text, str): return {"feat_unusual_clause_count": 0, "feat_scope_vagueness_score": 0.0, "feat_penalty_asymmetry_score": 0.0, "feat_exclusivity_clause_flag": 0}
        u = self._count_matches(text, self._unusual_re)
        v = min(self._count_matches(text, self._vague_re) / 5.0, 1.0)
        p = min(self._count_matches(text, self._penalty_re) / 3.0, 1.0)
        e = int(self._count_matches(text, self._excl_re) > 0)
        return {"feat_unusual_clause_count": u, "feat_scope_vagueness_score": v, "feat_penalty_asymmetry_score": p, "feat_exclusivity_clause_flag": e}

    def enrich(self, df):
        if CONTRACT_TEXT_COL not in df.columns: return df
        res = df[CONTRACT_TEXT_COL].fillna("").map(self.score_text)
        return df.assign(**pd.DataFrame(res.tolist(), index=df.index))

class ContractRuleEngine:
    def __init__(self, thresholds=(3, 15, 1.2, 0.5, 1)):
        (self.u_t, self.d_t, self.v_t, self.p_t, self.e_t) = thresholds

    def flag(self, X):
        nc = X.get("feat_competitive_flag", pd.Series(1, index=X.index)) == 0
        fs = X.get("feat_days_to_contract_signing", pd.Series(99, index=X.index)) < self.d_t
        uc = X.get("feat_unusual_clause_count", pd.Series(0, index=X.index)) >= self.u_t
        ov = X.get("feat_contract_value_vs_budget_ratio", pd.Series(1.0, index=X.index)) >= self.v_t
        ap = X.get("feat_penalty_asymmetry_score", pd.Series(0.0, index=X.index)) >= self.p_t
        ex = X.get("feat_exclusivity_clause_flag", pd.Series(0, index=X.index)) >= self.e_t
        return (nc | fs | uc | ov | ap | ex).rename("rule_flag")

class ProcurementPrior:
    def __init__(self, risk_map=None): self.risk_map = risk_map or {}
    def enrich(self, df):
        if VENDOR_ID_COL not in df.columns: return df
        df[PROCUREMENT_RISK_SCORE_COL] = df[VENDOR_ID_COL].astype(str).map(lambda x: self.risk_map.get(x, 0.1))
        return df

class ContractRiskCalculator:
    WEIGHTS = {"rule": 0.4, "nlp": 0.3, "prior": 0.2, "process": 0.1}
    def fit(self, X): 
        self.stats = {f: {'min': X[f].min(), 'max': X[f].max()+1e-9} for f in ALL_FEATURES if f in X.columns}
        return self
    def _norm(self, val, f): s = self.stats.get(f, {'min':0, 'max':1}); return np.clip((val-s['min'])/(s['max']-s['min']), 0, 1)
    def compute(self, X):
        scores = []
        for _, r in X.iterrows():
            nlp = 0.35*self._norm(r.get("feat_unusual_clause_count",0), "feat_unusual_clause_count") + 0.25*r.get("feat_exclusivity_clause_flag",0)
            rule = 1.0 if r.get("feat_competitive_flag",1)==0 else 0.2
            prior = r.get(PROCUREMENT_RISK_SCORE_COL, 0.1)
            proc = 0.6*self._norm(r.get("feat_vendor_contract_share_pct",0), "feat_vendor_contract_share_pct")
            scores.append(self.WEIGHTS['rule']*rule + self.WEIGHTS['nlp']*nlp + self.WEIGHTS['prior']*prior + self.WEIGHTS['process']*proc)
        return np.array(scores)

class ContractDetector:
    def __init__(self, prior_map=None):
        self.nlp, self.rule, self.prior, self.calc = ContractNLPScorer(), ContractRuleEngine(), ProcurementPrior(prior_map), ContractRiskCalculator()
    def fit(self, X, y=None): 
        X_e = self.prior.enrich(self.nlp.enrich(X))
        self.calc.fit(X_e); return self
    def score(self, X): return pd.Series(self.calc.compute(self.prior.enrich(self.nlp.enrich(X))), index=X.index, name="contract_risk_score")
    def predict(self, X, t=0.35): return (self.score(X) >= t).astype(int)

def train_contract_model(df, prior_map=None):
    print("="*50 + "\nTraining Contract Risk Model\n" + "="*50)
    detector = ContractDetector(prior_map).fit(df, df[TARGET])
    scores = detector.score(df)
    print(f"ROC-AUC: {roc_auc_score(df[TARGET], scores):.4f}")
    
    # Save model
    model_path = "selene-contract-model.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(detector, f)
        print(f"Successfully saved {type(detector).__name__} to {model_path}")
    except NameError:
        print("Error: 'detector' variable not found. Please ensure the training cell has been executed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return detector

if __name__ == "__main__":
    path = '/content/simulated_corpus.parquet'
    if pd.io.common.file_exists(path):
        df_full = pd.read_parquet(path)
        df_contract = df_full[df_full['domain'] == 'contract'].copy()
        df_contract = df_contract[ALL_FEATURES + [TARGET, VENDOR_ID_COL]].dropna(subset=ALL_FEATURES)
        model = train_contract_model(df_contract)
        df_contract['risk_score'] = model.score(df_contract)
        display(df_contract[['risk_score', TARGET] + ALL_FEATURES].head())



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc

# 1. Prepare Data
df_eval = df_contract.copy()
X_eval = df_eval[ALL_FEATURES]
y_eval = df_eval[TARGET].astype(int)

# Get Scores and Predictions
test_scores = model.score(X_eval)
df_eval['risk_score'] = test_scores

# 2. Visualisation Dashboard (Expanded to 2x2)
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# A. Top Anomaly Indicators (Feature contribution to score)
# For this heuristic model, we look at the correlation of each feature with the risk score
corrs = X_eval.corrwith(pd.Series(test_scores, index=X_eval.index)).sort_values(ascending=False)
sns.barplot(x=corrs.values, y=corrs.index, ax=axes[0, 0], palette="Reds_r")
axes[0, 0].set_title("Top Anomaly Indicators (Correlation with Risk Score)")
axes[0, 0].set_xlabel("Correlation Coefficient")

# B. Density of Risk Score (Separation by Target)
sns.kdeplot(data=df_eval[df_eval[TARGET] == 0], x='risk_score', fill=True, label='Normal', ax=axes[0, 1], color="#3498db")
sns.kdeplot(data=df_eval[df_eval[TARGET] == 1], x='risk_score', fill=True, label='Corrupt', ax=axes[0, 1], color="#e74c3c")
axes[0, 1].set_title("Density of Risk Scores by Class")
axes[0, 1].set_xlabel("Risk Score")
axes[0, 1].legend()

# C. ROC Curve
fpr, tpr, _ = roc_curve(y_eval, test_scores)
axes[1, 0].plot(fpr, tpr, label=f"AUC: {auc(fpr, tpr):.4f}", color="crimson", lw=3)
axes[1, 0].plot([0,1], [0,1], 'k--', alpha=0.5)
axes[1, 0].set_title("ROC Performance")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].legend(loc="lower right")

# D. Confusion Matrix
y_pred = model.predict(X_eval)
cm = confusion_matrix(y_eval, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 1])
axes[1, 1].set_title("Confusion Matrix (Threshold=0.35)")
axes[1, 1].set_xlabel("Predicted")
axes[1, 1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Summary Summary
print("\n" + "="*40)
print("      CONTRACT RISK ANALYSIS SUMMARY")
print("="*40)
print(f"Total Contracts Analyzed : {len(df_eval):,}")
print(f"Corrupt Cases Detected   : {y_eval.sum()}")
print(f"Model AUC-ROC Score      : {roc_auc_score(y_eval, test_scores):.4f}")
print(f"Top Indicator            : {corrs.index[0]}")
print("="*40)