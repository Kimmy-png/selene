import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

ALL_TABULAR_FEATURES = [
    "feat_undocumented_spend_pct", "feat_salary_change_backdated_days",
    "feat_submitter_rejection_rate", "feat_night_step_count",
    "feat_conformance_fitness", "feat_skipped_steps",
    "feat_failed_auth_count", "feat_salary_vs_peer_ratio"
]
TARGET = "is_corrupt"
ID_COLS = {"vendor": "agent_id", "employee": "agent_manager_id"}

class ProcurementXGBoost:
    def __init__(self, corrupt_ratio: float = 0.04):
        spw = (1 - corrupt_ratio) / corrupt_ratio
        self.model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="aucpr", tree_method="hist", random_state=42
        )
        self.features = ALL_TABULAR_FEATURES
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProcurementXGBoost":
        self.model.fit(X[self.features].astype(float), y.astype(int), verbose=False)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted: raise RuntimeError("Model belum ditraining.")
        return self.model.predict_proba(X[self.features].astype(float))[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.features,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

class VendorRelationGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.risk_scores: dict[str, float] = {}

    def build_graph(self, df: pd.DataFrame) -> "VendorRelationGraph":
        for _, row in df.sample(min(10000, len(df))).iterrows():
            v, e = str(row[ID_COLS['vendor']]), str(row[ID_COLS['employee']])
            self.G.add_edge(v, e, weight=1.0)
        return self

    def compute_risk_scores(self) -> dict[str, float]:
        if len(self.G) == 0: return {}
        self.risk_scores = nx.pagerank(self.G, weight="weight", max_iter=100)
        return self.risk_scores

    def get_score(self, agent_id) -> float:
        return self.risk_scores.get(str(agent_id), 0.0)

class ProcurementDetector:
    def __init__(self):
        self.xgb = ProcurementXGBoost()
        self.graph = VendorRelationGraph()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProcurementDetector":
        self.xgb.fit(X, y)
        self.graph.build_graph(X).compute_risk_scores()
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        xgb_scores = self.xgb.predict_proba(X)
        graph_raw = X[ID_COLS['vendor']].map(self.graph.get_score).fillna(0.0)
        max_g = graph_raw.max() or 1.0
        graph_norm = (graph_raw / max_g).values
        return pd.Series(0.7 * xgb_scores + 0.3 * graph_norm, index=X.index)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        return (self.score(X) >= threshold).astype(int)

# --- EXECUTING GLOBALLY ---
DATA_PATH = "simulated_corpus.parquet"
if os.path.exists(DATA_PATH):
    print("Loading Parquet dataset...")
    df = pd.read_parquet(DATA_PATH)
    df[ALL_TABULAR_FEATURES] = df[ALL_TABULAR_FEATURES].fillna(0)

    print("Training Model...")
    X_train, X_test, y_train, y_test = train_test_split(
        df, df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=42
    )
    detector = ProcurementDetector().fit(X_train, y_train)
    df["risk_score"] = detector.score(df)
    
    # Save model
    model_path = "selene-procurement-model.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(detector, f)
        print(f"Successfully saved {type(detector).__name__} to {model_path}")
    except NameError:
        print("Error: 'detector' variable not found. Please ensure the training cell has been executed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    print("Training complete. Variables 'detector', 'X_test', 'y_test' are now available.")
else:
    print("File not found.")


# summary and visualization


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set plotting style
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Feature Importance
imp_df = detector.xgb.feature_importance()
sns.barplot(x="importance", y="feature", data=imp_df, ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title("Top Corruption Indicators (XGBoost)")

# 2. Risk Score Distribution
sns.histplot(df, x="risk_score", hue=TARGET, element="step", stat="density", common_norm=False, ax=axes[0, 1])
axes[0, 1].set_title("Distribution of Risk Scores by Label")

# 3. Confusion Matrix (on Test Set)
y_test_pred = detector.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, detector.score(X_test))
roc_auc = auc(fpr, tpr)
axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1, 1].set_title("Receiver Operating Characteristic (ROC)")
axes[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.show()

print("EXECUTIVE SUMMARY")
print("="*30)
high_risk_count = (df['risk_score'] > 0.5).sum()
print(f"Total Transactions Analyzed: {len(df):,}")
print(f"High Risk Agents Flagged: {high_risk_count} ({high_risk_count/len(df)*100:.2f}%)")

print("\nTOP 5 HIGH-RISK AGENTS:")
display(df[['agent_id', 'risk_score', 'feat_undocumented_spend_pct']].sort_values('risk_score', ascending=False).head(5))

print("\nKEY INSIGHT:")
# Calculate importance if not already calculated in previous cell
imp_df = detector.xgb.feature_importance()
top_feat = imp_df.iloc[0]['feature']
print(f"The most critical factor in detecting corruption is '{top_feat}'.")

model_path = "selene/model/selene-procurement-model.pkl"

try:
    with open(model_path, "wb") as f:
        pickle.dump(detector, f)
    print(f"Successfully saved {type(detector).__name__} to {model_path}")
except NameError:
    print("Error: 'detector' variable not found. Please ensure the training cell has been executed.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")