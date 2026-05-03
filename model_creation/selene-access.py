import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# Feature Groups
FEATURES_UNAUTHORIZED = [
    "feat_resource_first_time_flag",
    "feat_cross_dept_resource_flag",
    "feat_resource_novelty_score",
    "feat_resource_sensitivity_level",
]

FEATURES_PRIVILEGE = [
    "feat_hour_deviation_from_baseline",
    "feat_is_off_hours",
    "feat_action_type_entropy",
    "feat_login_cadence_irregularity",
]

FEATURES_EXFIL = [
    "feat_volume_z_score",
    "feat_data_volume_mb",
    "feat_autoencoder_reconstruction_error",
    "feat_failed_auth_count",
    "feat_ip_address_entropy",
]

ALL_TABULAR_FEATURES = FEATURES_UNAUTHORIZED + FEATURES_PRIVILEGE + FEATURES_EXFIL
SEQUENCE_FEATURE = "feat_activity_sequence"
TARGET = "is_corrupt"

# Model Definitions
class AccessAutoencoder:
    def __init__(self):
        self.scaler = StandardScaler()
        self._sklearn_ae = None
        self.is_fitted = False

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        feats = [f for f in ALL_TABULAR_FEATURES if f in X.columns]
        X_normal = X[y == 0][feats].fillna(0)
        if X_normal.empty: return self
        X_scaled = self.scaler.fit_transform(X_normal)
        self._sklearn_ae = MLPRegressor(hidden_layer_sizes=(32, 16, 32), activation="tanh", max_iter=100, random_state=42)
        self._sklearn_ae.fit(X_scaled, X_scaled)
        self.is_fitted = True
        return self

    def get_score(self, X):
        if not self.is_fitted: return np.zeros(len(X))
        feats = [f for f in ALL_TABULAR_FEATURES if f in X.columns]
        X_scaled = self.scaler.transform(X[feats].fillna(0))
        recon = self._sklearn_ae.predict(X_scaled)
        errors = np.mean((X_scaled - recon) ** 2, axis=1)
        return errors

class LSTMSequenceScorer:
    def __init__(self):
        self.is_fitted = False
        self.n_actions = 10
        self._transition_matrix = None

    def _encode_sequence(self, seq):
        if isinstance(seq, (list, np.ndarray)): return [int(s) for s in seq if s is not None]
        if isinstance(seq, str):
            try:
                tokens = seq.strip("[]").split(",")
                return [int(t.strip()) for t in tokens if t.strip().isdigit()]
            except: return []
        return []

    def fit(self, sequences, labels):
        matrix = np.ones((self.n_actions, self.n_actions))
        for seq_raw, label in zip(sequences, labels):
            if label != 0: continue
            seq = self._encode_sequence(seq_raw)
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i+1]
                if 0 <= a < self.n_actions and 0 <= b < self.n_actions: matrix[a][b] += 1
        self._transition_matrix = matrix / matrix.sum(axis=1, keepdims=True)
        self.is_fitted = True
        return self

    def get_score(self, sequences):
        scores = []
        for seq_raw in sequences:
            seq = self._encode_sequence(seq_raw)
            if not seq or not self.is_fitted:
                scores.append(0.0)
                continue
            log_p = 0.0
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i+1]
                if 0 <= a < self.n_actions and 0 <= b < self.n_actions:
                    log_p += np.log(max(self._transition_matrix[a][b], 1e-10))
            scores.append(-log_p / max(len(seq), 1))
        return np.array(scores)

class AccessDetector:
    def __init__(self):
        self.autoencoder = AccessAutoencoder()
        self.lstm = LSTMSequenceScorer()
        self.meta_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=10,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def _preprocess_volume(self, X):
        X_p = X.copy()
        if "feat_data_volume_mb" in X_p.columns:
            X_p["feat_data_volume_mb"] = np.log1p(X_p["feat_data_volume_mb"])
        return X_p

    def _build_features(self, X):
        X_p = self._preprocess_volume(X)
        ae_feat = self.autoencoder.get_score(X_p)
        lstm_feat = self.lstm.get_score(X_p[SEQUENCE_FEATURE]) if SEQUENCE_FEATURE in X_p.columns else np.zeros(len(X_p))
        tabular = X_p[ALL_TABULAR_FEATURES].fillna(0).values
        return np.column_stack([tabular, ae_feat, lstm_feat])

    def fit(self, X, y):
        X_p = self._preprocess_volume(X)
        self.autoencoder.fit(X_p, y)
        if SEQUENCE_FEATURE in X_p.columns: self.lstm.fit(X_p[SEQUENCE_FEATURE], y)
        augmented_X = self._build_features(X)
        self.meta_model.fit(augmented_X, y)
        return self

    def predict_proba(self, X):
        augmented_X = self._build_features(X)
        return self.meta_model.predict_proba(augmented_X)[:, 1]

    def predict(self, X, threshold=0.3):
        # Support custom thresholding
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

def train(df, threshold=0.3):
    X = df.drop(columns=[TARGET]) if TARGET in df.columns else df
    y = df[TARGET] if TARGET in df.columns else pd.Series(0, index=df.index)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    detector = AccessDetector()
    detector.fit(X_train, y_train)

    # Evaluate with the requested threshold
    y_pred = detector.predict(X_test, threshold=threshold)
    print(f"Augmented XGBoost Meta-Model Results (Threshold={threshold}):")
    print(classification_report(y_test, y_pred))
    return detector

# 1. Load dataset
path = '/kaggle/input/datasets/mhiskabee/celine-corp-dataset/simulated_corpus.parquet'
df_custom = pd.read_parquet(path)

print(f"Dataset loaded: {df_custom.shape[0]} records, {df_custom.shape[1]} columns")

# 2. Run training with the new supervised meta-model and threshold 0.3
try:
    trained_model = train(df_custom, threshold=0.3)
    
    # Save model
    model_path = "selene-access-model.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(trained_model, f)
        print(f"Successfully saved {type(trained_model).__name__} to {model_path}")
    except NameError:
        print("Error: 'trained_model' variable not found. Please ensure the training cell has been executed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
except Exception as e:
    print(f"Error during training: {e}")

#summary and visualization

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# 1. Prepare evaluation data
y_true = df_custom[TARGET].astype(int)

# 2. Use the new threshold of 0.3
current_threshold = 0.3
risk_probs = trained_model.predict_proba(df_custom)
y_pred = (risk_probs >= current_threshold).astype(int)
df_custom['risk_score'] = risk_probs

# Set plotting style
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- 1. Top Anomaly Indicators ---
normal_avg = df_custom[df_custom[TARGET] == 0][ALL_TABULAR_FEATURES].mean()
corrupt_avg = df_custom[df_custom[TARGET] == 1][ALL_TABULAR_FEATURES].mean()
imp_series = ((corrupt_avg - normal_avg).abs()).sort_values(ascending=False).head(10)
sns.barplot(x=imp_series.values, y=imp_series.index, ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title("Top Anomaly Indicators (Feature Deviation)")

# --- 2. Risk Score Distribution ---
sns.kdeplot(data=df_custom, x="risk_score", hue=TARGET, fill=True, common_norm=False, ax=axes[0, 1])
axes[0, 1].axvline(current_threshold, color='red', linestyle='--', label=f'Threshold={current_threshold}')
axes[0, 1].set_title("Density of Risk Scores by Class")
axes[0, 1].legend()

# --- 3. Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0])
axes[1, 0].set_title(f"Confusion Matrix (Threshold={current_threshold})")
axes[1, 0].set_xlabel("Predicted Corruption")
axes[1, 0].set_ylabel("Actual")

# --- 4. ROC Curve ---
fpr, tpr, _ = roc_curve(y_true, risk_probs)
roc_auc = auc(fpr, tpr)
axes[1, 1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
axes[1, 1].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[1, 1].set_title("Receiver Operating Characteristic")
axes[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.show()

print(" SUPERVISED MODEL SUMMARY (THRESHOLD: 0.3)")
print("="*45)
high_risk_df = df_custom[df_custom['risk_score'] >= current_threshold]
precision_val = (high_risk_df[TARGET] == 1).sum() / len(high_risk_df) if len(high_risk_df) > 0 else 0
recall_val = (high_risk_df[TARGET] == 1).sum() / (df_custom[TARGET] == 1).sum()

print(f"Total Records: {len(df_custom):,}")
print(f"Model Flags: {len(high_risk_df)} suspicious cases")
print(f"Model Precision: {precision_val:.2%}")
print(f"Model Recall: {recall_val:.2%}")