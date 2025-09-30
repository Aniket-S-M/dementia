import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# Load dataset
# =======================
audio_df = pd.read_csv(r'features\\audio_features.csv')

# Features & target
X = audio_df.drop(columns=["audio_id", "label"], errors="ignore")
y = audio_df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# =======================
# XGBoost with regularization
# =======================
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=500,
    reg_alpha=0.5,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2
)

eval_set = [(X_train_bal, y_train_bal), (X_test_scaled, y_test)]
xgb_clf.fit(
    X_train_bal, y_train_bal,
    eval_set=eval_set,
    early_stopping_rounds=20,
    verbose=False
)

# =======================
# Evaluation
# =======================
y_pred = xgb_clf.predict(X_test_scaled)
y_prob = xgb_clf.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
roc = roc_auc_score(y_test, y_prob)

print("\n=== Classification Report ===\n")
print(classification_report(y_test, y_pred))

print("✅ Accuracy :", f"{acc*100:.2f}%")
print("Precision  :", f"{prec:.3f}")
print("Recall     :", f"{rec:.3f}")
print("F1 Score   :", f"{f1:.3f}")
print("ROC-AUC    :", f"{roc:.3f}")

# =======================
# Plots
# =======================

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Class 0","Class 1"],
            yticklabels=["Class 0","Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)")
plt.tight_layout()
plt.show()

# --- ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color="blue", label=f"XGBoost (AUC = {roc:.3f})")
plt.plot([0,1], [0,1], color="red", linestyle="--")  # baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()
