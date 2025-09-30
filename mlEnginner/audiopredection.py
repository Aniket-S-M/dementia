import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  


audio_df = pd.read_csv(r"C:\Users\new user\OneDrive\Desktop\toolfor\mlEnginner\features\audio_features.csv")# replace with your file /dir 


X = audio_df.drop(columns=["audio_id", "label"], errors="ignore")
y = audio_df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)


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
    verbose=True
)


y_pred = xgb_clf.predict(X_test_scaled)
y_prob = xgb_clf.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall   :", recall_score(y_test, y_pred, average="macro"))
print("F1 Score :", f1_score(y_test, y_pred, average="macro"))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0","Class 1"], yticklabels=["Class 0","Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

joblib.dump(xgb_clf, "xgb_audio_model.pkl")
joblib.dump(scaler, "scaler_audio.pkl")
print("Model and scaler saved as 'xgb_audio_model.pkl' and 'scaler_audio.pkl'")
