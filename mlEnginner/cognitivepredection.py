import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import joblib

# --- Load CSV ---
cog_df = pd.read_csv(r'C:\Users\new user\OneDrive\Desktop\ml_data_extraccion_dementia\mlEnginner\features\cognitive_features.csv')

# --- Features & target ---
X_cog = cog_df.drop(columns=["subject_id", "diagnosis"])
y_cog = cog_df["diagnosis"]

# Encode diagnosis
le_diag = LabelEncoder()
y_cog_encoded = le_diag.fit_transform(y_cog)

# Encode gender if exists
if "gender" in X_cog.columns:
    gender_encoder = LabelEncoder()
    X_cog["gender"] = gender_encoder.fit_transform(X_cog["gender"])
else:
    gender_encoder = None

# Scale features
scaler_cog = StandardScaler()
X_cog_scaled = scaler_cog.fit_transform(X_cog)

# Shuffle dataset
X_cog_scaled, y_cog_encoded = shuffle(X_cog_scaled, y_cog_encoded, random_state=42)

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X_cog_scaled, y_cog_encoded, test_size=0.2, stratify=y_cog_encoded, random_state=42
)

# --- Logistic Regression (multi-class) ---
log_reg = LogisticRegression(
    multi_class='multinomial',  # for multi-class classification
    solver='lbfgs',
    max_iter=1000,
    C=0.5,                      # regularization strength
    random_state=42
)
log_reg.fit(X_train, y_train)

# --- Predictions ---
y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

# --- Training metrics ---
print("\n=== Training Metrics ===")
print("Accuracy :", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred, average="macro"))
print("Recall   :", recall_score(y_train, y_train_pred, average="macro"))
print("F1 Score :", f1_score(y_train, y_train_pred, average="macro"))
print("\nClassification Report (Training):")
print(classification_report(y_train, y_train_pred, target_names=le_diag.classes_))

# --- Testing metrics ---
print("\n=== Testing Metrics ===")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, average="macro"))
print("Recall   :", recall_score(y_test, y_test_pred, average="macro"))
print("F1 Score :", f1_score(y_test, y_test_pred, average="macro"))
print("\nClassification Report (Testing):")
print(classification_report(y_test, y_test_pred, target_names=le_diag.classes_))

# --- Cross-validation ---
cv_scores = cross_val_score(log_reg, X_cog_scaled, y_cog_encoded, cv=5, scoring='accuracy')
print("\n=== 5-Fold Cross-Validation Accuracy ===")
print("CV scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())

# --- Learning Curve ---
train_sizes, train_scores, test_scores = learning_curve(
    log_reg, X_cog_scaled, y_cog_encoded,
    cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', n_jobs=-1
)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# --- Plot Learning Curve ---
plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, label='Training Accuracy', marker='o')
plt.plot(train_sizes, test_mean, label='Validation Accuracy', marker='s')
plt.title("Learning Curve (Logistic Regression)")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()

# --- Save model ---
joblib.dump(log_reg, "logreg_model.pkl")
joblib.dump(scaler_cog, "cog_scaler.pkl")
joblib.dump(le_diag, "cog_label_encoder.pkl")
joblib.dump(gender_encoder, "gender_encoder.pkl")
