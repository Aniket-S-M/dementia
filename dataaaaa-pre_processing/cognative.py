import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load CSV
cog_df = pd.read_csv(r"C:\Users\new user\OneDrive\Desktop\toolfor\features\audio_features.csv")

# Drop subject_id
X = cog_df.drop(columns=['subject_id', 'diagnosis'])

# Optional: Fill missing values
X = X.fillna(X.mean())

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target
y = LabelEncoder().fit_transform(cog_df['diagnosis'])  # Normal=0, MCI=1, Dementia=2

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors (optional if using PyTorch)
import torch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
