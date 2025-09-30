import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Example: Load MFCC dataset
mfcc_data = np.load(r"C:\Users\new user\OneDrive\Desktop\toolfor\features\mfcc_data.npy")  # shape: (1000, 40, 40)
mfcc_labels = np.load(r"C:\Users\new user\OneDrive\Desktop\toolfor\features\mfcc_labels.npy")  # shape: (1000,)

# Normalize each MFCC feature across time
num_samples = mfcc_data.shape[0]
for i in range(num_samples):
    scaler = StandardScaler()
    mfcc_data[i] = scaler.fit_transform(mfcc_data[i])

# Add channel dimension for CNN: (samples, 1, n_mfcc, n_frames)
mfcc_data = np.expand_dims(mfcc_data, axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(mfcc_data, mfcc_labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# PyTorch Dataset
class MFCCDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MFCCDataset(X_train_tensor, y_train_tensor)
test_dataset = MFCCDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
