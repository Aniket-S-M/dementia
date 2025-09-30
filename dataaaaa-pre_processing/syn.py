import pandas as pd
import numpy as np

# Parameters
n_samples = 1000
np.random.seed(42)

# Generate synthetic dataset
data = {
    "audio_id": [f"audio_{i}.wav" for i in range(1, n_samples+1)],
    "duration": np.random.uniform(2.0, 15.0, n_samples),     # seconds
    "rms": np.random.uniform(0.01, 0.2, n_samples),          # root mean square energy
}

# Add 13 MFCC features
for i in range(1, 14):
    data[f"mfcc{i}"] = np.random.normal(0, 5, n_samples)  # typical MFCC range

# Labels: 0 = Non-Dementia, 1 = Dementia
data["label"] = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("synthetic_audio_features.csv", index=False)

print("✅ Synthetic dataset created: synthetic_audio_features.csv")
print(df.head())
