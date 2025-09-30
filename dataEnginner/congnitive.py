import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 1000  # number of subjects

# Demographics
age = np.random.randint(50, 90, n)
gender = np.random.choice(['Male', 'Female'], n)
education_years = np.random.randint(0, 20, n)

# Cognitive scores
memory_score = np.clip(np.random.normal(25 - 0.2*(age-50), 3, n), 0, 30)
attention_score = np.clip(np.random.normal(24 - 0.15*(age-50), 4, n), 0, 30)
language_score = np.clip(np.random.normal(26 - 0.1*(age-50), 3, n), 0, 30)
executive_score = np.clip(np.random.normal(23 - 0.2*(age-50), 5, n), 0, 30)
visuospatial_score = np.clip(np.random.normal(22 - 0.15*(age-50), 4, n), 0, 30)
reaction_time_ms = np.clip(np.random.normal(300 + 3*(age-50), 50, n), 150, 800)
mood_score = np.random.randint(1, 10, n)

# Diagnosis (simplified)
diagnosis = []
for mem in memory_score:
    if mem > 23:
        diagnosis.append('Normal')
    elif mem > 18:
        diagnosis.append('MCI')  # Mild Cognitive Impairment
    else:
        diagnosis.append('Dementia')

# Assemble dataset
df = pd.DataFrame({
    'subject_id': range(1, n+1),
    'age': age,
    'gender': gender,
    'education_years': education_years,
    'memory_score': memory_score,
    'attention_score': attention_score,
    'language_score': language_score,
    'executive_score': executive_score,
    'visuospatial_score': visuospatial_score,
    'reaction_time_ms': reaction_time_ms,
    'mood_score': mood_score,
    'diagnosis': diagnosis
})

# Path to save CSV
output_path = r"C:\Users\new user\OneDrive\Desktop\toolfor\features\cognitive_features.csv"

# Ensure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save CSV
df.to_csv(output_path, index=False)

print(f"CSV saved to: {output_path}")
print(df.head())
