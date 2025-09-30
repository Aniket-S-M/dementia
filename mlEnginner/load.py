import pandas as pd



# Load CSV files

cog_df = pd.read_csv(r"user\OneDrive\Desktop\ml_data_extraccion_dementia\mlEnginner\features\audio_features.csv")



# Inspect merged dat
print("\n=== Cognitive Features Range ===")
print(cog_df.describe().loc[['min', 'max']])

