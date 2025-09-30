import os
import librosa
import numpy as np
import pandas as pd

# Paths
base_path = 'data'
categories = ['dementia', 'nondementia']

# Feature extraction
audio_feature_list = []

for label, category in enumerate(categories):  # dementia=0, non_dementia=1
    folder = os.path.join(base_path, category)
    audio_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    
    for file in audio_files:
        path = os.path.join(folder, file)
        
        # Load audio
        y, sr = librosa.load(path, sr=16000)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1)
        
        # RMS
        rms = librosa.feature.rms(y=y).mean()
        
        # Duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Save features + label
        audio_feature_list.append({
            'audio_id': file,
            'duration': duration,
            'rms': rms,
            **{f'mfcc{i+1}': mfccs_mean[i] for i in range(13)},
            'label': label  # dementia=0, non_dementia=1
        })

# Convert to DataFrame
audio_features_df = pd.DataFrame(audio_feature_list)

# Save
os.makedirs('features', exist_ok=True)
audio_features_df.to_csv('features/audio_features.csv', index=False)

print("✅ Audio features with labels saved to features/audio_features.csv")
