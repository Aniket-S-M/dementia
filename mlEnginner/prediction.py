import pandas as pd
import joblib


audio_clf = joblib.load("audio_model.pkl")
scaler_audio = joblib.load("audio_scaler.pkl")




#audio end point broo
def predict_audio(audio_features: dict):
    df = pd.DataFrame([audio_features])
    X_scaled = scaler_audio.transform(df)
    pred = audio_clf.predict(X_scaled)
    return int(pred[0])


#cognative end point 
def predict_cognitive(cog_features: dict):
    import pandas as pd
    import joblib

    model = joblib.load("cog_model.pkl")
    scaler = joblib.load("cog_scaler.pkl")
    le_diag = joblib.load("cog_label_encoder.pkl")
    gender_encoder = joblib.load("gender_encoder.pkl")

    df = pd.DataFrame([cog_features])

    # --- FIX GENDER LABELS ---
    if gender_encoder and "gender" in df.columns:
        # Capitalize first letter to match encoder's classes
        df["gender"] = df["gender"].str.capitalize()
        # Optional: check if gender still unseen
        unseen = set(df["gender"]) - set(gender_encoder.classes_)
        if unseen:
            raise ValueError(f"Unseen gender labels: {unseen}")
        df["gender"] = gender_encoder.transform(df["gender"])

    X_scaled = scaler.transform(df)
    pred_encoded = model.predict(X_scaled)
    pred_label = le_diag.inverse_transform(pred_encoded)
    return pred_label[0]




#combined
def predict_combined(audio_features: dict, cog_features: dict):
    pred_a = predict_audio(audio_features)
    pred_c = predict_cognitive(cog_features)

    
    if pred_a == pred_c:
        return pred_a
    else:
        return pred_c   
