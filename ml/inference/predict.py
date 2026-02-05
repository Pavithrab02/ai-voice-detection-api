import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from joblib import load

TARGET_SR = 16000

# Load model + scaler
clf = load("ml/models/voice_classifier.joblib")
scaler = load("ml/models/signal_scaler.joblib")

# Load wav2vec
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def extract_embedding(audio_path):
    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    inputs = processor(
        y,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def extract_signal_features(audio_path):
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    f0 = librosa.yin(y, fmin=50, fmax=300)
    pitch_var = np.var(f0)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_mean = np.mean(spec_centroid)
    spec_var = np.var(spec_centroid)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    return np.array([pitch_var, spec_mean, spec_var, zcr_mean])


def generate_explanation(sig_feats, is_ai):
    pitch_var, spec_mean, spec_var, zcr = sig_feats

    if is_ai:
        reasons = []

        if pitch_var < 3000:
            reasons.append("unnaturally stable pitch")
        if spec_var < 8e5:
            reasons.append("overly smooth spectral profile")
        if zcr < 0.1:
            reasons.append("robotic waveform structure")

        if reasons:
            return " and ".join(reasons).capitalize() + " detected"
        else:
            return "Acoustic patterns consistent with synthetic speech detected"

    else:
        return "Natural pitch variation and spectral dynamics detected"


def predict(audio_path):
    emb = extract_embedding(audio_path)
    sig = extract_signal_features(audio_path)

    sig_scaled = scaler.transform(sig.reshape(1, -1))
    X = np.concatenate([emb.reshape(1, -1), sig_scaled], axis=1)

    prob_ai = clf.predict_proba(X)[0][1]
    is_ai = prob_ai >= 0.5

    label = "AI_GENERATED" if is_ai else "HUMAN"
    explanation = generate_explanation(sig, is_ai)

    # Confidence in the predicted class
    confidence = prob_ai if is_ai else (1 - prob_ai)

    return {
        "classification": label,
        "confidenceScore": round(float(confidence), 3),
        "explanation": explanation
    }
