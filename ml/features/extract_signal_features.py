import os
import librosa
import numpy as np

DATA_BASE = "ml/data/audio_clean"
LANGUAGES = ["english", "hindi", "malayalam", "tamil", "telugu"]
CLASSES = ["human", "ai"]
TARGET_SR = 16000

signal_features = []
labels = []
languages = []

def extract_signal_features(wav_path):
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    # Pitch (fundamental frequency)
    f0 = librosa.yin(y, fmin=50, fmax=300)
    pitch_var = np.var(f0)

    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_mean = np.mean(spec_centroid)
    spec_var = np.var(spec_centroid)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    return [pitch_var, spec_mean, spec_var, zcr_mean]

for lang in LANGUAGES:
    for cls in CLASSES:
        folder = os.path.join(DATA_BASE, lang, cls)
        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            if not file.endswith(".wav"):
                continue

            path = os.path.join(folder, file)
            feats = extract_signal_features(path)

            signal_features.append(feats)
            labels.append(1 if cls == "ai" else 0)
            languages.append(lang)

            print(f"Processed {lang}/{cls}/{file}")

X_sig = np.array(signal_features)
y = np.array(labels)
lang_arr = np.array(languages)

np.save("ml/features/X_signal.npy", X_sig)
np.save("ml/features/y_signal.npy", y)
np.save("ml/features/lang_signal.npy", lang_arr)

print("Signal feature extraction complete")
print("Shape:", X_sig.shape)
print("Label distribution:", np.unique(y, return_counts=True))