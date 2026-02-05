import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# -------- CONFIG --------
DATA_BASE = "ml/data/audio_clean"
LANGUAGES = ["english", "hindi", "malayalam", "tamil", "telugu"]
CLASSES = ["human", "ai"]
TARGET_SR = 16000

# -------- LOAD MODEL --------
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
)
model.eval()

def extract_embedding(wav_path):
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    inputs = processor(
        y,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling over time dimension
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()

# -------- MAIN LOOP --------
features = []
labels = []
languages = []

for lang in LANGUAGES:
    for cls in CLASSES:
        folder = os.path.join(DATA_BASE, lang, cls)
        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            if not file.endswith(".wav"):
                continue

            path = os.path.join(folder, file)
            emb = extract_embedding(path)

            features.append(emb)
            labels.append(1 if cls == "ai" else 0)
            languages.append(lang)

            print(f"Processed {lang}/{cls}/{file}")

# Convert to arrays
X = np.array(features)
y = np.array(labels)
lang_arr = np.array(languages)

# Save for next steps
np.save("ml/features/X_embeddings.npy", X)
np.save("ml/features/y_labels.npy", y)
np.save("ml/features/lang_labels.npy", lang_arr)

print("Embedding extraction complete")
print("Shape:", X.shape)
