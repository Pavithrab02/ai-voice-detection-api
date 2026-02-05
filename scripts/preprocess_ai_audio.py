import os
import librosa
import soundfile as sf
import numpy as np

# ===== CONFIG =====
RAW_BASE = "tts_pipeline/output/ai"
OUT_BASE = "ml/data/audio_clean"

LANGUAGES = ["en", "hi", "ml", "ta", "te"]
LANG_MAP = {
    "en": "english",
    "hi": "hindi",
    "ml": "malayalam",
    "ta": "tamil",
    "te": "telugu"
}

SR = 16000
TARGET_SEC = 5.0
TARGET_LEN = int(SR * TARGET_SEC)

os.makedirs(OUT_BASE, exist_ok=True)

# ===== CORE PREPROCESS FUNCTION =====
def preprocess_and_chunk(wav_path, out_dir, base_name):
    # Load → force mono 16k
    y, _ = librosa.load(wav_path, sr=SR, mono=True)

    # Trim silence (same spirit as human)
    y, _ = librosa.effects.trim(y, top_db=35)

    # Peak normalize (safe, non-cheating)
    if len(y) > 0 and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    chunks = []

    if len(y) >= TARGET_LEN:
        # Split into exact 5s chunks
        for i in range(0, len(y) - TARGET_LEN + 1, TARGET_LEN):
            chunks.append(y[i:i + TARGET_LEN])
    else:
        # Pad short files
        pad_len = TARGET_LEN - len(y)
        chunks.append(np.pad(y, (0, pad_len)))

    # Save chunks
    for idx, chunk in enumerate(chunks):
        out_path = os.path.join(
            out_dir,
            f"{base_name}_{idx:02d}.wav"
        )
        sf.write(out_path, chunk, SR)

# ===== MAIN LOOP =====
for lang in LANGUAGES:
    in_dir = os.path.join(RAW_BASE, lang)
    out_dir = os.path.join(OUT_BASE, LANG_MAP[lang], "ai")

    if not os.path.exists(in_dir):
        print(f"⚠️ Skipping {lang}, folder not found")
        continue

    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(in_dir):
        if not file.lower().endswith(".wav"):
            continue

        in_path = os.path.join(in_dir, file)
        base_name = os.path.splitext(file)[0]

        preprocess_and_chunk(in_path, out_dir, base_name)

    print(f"✅ Processed AI audio for {LANG_MAP[lang]}")

print("\n🎯 DONE: All AI audio standardized to 5.0s, 16kHz, mono")
