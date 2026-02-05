import os
import librosa
import soundfile as sf

RAW_BASE = "ml/data/audio"
OUT_BASE = "ml/data/audio_clean"

LANGUAGES = ["tamil", "english", "hindi", "malayalam", "telugu"]

os.makedirs(OUT_BASE, exist_ok=True)

def preprocess_mp3(mp3_path, wav_out_path):
    # Load MP3 → WAV (librosa handles decoding)
    y, sr = librosa.load(mp3_path, sr=16000, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Save cleaned WAV
    sf.write(wav_out_path, y, 16000)

for lang in LANGUAGES:
    in_dir = os.path.join(RAW_BASE, lang, "human")
    out_dir = os.path.join(OUT_BASE, lang, "human")

    if not os.path.exists(in_dir):
        print(f"Skipping {lang}, folder not found")
        continue

    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(in_dir):
        if not file.lower().endswith(".mp3"):
            continue

        in_path = os.path.join(in_dir, file)
        out_file = file.replace(".mp3", ".wav")
        out_path = os.path.join(out_dir, out_file)

        preprocess_mp3(in_path, out_path)

    print(f"Processed {lang}")
