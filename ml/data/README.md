# How to place datasets (no audio here)

# Dataset Instructions

Audio files are NOT stored in this repository.

Expected local structure:

ml/data/audio/
  <language>/
    human/
    ai/

Languages supported:
- tamil
- english
- hindi
- malayalam
- telugu

Audio requirements:
- WAV format
- 16kHz
- Mono
- 3–6 seconds

Human data source:
- Indian Languages Audio Dataset

AI data sources:
- Coqui TTS
- Festival
- VITS-based TTS

Use scripts in /scripts to preprocess audio.
