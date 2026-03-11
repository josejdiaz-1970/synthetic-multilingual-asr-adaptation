from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

AUDIO_DIR = DATA_DIR / "generated_audio"

print("Checking audio directory:", AUDIO_DIR)

if not AUDIO_DIR.exists():
    print("No generated audio yet.")
else:
    print("Audio directory exists.")