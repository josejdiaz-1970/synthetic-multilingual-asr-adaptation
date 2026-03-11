from pathlib import Path
from datasets import load_from_disk
from TTS.api import TTS
import soundfile as sf
import random

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

EN_DATASET = DATA_DIR / "tts_input_en_0p25h" / "tts_input_en_0p25h_ds"
ES_DATASET = DATA_DIR / "mls_es_0p1h" / "mls_es_0p1h_ds"
FR_DATASET = DATA_DIR / "mls_fr_0p1h" / "mls_fr_0p1h_ds"

OUT_DIR = DATA_DIR / "generated_audio"
OUT_DIR.mkdir(exist_ok=True)

print("Using datasets:")
print("EN:", EN_DATASET)
print("ES:", ES_DATASET)
print("FR:", FR_DATASET)

# load datasets
en_ds = load_from_disk(str(EN_DATASET))
es_ds = load_from_disk(str(ES_DATASET))
fr_ds = load_from_disk(str(FR_DATASET))

print("Datasets loaded")

# load multilingual TTS
print("Loading TTS model (this may download ~2GB first time)...")

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

def generate_samples(dataset, lang, count=10):

    lang_dir = OUT_DIR / lang
    lang_dir.mkdir(exist_ok=True)

    samples = random.sample(list(dataset), min(count, len(dataset)))

    for i, row in enumerate(samples):

        text = row.get("text") or row.get("transcript")

        if not text:
            continue

        out_file = lang_dir / f"{lang}_{i}.wav"

        print(f"{lang} -> {text[:60]}")

        wav = tts.tts(text=text, language=lang)

        sf.write(out_file, wav, 22050)

print("\nGenerating English samples...")
generate_samples(en_ds, "en")

print("\nGenerating Spanish samples...")
generate_samples(es_ds, "es")

print("\nGenerating French samples...")
generate_samples(fr_ds, "fr")

print("\nFinished generating synthetic audio.")
print("Audio saved to:", OUT_DIR)