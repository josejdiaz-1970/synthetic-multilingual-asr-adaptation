import os, time, shutil
import soundfile as sf
from datasets import load_dataset, Dataset, Audio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_OUT = DATA_DIR / "mls_fr_0p1h"

MIN_FREE_MB = 1500
TARGET_HOURS = 0.1
MAX_FILES = 400

os.makedirs(BASE_OUT, exist_ok=True)

def free_mb():
    t,u,f = shutil.disk_usage(DATA_DIR)
    return f // (2**20)

def fr_builder():

    audio_dir = BASE_OUT / "mls_fr_0p1h_wav"
    ds_dir = BASE_OUT / "mls_fr_0p1h_ds"

    shutil.rmtree(ds_dir, ignore_errors=True)
    shutil.rmtree(audio_dir, ignore_errors=True)

    os.makedirs(audio_dir, exist_ok=True)

    rows = []
    saved = 0
    total_sec = 0.0

    print(f"[start] free={free_mb()}MB")

    stream = load_dataset(
        "facebook/multilingual_librispeech",
        "french",
        split="train",
        streaming=True
    )

    for ex in stream:

        if saved >= MAX_FILES:
            break

        if free_mb() < MIN_FREE_MB:
            print("[stop] low disk space")
            break

        text = (ex.get("text") or ex.get("transcript") or "").strip()
        if not text:
            continue

        a = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]

        out_wav = audio_dir / f"fr_{saved}.wav"

        sf.write(out_wav, a, sr)

        rows.append({
            "audio": str(out_wav),
            "text": text
        })

        saved += 1
        total_sec += len(a)/sr

        if total_sec/3600 >= TARGET_HOURS:
            break

    ds = Dataset.from_list(rows).cast_column("audio", Audio())
    os.makedirs(ds_dir, exist_ok=True)
    ds.save_to_disk(ds_dir)

    print("French dataset saved:", ds_dir)

    return ds_dir

if __name__ == "__main__":
    fr_builder()