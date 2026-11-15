"""Stage 0 quick WER: the pipeline's default STT handler on LibriSpeech test-clean.

``benchmark_stt.py`` times one file and has no reference text, so it cannot give a WER.
This script runs ``ParakeetTDTSTTHandler`` (the ``--stt parakeet-tdt`` default, same code
path as the server) over the first N utterances of LibriSpeech test-clean and scores
the final transcripts with jiwer.

LibriSpeech (openslr/librispeech_asr, CC BY 4.0) is read from the Hub parquet file at a
pinned revision. Only the row groups that hold the first N rows are fetched.

Usage:
    python scripts/stage0_wer.py --limit 100 --device cuda --out runs/s0/wer.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import time
from queue import Queue
from threading import Event

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler

DATASET = "openslr/librispeech_asr"
REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
PARQUET = "clean/test/0000.parquet"


def normalize(text: str) -> str:
    """Lowercase, keep letters, digits and apostrophes, collapse spaces."""
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    return " ".join(text.split())


def load_rows(limit: int) -> list[dict]:
    fs = HfFileSystem()
    path = f"datasets/{DATASET}@{REVISION}/{PARQUET}"
    rows: list[dict] = []
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        for group in range(pf.num_row_groups):
            table = pf.read_row_group(group, columns=["id", "text", "audio"])
            rows.extend(table.to_pylist())
            if len(rows) >= limit:
                break
    return rows[:limit]


def decode(audio: dict) -> np.ndarray:
    samples, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if sr != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sr}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=100, help="Number of utterances (file order).")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--out", default=None, help="Write per-utterance results and totals as JSON.")
    args = parser.parse_args()

    rows = load_rows(args.limit)
    print(f"loaded {len(rows)} utterances from {DATASET}@{REVISION[:8]} {PARQUET}")

    handler = ParakeetTDTSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"device": args.device, "enable_live_transcription": False},
    )

    refs, hyps, seconds, audio_s, items = [], [], [], [], []
    for row in rows:
        audio = decode(row["audio"])
        start = time.perf_counter()
        outputs = list(handler.process(VADAudio(audio=audio, mode="final")))
        seconds.append(time.perf_counter() - start)
        audio_s.append(len(audio) / 16000)
        text = next((o.text for o in outputs if isinstance(o, Transcription)), "")
        refs.append(normalize(row["text"]))
        hyps.append(normalize(text))
        items.append({"id": row["id"], "ref": row["text"], "hyp": text, "seconds": round(seconds[-1], 4)})

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    q = statistics.quantiles(seconds, n=10)
    totals = {
        "dataset": f"{DATASET}@{REVISION}",
        "file": PARQUET,
        "samples": len(rows),
        "audio_minutes": round(sum(audio_s) / 60, 2),
        "model": handler.model_name,
        "device": handler.device,
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "latency_p50_s": round(statistics.median(seconds), 4),
        "latency_p90_s": round(q[8], 4),
        "rtfx": round(sum(audio_s) / sum(seconds), 1),
    }
    print(json.dumps(totals, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"totals": totals, "items": items}, fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
