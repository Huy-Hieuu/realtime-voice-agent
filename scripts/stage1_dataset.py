"""Stage 1 dataset builder (docs/stage1-walkthrough.md): MInDS-14 bank calls -> audio, transcript, intent, response, output audio.

One manifest per dataset, <data>/manifest.jsonl, one JSON object per clip. The first three keys are the plain
ASR format {"id", "audio", "text"}, so Stage 2 can score any split directly:

  id, audio, text                 clip id, 16 kHz mono PCM16 WAV (relative path), reference transcript
  intent                          one of the 14 MInDS-14 intent names
  duration, source_sr             clip seconds, sample rate of the original recording (8000 for MInDS-14)
  response, response_model        the assistant's spoken reply and the LLM that wrote it   (respond)
  response_audio, response_voice  24 kHz WAV of the reply and the TTS voice that spoke it   (speak)
  qc                              list of failed quality rules; [] means the clip passed     (check)
  split                           train / validation / test, only for clips that passed     (split)

Subcommands, in order (each one rewrites the manifest and can be re-run; respond/speak resume where they stopped):
  fetch    download one MInDS-14 config (en-US: 34 MB, 563 clips) and write 16 kHz WAVs + the manifest
  respond  write a short spoken reply per clip with a small instruct LLM (CPU is fine for 0.5B)
  speak    synthesize each reply with Kokoro-82M on CPU
  check    run the quality rules, print a table of failures, store them in the qc field
  split    grouped + stratified train/validation/test split; also writes <data>-<split>/manifest.jsonl (id, audio, text)

Example:  python scripts/stage1_dataset.py fetch --out $S2S_WORK/data/minds14-en --limit 20
"""

import argparse
import hashlib
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = "PolyAI/minds14"  # CC BY 4.0
INTENTS = [
    "abroad",
    "address",
    "app_error",
    "atm_limit",
    "balance",
    "business_loan",
    "card_issues",
    "cash_deposit",
    "direct_debit",
    "freeze",
    "high_value_payment",
    "joint_account",
    "latest_transactions",
    "pay_bill",
]
SR = 16000
SYSTEM_PROMPT = (
    "You are the voice assistant of a retail bank, speaking on the phone. "
    "The caller's request has the intent '{intent}'. Reply in one or two short spoken sentences, "
    "at most 30 words. Ask for what you need to help. No lists, no markdown, no invented numbers."
)


def load(data: Path) -> list:
    return [json.loads(line) for line in (data / "manifest.jsonl").read_text().splitlines() if line.strip()]


def save(data: Path, rows: list) -> None:
    tmp = data / "manifest.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(data / "manifest.jsonl")  # atomic: an interrupted run never leaves half a manifest


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def fetch(args) -> None:
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr
    from huggingface_hub import hf_hub_download

    parquet = hf_hub_download(REPO, f"{args.config}/train-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(parquet).to_pylist()
    if args.limit:
        table = table[: args.limit]
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in table:
        clip_id = f"minds14-{args.config}-{Path(rec['path']).stem}"
        wav, rate = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if rate != SR:
            wav = soxr.resample(wav, rate, SR)
        rel = f"audio/{clip_id}.wav"
        sf.write(args.out / rel, np.clip(wav, -1, 1), SR, subtype="PCM_16")
        rows.append(
            {
                "id": clip_id,
                "audio": rel,
                "text": rec["transcription"].strip(),
                "intent": INTENTS[rec["intent_class"]],
                "duration": round(len(wav) / SR, 3),
                "source_sr": rate,
            }
        )
    save(args.out, rows)
    print(f"fetched {len(rows)} clips into {args.out} ({sum(r['duration'] for r in rows) / 60:.1f} min of audio)")


def respond(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load(args.data)
    todo = [r for r in rows if not r.get("response")][: args.limit or None]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32 if args.device == "cpu" else "auto")
    model.to(args.device).eval()
    start, new_tokens = time.time(), 0
    for n, row in enumerate(todo, 1):
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT.format(intent=row["intent"].replace("_", " "))},
            {"role": "user", "content": row["text"]},
        ]
        ids = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(args.device) for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=60, do_sample=False)
        reply = out[0, ids["input_ids"].shape[1] :]
        new_tokens += len(reply)
        row["response"] = tok.decode(reply, skip_special_tokens=True).strip()
        row["response_model"] = args.model
        if n % 20 == 0 or n == len(todo):
            save(args.data, rows)  # resume point
            print(f"{n}/{len(todo)} replies, {new_tokens / (time.time() - start):.1f} tok/s", flush=True)
    save(args.data, rows)
    if todo:
        print(f"example: {todo[0]['text']!r} -> {todo[0]['response']!r}")


def speak(args) -> None:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    rows = load(args.data)
    todo = [r for r in rows if r.get("response") and not r.get("response_audio")][: args.limit or None]
    tts = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")  # "a" = American English
    (args.data / "response_audio").mkdir(exist_ok=True)
    start, audio_s = time.time(), 0.0
    for n, row in enumerate(todo, 1):
        wav = np.concatenate([a.numpy() for _, _, a in tts(row["response"], voice=args.voice)])
        rel = f"response_audio/{row['id']}.wav"
        sf.write(args.data / rel, wav, 24000, subtype="PCM_16")
        audio_s += len(wav) / 24000
        row["response_audio"], row["response_voice"] = rel, args.voice
        if n % 20 == 0 or n == len(todo):
            save(args.data, rows)
            print(f"{n}/{len(todo)} replies spoken, RTF {(time.time() - start) / audio_s:.2f}", flush=True)
    save(args.data, rows)


def audio_rules(path: Path, prefix: str, sr: int, short: float, long: float) -> tuple:
    """Return (failed rules, seconds, sha256) for one WAV file."""
    import numpy as np
    import soundfile as sf

    if not path.is_file():
        return [f"{prefix}missing"], 0.0, None
    try:
        wav, rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return [f"{prefix}unreadable"], 0.0, None
    bad, secs = [], len(wav) / rate
    if rate != sr or wav.shape[1] != 1:
        bad.append(f"{prefix}format")
    if not short <= secs <= long:
        bad.append(f"{prefix}duration")
    mono = wav[:, 0]
    if 20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-12) < -45:  # RMS below -45 dBFS: near silence
        bad.append(f"{prefix}silent")
    if np.mean(np.abs(mono) >= 0.999) > 0.001:  # more than 0.1% of samples at full scale
        bad.append(f"{prefix}clipped")
    return bad, secs, hashlib.sha256(path.read_bytes()).hexdigest()


def check(args) -> None:
    rows = load(args.data)
    seen, fails = {}, Counter()
    want_reply = any("response" in r for r in rows)  # once respond ran, every clip needs a reply
    want_audio = any("response_audio" in r for r in rows)
    for row in rows:
        bad, secs, digest = audio_rules(args.data / row["audio"], "audio_", SR, 0.5, 30.0)
        words = len(norm(row["text"]).split())
        if words == 0:
            bad.append("text_empty")
        elif secs and not 0.5 <= words / secs <= 5.0:  # words per second outside normal speech
            bad.append("text_rate")
        if row.get("intent") not in INTENTS:
            bad.append("intent_unknown")
        if digest in seen:
            bad.append("audio_duplicate")
        seen.setdefault(digest, row["id"])
        reply = row.get("response") or ""
        if want_reply and not reply.strip():
            bad.append("response_missing")
        elif len(reply.split()) > 40:
            bad.append("response_long")
        if re.search(r"[*#`]|^\s*[-\d]+[.)]\s", reply, re.M):
            bad.append("response_markup")
        if row.get("response_audio"):
            bad += audio_rules(args.data / row["response_audio"], "reply_audio_", 24000, 0.5, 30.0)[0]
        elif want_audio:
            bad.append("reply_audio_missing")
        row["qc"] = bad
        fails.update(bad)
    save(args.data, rows)
    passed = sum(not r["qc"] for r in rows)
    print(f"checked {len(rows)} clips: {passed} pass, {len(rows) - passed} fail")
    for rule, count in fails.most_common():
        print(f"  {rule:<22}{count}")
    hours = sum(r.get("duration", 0) for r in rows if not r["qc"]) / 3600
    print(
        f"clean audio: {hours * 60:.1f} min; intents: {dict(sorted(Counter(r['intent'] for r in rows if not r['qc']).items()))}"
    )


def split(args) -> None:
    rows = load(args.data)
    if any("qc" not in r for r in rows):
        sys.exit("run check first: some clips have no qc field")
    names = ("train", "validation", "test")
    ratios = dict(zip(names, args.ratios))
    # Group clips that say the same thing, so one sentence never sits in two splits; then fill each
    # intent's splits in a fixed pseudo-random order, so every split sees every intent.
    groups = defaultdict(list)
    for row in rows:
        row.pop("split", None)
        if not row["qc"]:
            groups[(row["intent"], norm(row["text"]))].append(row)
    by_intent = defaultdict(list)
    for (intent, text), members in groups.items():
        by_intent[intent].append(members)
    for intent, members in by_intent.items():
        members.sort(key=lambda g: hashlib.sha256(f"{args.seed}:{norm(g[0]['text'])}".encode()).hexdigest())
        total, filled = sum(len(g) for g in members), Counter()
        for group in members:
            # the split furthest below its target share takes the next group
            name = min(names, key=lambda s: filled[s] / total - ratios[s])
            for row in group:
                row["split"] = name
            filled[name] += len(group)
    save(args.data, rows)
    table = Counter((r["split"], r["intent"]) for r in rows if "split" in r)
    print(f"{'intent':<22}" + "".join(f"{s:>12}" for s in names))
    for intent in sorted({i for _, i in table}):
        print(f"{intent:<22}" + "".join(f"{table[(s, intent)]:>12}" for s in names))
    print(f"{'total':<22}" + "".join(f"{sum(v for (s, _), v in table.items() if s == n):>12}" for n in names))
    texts = {n: {norm(r["text"]) for r in rows if r.get("split") == n} for n in names}
    leaks = len(texts["train"] & texts["test"]) + len(texts["train"] & texts["validation"])
    print(f"transcripts shared between train and validation/test: {leaks}")
    for name in names:  # one lab-run dataset per split; audio stays in the parent folder
        out = args.data.parent / f"{args.data.name}-{name}"
        out.mkdir(exist_ok=True)
        with open(out / "manifest.jsonl", "w") as f:
            for r in rows:
                if r.get("split") == name:
                    f.write(
                        json.dumps(
                            {"id": r["id"], "audio": f"../{args.data.name}/{r['audio']}", "text": r["text"]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        print(f"wrote {out}/manifest.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", default="en-US", help="MInDS-14 config, e.g. en-US, en-GB, en-AU")
    p.add_argument("--limit", type=int, default=0, help="only the first N clips (0 = all)")
    p = sub.add_parser("respond")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("speak")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("check")
    p.add_argument("--data", type=Path, required=True)
    p = sub.add_parser("split")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()
    {"fetch": fetch, "respond": respond, "speak": speak, "check": check, "split": split}[args.cmd](args)


if __name__ == "__main__":
    main()
