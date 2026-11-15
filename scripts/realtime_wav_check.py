"""Send one WAV file to a running Realtime server and save the spoken reply.

Use it to check a server without a microphone:

    python scripts/realtime_wav_check.py question.wav --out reply.wav

The script streams the WAV at real-time speed as 16 kHz PCM16 (the server's
native rate), then 1.5 s of silence so server VAD ends the turn. It prints the
user transcript, the assistant transcript, and two timings, both measured from
the end of the speech audio:

* first audio: until the first ``response.output_audio.delta``
* total: until ``response.done``
"""

import argparse
import asyncio
import base64
import json
import sys
import time
import wave

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly

RATE = 16000
CHUNK_MS = 20
CHUNK_SAMPLES = RATE * CHUNK_MS // 1000
TRAILING_SILENCE_MS = 1500


def load_pcm16(path: str) -> bytes:
    data, src_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    if src_rate != RATE:
        mono = resample_poly(mono, RATE, src_rate)
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def append_event(chunk: bytes) -> str:
    return json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})


async def run(args: argparse.Namespace) -> int:
    pcm = load_pcm16(args.wav)
    chunk_bytes = CHUNK_SAMPLES * 2
    reply = bytearray()
    user_text = assistant_text = ""
    speech_end = first_audio = None

    async with websockets.connect(args.url, max_size=2**24) as ws:
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if first.get("type") != "session.created":
            print(f"unexpected first event: {first}", file=sys.stderr)
            return 1

        async def send() -> None:
            nonlocal speech_end
            for i in range(0, len(pcm), chunk_bytes):
                await ws.send(append_event(pcm[i : i + chunk_bytes].ljust(chunk_bytes, b"\x00")))
                await asyncio.sleep(CHUNK_MS / 1000)
            speech_end = time.monotonic()
            for _ in range(TRAILING_SILENCE_MS // CHUNK_MS):
                await ws.send(append_event(b"\x00" * chunk_bytes))
                await asyncio.sleep(CHUNK_MS / 1000)

        sender = asyncio.create_task(send())
        deadline = time.monotonic() + args.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("timeout waiting for response.done", file=sys.stderr)
                return 1
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            kind = event.get("type", "")
            if kind == "conversation.item.input_audio_transcription.completed":
                user_text += event.get("transcript", "")
            elif kind in ("response.output_audio.delta", "response.audio.delta"):
                if first_audio is None:
                    first_audio = time.monotonic()
                reply.extend(base64.b64decode(event.get("delta", "")))
            elif kind in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
                assistant_text += event.get("delta", "")
            elif kind == "error":
                print(f"server error: {event.get('error')}", file=sys.stderr)
                return 1
            elif kind == "response.done":
                done = time.monotonic()
                break
        sender.cancel()

    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(reply))

    print(f"USER: {user_text.strip()}")
    print(f"ASSISTANT: {assistant_text.strip()}")
    print(f"reply audio: {len(reply) / 2 / RATE:.2f} s -> {args.out}")
    if first_audio is not None and speech_end is not None:
        print(f"first audio after end of speech: {first_audio - speech_end:.2f} s")
    if speech_end is not None:
        print(f"total (end of speech -> response.done): {done - speech_end:.2f} s")
    return 0 if reply else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav", help="Input WAV (any rate; mono or stereo).")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime")
    parser.add_argument("--out", default="reply.wav", help="Where to save the reply audio (16 kHz WAV).")
    parser.add_argument("--timeout", type=float, default=120.0, help="Seconds to wait for response.done.")
    sys.exit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
