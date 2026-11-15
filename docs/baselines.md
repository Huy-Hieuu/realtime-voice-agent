# Baselines

Measured numbers for the upstream pipeline and every later change. One row per run. Never edit an old row: add a new one.
How to measure: [stage0-baseline.md](stage0-baseline.md). Roadmap: [ROADMAP.md](ROADMAP.md).

Columns:
- **GPU free at start:** `gpufree` just before the GPU steps. The GPU is a shared MIG 7g.80gb (79.1 GiB usable).
- **First-audio latency:** Big Bench Audio `vibe` sample, first N questions, `--speed 1`. Client-side `ttfb` = end of question audio → first reply audio delta. p90 from the per-item values in the report JSON.
- **STT WER:** `scripts/stage0_wer.py`, first N utterances of LibriSpeech test-clean (`openslr/librispeech_asr@71cacbfb`, file order), normalized (lowercase, letters/digits/apostrophes), corpus-level `jiwer`.

| Date (UTC) | Git SHA | Models (VAD / STT / LLM / TTS) | GPU free at start | First-audio p50 / p90 | STT WER (set, n) | Notes |
|---|---|---|---|---|---|---|
| 2026-11-13 | `7681845` (upstream, unchanged) | Silero + Smart Turn v3.2 / `nvidia/parakeet-tdt-0.6b-v3` / **`Qwen/Qwen3.5-0.8B`** (swap) / `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (GGML) | 18.2 GiB at 13:43Z (latency run); 8.0 GiB at 12:43Z (WER run) | **2.89 s / 4.43 s** (n=40); server `e2e` 3.17 s / 4.85 s | **1.82%** WER, 0.59% CER (LibriSpeech test-clean, n=100, 11.2 min) | Default `Qwen/Qwen3.5-4B` not measured: free GPU memory stayed 1.1-19.2 GiB from 11:38Z to 13:44Z, below its ~24 GB. Big Bench Audio accuracy 42.5% (17/40, 0 errors). Stage p50: `vad_decision` 0.96 s, `stt` 0.30 s, `llm_first_tts` 1.40 s, `tts_ttfa` 0.34 s. Pipeline used ~9.6 GiB GPU (drop in MIG free memory). STT p50 0.28 s/utterance on GPU, 0.30 s on CPU; probably the other user's job kept the GPU busy [INFERENCE]. torch 2.9.1+cu128, transformers 5.22.2, nano-parakeet 0.2.1, faster-qwen3-tts 0.5.4, qwentts-cpp-python 0.5.0+cu124; env lock sha256 `2798463a7a41`. |
