# Stage 0 walkthrough: measure the upstream pipeline

Part of [ROADMAP.md](ROADMAP.md), Stage 0. Do it first, before [Stage 1](stage1-walkthrough.md).
You run every step yourself. Each step gives the command, the expected output, and how to tell that it worked.
Writing rule: short sentences. Expected outputs come from the reference run on 2026-11-13 (row 1 of [baselines.md](baselines.md)). Your numbers will differ a little. The reference latency run used the small LLM (`Qwen/Qwen3.5-0.8B`), because the GPU never had the ~24 GiB the default `Qwen/Qwen3.5-4B` needs.

What you do:
1. Install the upstream pipeline, unchanged, with torch pinned for driver 550.
2. Measure the STT word error rate (WER) on 100 LibriSpeech utterances.
3. Measure first-audio latency of the full pipeline (VAD → STT → LLM → TTS) on 40 spoken questions.
4. Write one row in `docs/baselines.md`.

No microphone is needed. The latency test streams recorded questions to the server over its WebSocket.
No private data is used. Every dataset is public.

Paths:

| Path | What |
|---|---|
| `~/s2s` (`$S2S`) | This repo (code only) |
| `~/s2s-work` (`$S2S_WORK`) | Everything that is not code |
| `$S2S_WORK/envs/s2s` | Pipeline env (CUDA torch 2.9.1), ~7.9 GB |
| `$S2S_WORK/cache/hf` (`$HF_HOME`) | Model and dataset downloads, ~21 GB after this stage (all models below) |
| `$S2S_WORK/cache/uv`, `cache/pip` | Package caches (safe to delete) |
| `$S2S_WORK/runs/s0/` | Reports and logs of this stage |

Downloads (measured in the cache): Parakeet TDT 0.6B v3 2.4 GB; Qwen3.5-4B 8.8 GB (default LLM); Qwen3.5-0.8B 1.7 GB (small swap); Qwen3-TTS 1.7B CustomVoice 4.3 GB plus its GGUF files from `Serveurperso/Qwen3-TTS-GGUF` 4.0 GB; Smart Turn 8 MB; Silero VAD (torch hub) 36 MB; LibriSpeech test-clean first row group (100 utterances) ~13 MB; the 40 Big Bench Audio questions a few MB. The disk is shared and 93% full: download nothing else.

## Every session: start here

Run this block at the start of every terminal session (step 3 creates `env.sh`):

```bash
source ~/s2s-work/env.sh
gpufree; df -h $HOME | tail -1
```

- Expected: `GPU used=… MiB, total=80995 MiB, free=… MiB`, then one line for the disk that holds `$HOME`.
- It worked if: you see both lines. `free` is the GPU memory you may use now. Other users share this GPU.

## Steps

1. **Check the machine.**
   ```bash
   nvidia-smi | head -4 | tail -1
   nvidia-smi -L
   cat /sys/fs/cgroup/cpu.max
   df -h $HOME | tail -1
   ```
   Expected: `NVIDIA-SMI 550.163.01  Driver Version: 550.163.01  CUDA Version: 12.4`; `GPU 0: NVIDIA H100 80GB HBM3 …` and `MIG 7g.80gb Device 0`; `800000 100000`; one line for the disk that holds `$HOME`.
   It worked if: the driver is 550 and the CPU line says 8 CPUs. Driver 550 = CUDA 12.4: CUDA 13 wheels do not run here. Use 8 threads at most, even though `nproc` says 224.

2. **Clone this repo** to `~/s2s`, and make the work folder.
   ```bash
   git clone https://github.com/Huy-Hieuu/realtime-voice-agent.git ~/s2s
   mkdir -p ~/s2s-work/{envs,cache,data,runs/s0,notes}
   git -C ~/s2s log -1 --oneline
   ```
   Expected: one commit line.
   It worked if: `~/s2s/scripts/stage0_wer.py` exists.

3. **Write the session file.** It keeps every cache under `$HOME/s2s-work` and defines `gpufree`.
   ```bash
   cat > ~/s2s-work/env.sh <<'EOF'
   export S2S=$HOME/s2s S2S_WORK=$HOME/s2s-work
   export HF_HOME=$S2S_WORK/cache/hf TORCH_HOME=$S2S_WORK/cache/torch PIP_CACHE_DIR=$S2S_WORK/cache/pip
   export UV_CACHE_DIR=$S2S_WORK/cache/uv UV_PYTHON_INSTALL_DIR=$S2S_WORK/cache/python
   export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8          # the real CPU quota, not nproc's 224
   export PY=$S2S_WORK/envs/data/bin/python DATA=$S2S_WORK/data/minds14-en
   # Free GPU memory from the MIG table. It makes no CUDA context, so it is safe to run any time.
   gpufree() { nvidia-smi | awk '/MIG devices/{f=1} f && /MiB \/ /{gsub(/\|/,""); gsub(/MiB/,""); print "GPU used=" $5 " MiB, total=" $7 " MiB, free=" $7-$5 " MiB"; exit}'; }
   EOF
   source ~/s2s-work/env.sh && echo $HF_HOME $OMP_NUM_THREADS && gpufree
   ```
   Expected: `<your home>/s2s-work/cache/hf 8`, then the `GPU used=…` line.
   It worked if: `HF_HOME` is under `$HOME/s2s-work`. Stage 1 uses the same file (`PY` and `DATA` are for Stage 1).

4. **Build the pipeline env.** It installs the repo with torch pinned to 2.9.1 (CUDA 12.8 runtime, runs on driver 550) and the Qwen3-TTS GGML wheel built for CUDA 12.4. `jiwer` and `pyarrow` are for step 6.
   Run `uv` from `$S2S_WORK`, not inside `~/s2s`: the repo asks for uv ≥ 0.13.4, and the installed uv 0.12.16 refuses to run there.
   ```bash
   cd $S2S_WORK
   uv venv --python 3.11 envs/s2s
   uv pip install --python envs/s2s/bin/python -e "$S2S" torch==2.9.1 torchaudio==2.9.1 jiwer pyarrow \
     "https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/resolve/main/whl/cu124/qwentts_cpp_python-0.5.0%2Bcu124-py3-none-manylinux_2_35_x86_64.whl"
   uv pip freeze --python envs/s2s/bin/python > envs/s2s.lock
   export S2SPY=$S2S_WORK/envs/s2s/bin/python
   $S2SPY -c 'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)'
   ```
   Expected: about 1 minute (~4 GB download). The last line is `2.9.1+cu128 12.8 5.22.2`.
   It worked if: torch is `2.9.1+cu128`. If it shows `2.11` or `+cu130`, the pin was missing: delete `envs/s2s` and repeat. Never run `pip install -U torch` in this env.
   Add `export S2SPY=$S2S_WORK/envs/s2s/bin/python` to `env.sh`, so later sessions have it.
   `du -sh envs/s2s cache/uv` shows about 7.9 GB each. `uv cache clean` frees the second 7.9 GB if disk is short.

5. **Check free GPU memory.** Do this before every GPU step below.
   ```bash
   gpufree
   $S2SPY -c 'import torch; f, t = torch.cuda.mem_get_info(); print(f"free {f/2**30:.1f} GiB of {t/2**30:.1f} GiB")'
   ```
   Expected: two lines with the same free memory, e.g. `free 18.2 GiB of 79.1 GiB` (the reference run's start).
   It worked if: free is at least **24 GiB** for the default stack (the planning figure for Qwen3.5-4B is "24 GB of VRAM"; not measured here). The small-LLM stack used about **9.6 GiB** (drop in free memory during the reference run). If free is less than you need, wait and check again every 10 minutes (or use the loop in [When the GPU is busy](#when-the-gpu-is-busy)). Step 6 alone needs about 3 GiB.

6. **Measure STT WER** on the first 100 utterances of LibriSpeech test-clean. `scripts/stage0_wer.py` runs the pipeline's own STT handler (`ParakeetTDTSTTHandler`, the `--stt parakeet-tdt` default) and scores it with `jiwer`.
   ```bash
   gpufree
   $S2SPY $S2S/scripts/stage0_wer.py --limit 100 --device cuda --out $S2S_WORK/runs/s0/wer.json 2>&1 | tee $S2S_WORK/runs/s0/wer.log | tail -16
   ```
   Expected: one `USER: …` line per utterance, then:
   ```text
   {
     "dataset": "openslr/librispeech_asr@71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
     "file": "clean/test/0000.parquet",
     "samples": 100,
     "audio_minutes": 11.18,
     "model": "nvidia/parakeet-tdt-0.6b-v3",
     "device": "cuda",
     "wer": 0.0182,
     "cer": 0.0059,
     "latency_p50_s": 0.2768,
     "latency_p90_s": 0.5909,
     "rtfx": 20.0
   }
   wrote <your home>/s2s-work/runs/s0/wer.json
   ```
   It worked if: `samples` is 100, `device` is `cuda` and `wer` is near 0.018 (1.8%). The run takes about 1 minute. LibriSpeech test-clean is clean read speech, so a few percent is normal. The same run on CPU (`--device cpu`) gave the same WER and almost the same speed (p50 0.30 s vs 0.28 s). Probable cause: another user's job kept the GPU busy at the time [INFERENCE].
   Read 5 errors yourself:
   ```bash
   $S2SPY -c "
   import json, sys; sys.path.insert(0, '$S2S/scripts'); from stage0_wer import normalize
   for i in json.load(open('$S2S_WORK/runs/s0/wer.json'))['items']:
       if normalize(i['ref']) != normalize(i['hyp']): print(i['ref'], '\n  ', i['hyp'])
   " | head -10
   ```
   Expected: lines such as `… RAOUL …` → `… Rao …` (a name) and `HONOR` → `honour` (British spelling). Note in `$S2S_WORK/notes/stage0.md` what kind of errors they are (names, spelling variants, numbers written as digits, real mistakes). Stage 2 starts from these.

7. **Measure first-audio latency** of the whole pipeline. The repo's Big Bench Audio eval starts `speech-to-speech serve` with the default models, then streams 40 recorded questions over the Realtime WebSocket, one fresh session each, at real-time speed. `ttfb` is the time from the end of the question audio to the first reply audio.
   ```bash
   gpufree
   $S2SPY -m speech_to_speech.evals.big_bench_audio run --spawn --limit 40 \
     --out $S2S_WORK/runs/s0/bba40.json --spawn-log $S2S_WORK/runs/s0/server.log \
     -- --stt parakeet-tdt --tts qwen3
   ```
   Expected: `spawning: …`, `waiting for ws://127.0.0.1:8765/v1/realtime ...`, a wait while the models load (first time: downloads, see the top of this page), `server ready`, one line per question, then the summary. Reference run (small LLM):
   ```text
   [ 40/40] + web_of_lies       id=786  ttfb 3.5s   yes == Yes

   wrote <your home>/s2s-work/runs/s0/bba40.json

   Big Bench Audio vibe check -- vibe@2026-11-13T13:44
     subset   vibe (40 items) @ ArtificialAnalysis/big_bench_audio@af7bb9c
     target   ws://127.0.0.1:8765/v1/realtime

     category              acc   correct   parsed  errors
     ------------------ ------ --------- -------- -------
     formal_fallacies    60.0%    6/10     10/10        0
     navigate            50.0%    5/10      8/10        0
     object_counting     40.0%    4/10     10/10        0
     web_of_lies         20.0%    2/10      5/10        0
     ------------------ ------ --------- -------- -------
     OVERALL             42.5%   17/40     33/40        0

     latency  ttfb p50  2.89s  p95  4.82s   turn p50  8.88s  p95 23.14s
     wall     28.2 min
   ```
   The whole run takes about 30 minutes: the questions are sent at real-time speed. The server stops at the end and frees the GPU.
   It worked if: the summary shows 40 items and `errors` is 0, and after the run `gpufree` is back near its value at the start. (Under MIG, `nvidia-smi` lists no processes, so use the free memory.) An exit code of 1 means at least one turn failed (no audio, timeout, split turn): read `server.log`.
   The server log warns that `flash-linear-attention` is not installed, so Qwen3.5 uses a slower reference kernel. That is upstream behaviour: keep it for the baseline.
   If the server fails with `CUDA out of memory`, the GPU is busy: wait (step 5). If free memory stays between 10 and 24 GiB, run once with the small LLM and write the swap in the notes column: add `--model_name Qwen/Qwen3.5-0.8B` after `--tts qwen3`.

8. **Get p50/p90 and the per-stage times.** The report prints p50 and p95; p90 comes from the per-question values in the JSON.
   ```bash
   $S2SPY - <<'EOF'
   import json, os, re, statistics
   run = os.path.expandvars("$S2S_WORK/runs/s0")
   r = json.load(open(f"{run}/bba40.json"))
   items = r["items"]
   t = sorted(i["ttfb_s"] for i in items if i.get("ttfb_s") is not None)
   q = statistics.quantiles(t, n=10)
   print(f"ttfb n={len(t)} p50={statistics.median(t):.2f}s p90={q[8]:.2f}s")
   e2e = [float(m) for m in re.findall(r" e2e=([0-9.]+)s", open(f"{run}/server.log").read())]
   if len(e2e) > 1:
       q = statistics.quantiles(e2e, n=10)
       print(f"server e2e n={len(e2e)} p50={statistics.median(e2e):.2f}s p90={q[8]:.2f}s")
   EOF
   grep -m3 ' latency: ' $S2S_WORK/runs/s0/server.log
   ```
   Expected:
   ```text
   ttfb n=40 p50=2.89s p90=4.43s
   server e2e n=40 p50=3.17s p90=4.85s
   … stt=0.85s llm=4.04s llm_first_tts=3.61s llm_total=4.04s llm_rounds=1 tts_ttfa=0.40s e2e=6.50s … vad_decision=0.96s …
   ```
   It worked if: both lines print. `ttfb` is measured by the client; server `e2e` is measured from the estimated end of speech ([response-latency.md](response-latency.md)). The `latency:` lines split each turn into `stt=`, `llm=`, `llm_first_tts=` and `tts_ttfa=`. Note which stage is the biggest. Stage 5 works on it.
   Reference p50 over 40 turns: `vad_decision` 0.96 s, `stt` 0.30 s, `llm_first_tts` 1.40 s, `tts_ttfa` 0.34 s. The LLM wait for the first sentence batch (`--stream_batch_sentences 3`) is the biggest part. Stages overlap: do not add them up.

9. **Write your row** in `docs/baselines.md`, on a branch:
   ```bash
   git -C $S2S checkout -b stage0-baseline
   git -C $S2S rev-parse --short HEAD; sha256sum $S2S_WORK/envs/s2s.lock | cut -c1-12
   ```
   Add one row with: today's date, the short SHA, the models (and any swap), `gpufree` from step 5, ttfb p50/p90 from step 8, WER and sample count from step 6, and the lock hash in the notes. Commit it, push the branch, open a pull request.
   It worked if: your row sits under the reference row and every column is filled. Do not paste numbers you did not measure. If a step could not run, write why (for example, `not measured: GPU full (free 1.1 GiB at 12:00Z)`).

## When the GPU is busy

Wait without making a CUDA context. This loop checks every 10 minutes and prints `GPU free` when at least 24 GiB is free:

```bash
until [ "$(nvidia-smi | awk '/MIG devices/{f=1} f && /MiB \/ /{gsub(/\|/,""); gsub(/MiB/,""); print $7-$5; exit}')" -gt 24000 ]; do sleep 600; done; echo GPU free
```

Meanwhile, do Stage 1 Days 1-2: they need no GPU.

## Clean up

- The server stops by itself after step 7. Check: `gpufree` is back near its value before the run.
- Keep `envs/s2s` and `cache/hf`: Stages 2-5 reuse them. `uv cache clean` frees ~7.9 GB of package cache.
- Do not commit `$S2S_WORK` (envs, caches, reports) to git. Only the row in `docs/baselines.md` goes into the repo.
