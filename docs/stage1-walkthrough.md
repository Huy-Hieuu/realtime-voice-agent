# Stage 1 walkthrough: build the voice-turn dataset (days 1-5)

Part of [ROADMAP.md](ROADMAP.md), Stage 1. You run every step yourself.
Each day is about 4 hours. Each step gives the command, the expected output, and how to tell that it worked.

**Stage 1 needs no GPU.** Every required step runs on the 8 CPUs. The GPU was full at the last check, so the plan does not depend on it.
Two steps have an optional GPU variant. Each starts with the free-memory check.
Labels: **[CPU]** = runs on CPU only. **[GPU, optional]** = faster on GPU; skip it if `gpufree` shows too little memory.

What you build: `$S2S_WORK/data/minds14-en/manifest.jsonl`. One line per voice turn:

| Field | Example | Made by |
|---|---|---|
| `id` | `minds14-en-US-602ba55abb1e6d0fbce92065` | `fetch` |
| `audio` | `audio/<id>.wav` (16 kHz, mono, 16-bit) | `fetch` |
| `text` | `I would like to set up a joint account with my partner` | `fetch` (MInDS-14 transcript) |
| `intent` | `joint_account` (one of 14) | `fetch` |
| `duration`, `source_sr` | `10.837`, `8000` | `fetch` |
| `response`, `response_model` | `Sure, I can help you set up a joint account ...`, `Qwen/Qwen2.5-0.5B-Instruct` | `respond` |
| `response_audio`, `response_voice` | `response_audio/<id>.wav` (24 kHz), `af_heart` | `speak` |
| `qc` | `[]` = passed; else the failed rules | `check` |
| `split` | `train` / `validation` / `test` | `split` |

The first three fields are the plain ASR format `{"id", "audio", "text"}`. Stage 2 reads them directly.

Data source: [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14), config `en-US`. Licence CC BY 4.0: you may share and change it, with attribution. 563 real phone calls to a bank, 8 kHz audio.
Download sizes in this walkthrough: data env ~1.4 GB, MInDS-14 en-US 34 MB, Qwen2.5-0.5B-Instruct ~1 GB, Kokoro-82M ~330 MB. Disk at the end: ~1.4 GB in `$S2S_WORK/envs`, ~2.8 GB in `$S2S_WORK/cache`, ~330 MB of data (150 MB input audio, 180 MB reply audio).

## Every session: start here

Run this block at the start of every terminal session (Day 1 step 3 creates `env.sh`):

```bash
source ~/s2s-work/env.sh
gpufree; df -h $HOME | tail -1
```

- Expected: `GPU used=… MiB, total=80995 MiB, free=… MiB`, then one line for the disk that holds `$HOME`.
- It worked if: you see both lines. `free` is the GPU memory you may use now. Other users share this GPU.

## Day 1 - Setup and read the pipeline [CPU]

1. **Check the machine.**
   ```bash
   nvidia-smi -L
   cat /sys/fs/cgroup/cpu.max
   df -h $HOME | tail -1
   ```
   Expected: `GPU 0: NVIDIA H100 80GB HBM3 …` and `MIG 7g.80gb Device 0`; `800000 100000` (= 8 CPUs); one line for the disk that holds `$HOME`.
   It worked if: the CPU line says 8 CPUs. Use 8 threads at most, even though `nproc` says 224.

2. **Clone this repo** to `~/s2s`, and make the work folder for everything that is not code.
   ```bash
   git clone https://github.com/Huy-Hieuu/realtime-voice-agent.git ~/s2s
   mkdir -p ~/s2s-work/{envs,cache,data,runs,notes}
   git -C ~/s2s log -1 --oneline
   ```
   Expected: one commit line. It worked if: `~/s2s/scripts/stage1_dataset.py` exists.
   If you did [Stage 0](stage0-baseline.md), the clone and `env.sh` already exist: skip steps 2 and 3.

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
   It worked if: `HF_HOME` is under `$HOME/s2s-work`. Downloads then go there and not to `~/.cache`.

4. **Build the data env.** It uses the CPU build of torch 2.9.1: Stage 1 needs no GPU, and the CPU build is ~1 GB smaller than the CUDA build. The last package is the English word model that Kokoro needs; Kokoro tries to install it at run time and fails in a uv env.
   ```bash
   uv venv --python 3.12 $S2S_WORK/envs/data
   uv pip install --python $PY --torch-backend cpu torch==2.9.1 transformers huggingface_hub pyarrow soundfile soxr numpy kokoro \
     "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
   uv pip freeze --python $PY > $S2S_WORK/envs/data.lock
   $PY -c 'import torch, transformers, kokoro; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
   ```
   Expected: the last line is `2.9.1+cpu 5.… False`. `du -sh $S2S_WORK/envs/data` shows about 1.4 GB.
   It worked if: torch ends with `+cpu`. If it shows `+cu128` or `+cu130`, `--torch-backend cpu` was missing: delete the env and repeat.
   To rebuild it later exactly: `uv pip install --python $PY --torch-backend cpu -r $S2S_WORK/envs/data.lock`.
   Run `uv` commands from `$S2S_WORK`, not from inside `~/s2s`. The repo's `pyproject.toml` asks for uv ≥ 0.13.4; the installed uv is 0.12.16, so `uv` fails there with `Required uv version >=0.13.4`.

5. **Read the pipeline** you will feed. Open these files in `~/s2s/src/speech_to_speech/` and write one sentence per file in `$S2S_WORK/notes/stage1.md`:
   - `s2s_pipeline.py`: find `build_pipeline` and `_build_handlers`. What connects the four handlers?
   - `backend_registry.py`: find `STT_BACKENDS`, `LLM_BACKENDS`, `TTS_BACKENDS`. Which names does `--stt` accept?
   - `VAD/vad_handler.py` (`VADHandler`), `STT/whisper_stt_handler.py` (`WhisperSTTHandler`), `LLM/language_model.py` (`LanguageModelHandler`), `TTS/kokoro_handler.py` (`KokoroTTSHandler`).
   - `docs/response-latency.md`: what do `stt=`, `llm_first_tts=`, `tts_ttfa=` and `e2e=` measure?

   It worked if: you can say which dataset field each stage will learn from: `audio`→`text` (STT, Stage 2), `text`+`intent`→`response` (LLM, Stage 3), `response`→`response_audio` (TTS, Stage 4).

## Day 2 - Get the audio and check it [CPU]

1. **Fetch a 20-clip sample first.** A small run finds problems in seconds.
   ```bash
   $PY $S2S/scripts/stage1_dataset.py fetch --out $DATA --limit 20
   head -n 2 $DATA/manifest.jsonl
   ```
   Expected: a warning about unauthenticated requests (OK: the dataset is public), then `fetched 20 clips into <your home>/s2s-work/data/minds14-en (… min of audio)`, then two JSON lines with `id`, `audio`, `text`, `intent`, `duration`, `source_sr: 8000`.
   It worked if: `ls $DATA/audio | wc -l` prints `20`.

2. **Look at one file.**
   ```bash
   f=$(ls $DATA/audio | head -n 1)
   $PY -c 'import soundfile as sf, sys; i = sf.info(sys.argv[1]); print(i.samplerate, i.channels, i.subtype, round(i.duration, 2))' $DATA/audio/$f
   ```
   Expected: `16000 1 PCM_16 <seconds>`.
   It worked if: the rate is 16000. The source is 8 kHz phone audio; `fetch` resamples it to the 16 kHz that every STT model in the pipeline expects. Phone audio has no sound above 4 kHz: remember this when Stage 2 WER is worse here than on LibriSpeech.
   Listen on your laptop: `scp <your-host>:~/s2s-work/data/minds14-en/audio/<id>.wav .`

3. **Fetch all 563 clips.**
   ```bash
   time $PY $S2S/scripts/stage1_dataset.py fetch --out $DATA
   ```
   Expected: `fetched 563 clips into … (80.5 min of audio)`, in about 15 s. `du -sh $DATA` shows about 150 MB.
   It worked if: `wc -l < $DATA/manifest.jsonl` prints `563`.

4. **Run the quality checks** on audio, transcript and intent.
   ```bash
   $PY $S2S/scripts/stage1_dataset.py check --data $DATA
   ```
   Expected:
   ```text
   checked 563 clips: 548 pass, 15 fail
     text_rate             11
     audio_duration        5
     audio_silent          1
   clean audio: 74.5 min; intents: {'abroad': 32, 'address': 33, …}
   ```
   It worked if: the numbers match. Each rule:
   - `audio_duration`: shorter than 0.5 s or longer than 30 s.
   - `audio_silent`: RMS level below -45 dBFS (nearly no sound).
   - `audio_clipped`: more than 0.1% of samples at full scale (distorted).
   - `text_rate`: fewer than 0.5 or more than 5 words per second. This finds transcripts that do not match the audio, or long silences.
   - `text_empty`, `intent_unknown`, `audio_duplicate`, `audio_format`: no failures expected.

5. **Read the failures yourself.** Rules can be wrong.
   ```bash
   $PY -c "
   import json
   for line in open('$DATA/manifest.jsonl'):
       r = json.loads(line)
       if r['qc']: print(r['qc'], r['duration'], repr(r['text'][:70]), r['audio'])
   "
   ```
   Expected: 15 lines. Listen to 3 `text_rate` clips.
   It worked if: you can say for each listened clip whether the rule was right. Write it in `$S2S_WORK/notes/stage1.md`. A failed clip is not deleted: it stays in the manifest with its `qc` reason and does not go into any split.

## Day 3 - Write the responses [CPU; GPU optional]

The reply comes from a small instruct model, [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache-2.0, ~1 GB download). The prompt gives it the intent and asks for one or two spoken sentences, at most 30 words, with no markdown. Read `SYSTEM_PROMPT` in `scripts/stage1_dataset.py` first.

1. **Try 10 replies.**
   ```bash
   time $PY $S2S/scripts/stage1_dataset.py respond --data $DATA --limit 10
   ```
   Expected: weight-loading progress bars, then `10/10 replies, ~20 tok/s` and an example such as
   `'I would like to set up a joint account with my partner' -> 'Sure, I can help you set up a joint account with your partner. Please provide me with their account details …'`. About 30 s, including the download.
   It worked if: the example reply answers the caller and is short. `respond` saves every 20 replies; if you stop it, the next run continues where it stopped.

2. **Write all replies** [CPU]. Run it in `tmux`: it takes about 12 minutes (measured: 553 replies at 18-25 tokens/s on 8 CPUs).
   ```bash
   tmux new -s stage1
   time $PY $S2S/scripts/stage1_dataset.py respond --data $DATA
   ```
   Expected: a progress line every 20 replies (`20/553 replies, … tok/s`), ending at `553/553`.
   It worked if: `grep -c '"response":' $DATA/manifest.jsonl` prints `563`.

   **[GPU, optional]** The same step on the GPU needs about 2 GB free [INFERENCE]. Check first, and run it only if `free` is above 4000 MiB:
   ```bash
   gpufree
   $PY $S2S/scripts/stage1_dataset.py respond --data $DATA --device cuda
   ```
   The data env has the CPU build of torch, so this fails with `Torch not compiled with CUDA enabled`. To use the GPU, run this step with the Stage 0 env (`$S2S_WORK/envs/s2s/bin/python`, CUDA torch 2.9.1) instead of `$PY`. It is optional: the CPU run is fast enough.

3. **Read 20 replies and judge them.**
   ```bash
   $PY -c "
   import json, random
   rows = [json.loads(l) for l in open('$DATA/manifest.jsonl')]
   for r in random.Random(1).sample(rows, 20): print(f\"[{r['intent']}] {r['text'][:60]!r}\n   -> {r['response']}\n\")
   "
   ```
   It worked if: you have a count in your notes: how many of 20 are (a) on-intent, (b) short and speakable, (c) free of invented facts. This is your Stage 3 baseline by hand. The replies are silver labels: good enough to test the plumbing, not a gold standard.

## Day 4 - Speak the responses [CPU]

The reply audio comes from [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0, ~330 MB download). It is the same model as the pipeline's `--tts kokoro` backend (`TTS/kokoro_handler.py`). It runs well on CPU. The voice is `af_heart` (American English).

1. **Speak 10 replies.**
   ```bash
   time $PY $S2S/scripts/stage1_dataset.py speak --data $DATA --limit 10
   ```
   Expected: a `WeightNorm` warning (OK), then `10/10 replies spoken, RTF 0.15` (about 20 s in total).
   It worked if: `ls $DATA/response_audio | wc -l` prints `10`. RTF 0.15 means one second of speech takes 0.15 s to make.

2. **Speak all replies** in `tmux` (about 12 minutes, measured).
   ```bash
   time $PY $S2S/scripts/stage1_dataset.py speak --data $DATA
   ```
   Expected: a progress line every 20 replies, ending at `553/553 replies spoken, RTF …`.
   It worked if: `ls $DATA/response_audio | wc -l` prints `563`.

3. **Run the full quality check.** Now it also checks the replies and the reply audio.
   ```bash
   $PY $S2S/scripts/stage1_dataset.py check --data $DATA
   ```
   Expected (measured on the full run; your reply counts can differ a little):
   ```text
   checked 563 clips: 532 pass, 31 fail
     response_long         16
     text_rate             11
     audio_duration        5
     audio_silent          1
     response_markup       1
   clean audio: 71.5 min; intents: {'abroad': 29, 'address': 33, …}
   ```
   It worked if: the response rules (`response_missing`, `response_long`, `response_markup`, `reply_audio_*`) fail on few clips. Measured: 17 of 563 (3%). If `response_long` or `response_markup` fail on more than 5% of clips, change `SYSTEM_PROMPT` and start again from `fetch` (it writes a new manifest without replies in about 15 s), then repeat Day 3 and 4.

4. **Listen to 5 reply files** on your laptop (`scp <your-host>:~/s2s-work/data/minds14-en/response_audio/<id>.wav .`).
   It worked if: the speech is clear and says the full reply text. Note any mispronounced words (numbers, bank terms) in your notes. Stage 4 measures this with WER on the synthesized audio.

## Day 5 - Split, document and freeze [CPU]

1. **Split the clean clips 80/10/10.**
   ```bash
   $PY $S2S/scripts/stage1_dataset.py split --data $DATA --seed 0
   ```
   Expected: a table of intent × split, then:
   ```text
   total                          424          55          53
   transcripts shared between train and validation/test: 0
   wrote <your home>/s2s-work/data/minds14-en-train/manifest.jsonl
   wrote <your home>/s2s-work/data/minds14-en-validation/manifest.jsonl
   wrote <your home>/s2s-work/data/minds14-en-test/manifest.jsonl
   ```
   It worked if: every intent has at least 3 clips in `validation` and `test`, and the shared-transcript count is `0`.
   How the split works: clips with the same transcript after lowercasing and removing punctuation form one group, and a group never crosses splits. Otherwise a model could memorize a test sentence from training. Inside each intent, groups go to the split that is furthest below its target share, in a fixed pseudo-random order from `--seed`. The same seed always gives the same split.

2. **Prove the split is deterministic.**
   ```bash
   sha256sum $DATA-test/manifest.jsonl
   $PY $S2S/scripts/stage1_dataset.py split --data $DATA --seed 0 > /dev/null
   sha256sum $DATA-test/manifest.jsonl
   ```
   Expected: the same hash twice. It worked if: the hashes are equal. With `--seed 1` the hash changes.

3. **Write the dataset card** in `$DATA/README.md`: source and licence (MInDS-14 en-US, CC BY 4.0, PolyAI, link), the generated fields and their models (Qwen2.5-0.5B-Instruct, Kokoro-82M `af_heart`, both Apache-2.0), the rule thresholds, the split method and seed, the counts from steps 1 and Day 4 step 3, and known limits (8 kHz audio; silver replies; one synthetic voice).
   It worked if: someone else could rebuild the dataset from the card alone.

4. **Freeze this version.** Record what makes it reproducible:
   ```bash
   { date -u +%FT%TZ; git -C $S2S rev-parse HEAD; sha256sum $DATA/manifest.jsonl $DATA-*/manifest.jsonl $S2S_WORK/envs/data.lock; } | tee $S2S_WORK/notes/minds14-en-v1.txt
   ```
   Expected: a time, a commit hash and five file hashes.
   It worked if: the file exists. Stage 6 turns this into a versioned dataset repo; for now this note is the version.

5. **Stage 1 done-check.**
   - `check`: clean clips and failures per rule are in your notes, with your verdict on the rules from Day 2.
   - `split`: three split manifests, every intent in every split, `0` shared transcripts.
   - Every clip in a split has all five parts: audio, transcript, intent, response, response audio.

6. **[CPU] If you have time:** add more accents. `--config en-GB` (592 clips, 35 MB) and `--config en-AU` (654 clips, 37 MB) have the same format. Fetch each into its own folder (`--out $S2S_WORK/data/minds14-en-gb`), because `fetch` rewrites the manifest of its folder. Stage 2 uses them as more training data.

## When the GPU is busy

Stage 1 does not need it. Do the CPU steps. For later stages, wait without making a CUDA context:

```bash
until [ "$(nvidia-smi | awk '/MIG devices/{f=1} f && /MiB \/ /{gsub(/\|/,""); gsub(/MiB/,""); print $7-$5; exit}')" -gt 20000 ]; do sleep 600; done; echo GPU free
```

## Clean up

- The data env and caches stay: Stage 2 and 3 reuse them. To free space: `rm -rf $S2S_WORK/cache/uv` (package cache; ~1 GB).
- Do not commit data, audio or `$S2S_WORK` to git. The manifests hold public CC BY 4.0 text and are safe to share, but keep them out of this repo too: Stage 6 versions them in a dataset repo.
