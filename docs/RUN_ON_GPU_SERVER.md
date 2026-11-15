# Run speech-to-speech on a GPU server and talk to it from your Mac

The server runs on a remote Linux GPU server. Your Mac runs only the microphone and speaker client. An SSH tunnel connects them.

The server uses the default pipeline (see the [README](../README.md#architecture)): Silero VAD → Parakeet TDT 0.6B v3 → Qwen3.5-4B → Qwen3-TTS 1.7B CustomVoice (GGML backend). No `HF_TOKEN` is necessary.

## What is on the GPU server

| Path | What it is |
| --- | --- |
| `$HOME/s2s-run/venv` | Python 3.11 env: this repo, `torch==2.9.1` (CUDA 12.8 runtime, runs on driver 550), Qwen3-TTS GGML wheel for CUDA 12.4 |
| `$HOME/s2s-run/speech-to-speech` | A copy of this repo. The server runs from it, not from a git worktree. |
| `$HOME/s2s-run/start.sh` | Starts (or restarts) the server in the tmux session `s2s-server` |
| `$HOME/s2s-run/server.log` | Server log |
| `$HOME/s2s-work/cache/hf` | Model cache (from [Stage 0](stage0-baseline.md)). `start.sh` sets `HF_HOME` to it. |

The server listens on `ws://127.0.0.1:8765/v1/realtime`. It accepts one session at a time.

## 1. Check that the server runs (on the GPU server)

```bash
tmux ls
grep -c "Uvicorn running" $HOME/s2s-run/server.log
```

Expected: `s2s-server: 1 windows ...` and a number that is 1 or more.

If `tmux ls` shows no `s2s-server` (for example, after a server restart), start it:

```bash
$HOME/s2s-run/start.sh
tail -f $HOME/s2s-run/server.log
```

Wait for `Uvicorn running on http://127.0.0.1:8765`. This takes about 1 minute. Stop `tail` with `Ctrl+C`; the server continues to run.

## 2. Install the client on your Mac (one time)

Use Python 3.11.

```bash
git clone https://github.com/Huy-Hieuu/realtime-voice-agent.git ~/s2s-client-src
python3.11 -m venv ~/s2s-client
source ~/s2s-client/bin/activate
pip install -e ~/s2s-client-src
```

The project contains the full pipeline, so the install takes some minutes.

## 3. Open the SSH tunnel (on your Mac, terminal 1)

```bash
ssh -N -L 8765:127.0.0.1:8765 <your-host>
```

Replace `<your-host>` with your own SSH alias for the GPU server. The command shows no output. Keep this terminal open.

If port 8765 is already in use on your Mac, use `-L 18765:127.0.0.1:8765` and use port `18765` in step 4.

## 4. Talk (on your Mac, terminal 2)

Use headphones. Without headphones, add `--echo-cancellation` (see [echo cancellation](echo-cancellation.md)).

```bash
source ~/s2s-client/bin/activate
speech-to-speech talk --url ws://localhost:8765/v1/realtime
```

Expected: `Ready — start speaking.` Then speak one question, for example "Why is the sky blue?", and pause.

Your words show in yellow. The reply shows in green, and you hear it in your headphones. The first audio comes about 3 to 4 seconds after you stop speaking.

macOS asks for microphone access for your terminal the first time. Allow it. If you do not hear anything, check **System Settings → Privacy & Security → Microphone**, then run `talk` again.

Stop the client with `Ctrl+C`.

## 5. How to tell it worked

- The client shows your words (yellow) and a reply (green), and you hear the reply.
- On the GPU server, each turn adds one `latency:` line to the log:

  ```bash
  grep " latency: " $HOME/s2s-run/server.log | tail -3
  ```

To check the server without a microphone, run this on the GPU server. It sends a WAV file and saves the spoken reply:

```bash
$HOME/s2s-run/venv/bin/python $HOME/s2s-run/speech-to-speech/scripts/realtime_wav_check.py question.wav --out reply.wav
```

Expected: `USER: ...`, `ASSISTANT: ...`, and the two latency lines.

## 6. Stop and restart (on the GPU server)

| Action | Command |
| --- | --- |
| Stop | `tmux kill-session -t s2s-server` |
| Start or restart | `$HOME/s2s-run/start.sh` |
| Watch the server live | `tmux attach -t s2s-server` (leave with `Ctrl+B`, then `D`) |
| Use the small LLM | `$HOME/s2s-run/start.sh --model_name Qwen/Qwen3.5-0.8B` |

`start.sh` passes extra arguments to `speech-to-speech serve`. The GPU is shared: stop the server when you do not use it.

If the log shows `CUDA out of memory`, another job uses the GPU. Check the free memory, then try again later, or use the small LLM:

```bash
$HOME/s2s-run/venv/bin/python -c 'import torch; f, t = torch.cuda.mem_get_info(); print(f"free {f/2**30:.1f} GiB of {t/2**30:.1f} GiB")'
```

## Measured numbers (2026-11-14)

Test input: a 4.5 s English question ("Hello! Can you tell me in two short sentences why the sky is blue?"), made with Qwen3-TTS (speaker Ryan). Sent 3 times with `scripts/realtime_wav_check.py`, one new session each time. The times start at the end of the speech audio.

| Run | First audio | Total (to `response.done`) | Reply audio |
| --- | --- | --- | --- |
| 1 (first turn after start) | 4.39 s | 11.93 s | 8.9 s |
| 2 | 3.26 s | 9.59 s | 7.6 s |
| 3 | 3.26 s | 10.38 s | 8.4 s |

Server log for runs 2 and 3: `stt` 0.31 s, `llm_first_tts` 1.55–1.58 s, `tts_ttfa` 0.64–0.66 s, `e2e` 3.27 s. All 3 runs transcribed the question correctly.

GPU memory: free memory went from 42.6 GiB to 27.5 GiB when the server started, so the server uses about **15 GiB** of the 79 GiB MIG device. The GPU is shared, so other jobs can change this number. `nvidia-smi` shows no processes under MIG.

Startup: about 1 minute from `start.sh` to `Uvicorn running` (models already in the cache).
