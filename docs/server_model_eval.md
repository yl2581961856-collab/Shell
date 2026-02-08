# Server Model Eval Prep (SenseVoice / Nano)

This guide helps you prepare server-side pseudo-streaming evaluation for two new models:
- SenseVoice-small
- Fun-ASR-Nano

The evaluation scripts are standalone and do not change the main service.

## 1) Upload + Extract Models
Pick a model root on the server (examples use `/data/models`).

```bash
mkdir -p /data/models
tar -xzvf sensevoice-small.tar.gz -C /data/models
tar -xzvf funasr-nano-2512.tar.gz -C /data/models
```

Record the extracted model directories:
- SenseVoice: `/data/models/<sensevoice_dir>`
- Nano: `/data/models/<nano_dir>`

## 2) Set Environment Variables
These scripts require model paths via env vars.

```bash
export SENSEVOICE_MODEL=/data/models/<sensevoice_dir>
export FUNASR_NANO_MODEL=/data/models/<nano_dir>
export SENSEVOICE_DEVICE=npu:0
export FUNASR_NANO_DEVICE=npu:0
```

Optional (override Python):
```bash
export PYTHON_BIN=python3.11
```

## 3) Run Pseudo-Streaming Eval
Use the helper runner to evaluate both models on the same input.

```bash
chmod +x ./shell/run_model_eval.sh
./shell/run_model_eval.sh --input /path/to/audio.wav --chunk-ms 1200 --update-every 1 --outdir ./eval_logs
```

To play audio while evaluating:

```bash
./shell/run_model_eval.sh --input /path/to/audio.wav --chunk-ms 1200 --update-every 1 --outdir ./eval_logs --play --realtime
```

Outputs:
- `./eval_logs/sensevoice_small_*.jsonl`
- `./eval_logs/funasr_nano_*.jsonl`

Each line is a JSON event: `ack`, `partial`, `final` (includes `ts`, `rtf`, `bytes`, `seq`).

## 4) Suggested Test Matrix
Run a few chunk sizes to compare partial quality and RTF:

```bash
for ms in 200 400 800 1200 1600; do
  ./shell/run_model_eval.sh --input /path/to/audio.wav --chunk-ms "$ms" --update-every 1 --outdir ./eval_logs
done
```

## 5) Metrics to Record
- TTFT: `first partial.ts - ack.ts`
- RTF: `rtf` field in partial/final
- Partial quality: readability and stability across chunks
- Final accuracy: compare to reference transcript

## 6) Notes
- These scripts expect 16k/mono WAV or PCM input.
- For PCM input, pass `--input-format pcm --sr 16000 --ch 1`.
- The server service is not modified in this flow.
- Audio playback requires `sounddevice` or `simpleaudio` on the machine.
