# Helper scripts

Python helpers (this folder):
- `ws_mic_client.py`: microphone streaming client for `/asr/stream`.
- `eval_sensevoice_small.py`: SenseVoice-small pseudo-streaming evaluator.
- `eval_fun_asr_nano.py`: Fun-ASR-Nano pseudo-streaming evaluator.

Shell launchers (see `shell/`):
- `shell/start_server.sh`: start the FastAPI/WS server via `uvicorn` (Python 3.11 target).
- `shell/bootstrap_py37.sh`: create a local venv on Python 3.7, install deps, and start the server.
- `shell/bootstrap_py311.sh`: create a local venv on Python 3.11, install deps, and start the server.
