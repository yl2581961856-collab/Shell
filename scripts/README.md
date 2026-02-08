# Helper scripts

Python helpers (this folder):
- `ws_mic_client.py`: microphone streaming client for `/asr/stream`.
- `eval_sensevoice_small.py`: SenseVoice-small pseudo-streaming evaluator.
- `eval_fun_asr_nano.py`: Fun-ASR-Nano pseudo-streaming evaluator.

Shell launchers (see `shell/`):
- `shell/start_server.sh`: start the FastAPI/WS server via `uvicorn` (Python 3.11 target).
- `shell/start_server_6008.sh`: start server on `0.0.0.0:6008`.
- `shell/start_server_debug.sh`: start server with DEBUG logs.
- `shell/bootstrap_py311.sh`: create a local venv on Python 3.11, install deps, and start the server.
