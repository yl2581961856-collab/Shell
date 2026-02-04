# C++ WebSocket Client (websocketpp + Boost)

This is a minimal Linux/GCC and Windows/MSVC client for the ASR WebSocket endpoint.

## Dependencies
- `websocketpp` (header-only)
- `Boost.Asio` (Boost.System)
- CMake + compiler toolchain

## Build (Ubuntu)
1. Install deps:
   - `sudo apt-get update`
   - `sudo apt-get install -y g++ cmake libboost-system-dev`
2. Build:
   - `mkdir -p build`
   - `cd build`
   - `cmake ..`
   - `cmake --build . -j`

## Build (Windows / Visual Studio)
1. Install vcpkg and deps:
```powershell
git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat
.\vcpkg\vcpkg.exe install websocketpp boost-system
```
2. Configure and build with CMake (VS generator):
```powershell
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>\scripts\buildsystems\vcpkg.cmake -A x64
cmake --build . --config Release
```
3. Run (from `build\Release\`):
```powershell
.\asr_ws_client.exe --uri ws://127.0.0.1:8000/asr/stream
```

## Run
- `./asr_ws_client --uri ws://127.0.0.1:8000/asr/stream`
- `./asr_ws_client --uri ws://127.0.0.1:8000/asr/stream --file audio.pcm`

## Notes
- The client sends a JSON start frame, then binary audio frames, then a JSON end frame.
- `--chunk-bytes` defaults to 3200 (16kHz * 16-bit * mono * 0.1s).
- `--bps` defaults to 32000 bytes/sec (16kHz * 16-bit * mono).
- For `wss://` TLS, you must switch to `asio_tls_client` and link OpenSSL.

