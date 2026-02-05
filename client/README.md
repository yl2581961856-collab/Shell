# C++ WebSocket Client (websocketpp + Boost)

This is a minimal Linux/GCC and Windows/MSVC client for the ASR WebSocket endpoint.

## Dependencies
- `websocketpp` (header-only)
- `Boost.Asio` (Boost.System)
- Optional: `PortAudio` (microphone streaming, build with `-DASRWS_ENABLE_PORTAUDIO=ON`)
- CMake + compiler toolchain

## Build (Ubuntu)
1. Install deps:
   - `sudo apt-get update`
   - `sudo apt-get install -y g++ cmake libboost-system-dev`
   - (Optional mic) `sudo apt-get install -y portaudio19-dev`
2. Build:
   - `mkdir -p build`
   - `cd build`
   - `cmake ..` (add `-DASRWS_ENABLE_PORTAUDIO=ON` to enable mic)
   - `cmake --build . -j`

## Build (Windows / Visual Studio)
1. Install vcpkg and deps:
```powershell
git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat
.\vcpkg\vcpkg.exe install websocketpp boost-system
# (Optional mic)
.\vcpkg\vcpkg.exe install portaudio
```
2. Configure and build with CMake (VS generator):
```powershell
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>\scripts\buildsystems\vcpkg.cmake -A x64 -DASRWS_ENABLE_PORTAUDIO=ON
cmake --build . --config Release
```
3. Run (from `build\Release\`):
```powershell
.\asr_ws_client.exe --uri ws://127.0.0.1:8000/asr/stream
```

## Run
- `./asr_ws_client --uri ws://127.0.0.1:8000/asr/stream`
- `./asr_ws_client --uri ws://127.0.0.1:8000/asr/stream --file audio.pcm`
- (Mic) list devices: `./asr_ws_client --list-devices`
- (Mic) stream: `./asr_ws_client --mic --device 0 --chunk-ms 100`

## Notes
- The client sends a JSON start frame, then binary audio frames, then a JSON end frame.
- `--chunk-bytes` defaults to 3200 (16kHz * 16-bit * mono * 0.1s).
- `--bps` defaults to 32000 bytes/sec (16kHz * 16-bit * mono).
- In mic mode, stop with Ctrl+C or use `--duration N`.
- For `wss://` TLS, you must switch to `asio_tls_client` and link OpenSSL.

