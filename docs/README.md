# API & Frontend Examples

This document is intended for frontend integration and QA validation. The browser target is QiAnXin (Chrome kernel), so standard Web APIs below are supported.
本说明面向前端联调与测试，目标浏览器为奇安信（Chrome 内核），可直接使用标准 Web API。

## Swagger
FastAPI exposes interactive docs at:
- `/docs`
- `/redoc`
FastAPI 自带交互式接口文档，可在上述路径查看与调试。

## Interface Overview
- HTTP `GET /health`: service health check.
- HTTP `POST /asr/file`: upload a full audio file and get a one-shot transcription result.
- WebSocket `/asr/stream`: streaming ASR with per-chunk `partial` and final result.
接口总览：健康检查、文件转写、流式转写三类。

Code references in `src/server.py`:
- `health`
- `asr_file`
- `asr_stream`

## HTTP: GET /health
Request: `GET /health`
用于探活，无参数。

Response JSON:
- `ok`: boolean

Example:
```json
{ "ok": true }
```

## HTTP: POST /asr/file
Request:
- `Content-Type: multipart/form-data`
- Field: `file` (single audio file)
文件上传转写，字段名必须是 `file`。

Response JSON:
- `text`: string
- `stub`: boolean
- `timestamps`: optional, word/segment timestamps if enabled by model

Example:
```json
{ "text": "你好世界", "stub": true, "timestamps": [] }
```

## WebSocket: /asr/stream
Session constraints:
- `sr` must be `16000`
- `ch` must be `1`
- `fmt` must be `pcm_s16le | pcm16le | s16le` (normalized)
- `encoding` optional, default `pcm`, supports `pcm | raw`
- Max binary frame: `1MB`
- Idle timeout: `30s`
流式只接受 16k/16bit/mono PCM，客户端需先处理好采样率。

Client message types:
- Text JSON `type=start` or `type=config`
  - Fields: `sr`, `ch`, `fmt`, `encoding` (optional), `session_id` (optional)
- Binary audio frames (raw PCM bytes)
- Text JSON `type=end`
必须先发 `start`，否则服务端会返回 `MISSING_START`。

Server message types:
- `ack`: confirm session params
- `partial`: per-chunk partial transcription (may include `timestamps`)
- `final`: final transcription, then server closes (may include `timestamps`)
- `error`: error frame
`partial/final` 可能带 `timestamps`，需做好前端兼容。

Close codes:
- `1000`: normal close (idle timeout or end)
- `1003`: unsupported format/invalid JSON
- `1008`: policy violation (missing start)
- `1009`: frame too large

## Browser Request Examples (JavaScript)

### 1) Health Check
```js
async function healthCheck() {
  const res = await fetch('http://127.0.0.1:8000/health');
  if (!res.ok) throw new Error(`health failed: ${res.status}`);
  return res.json();
}
```

### 2) Upload File
```js
async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch('http://127.0.0.1:8000/asr/file', {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(`upload failed: ${res.status}`);
  return res.json();
}
```

### 3) WebSocket (Start/Partial/Final)
```js
function createAsrWs({ url, sessionId }) {
  const ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    const start = {
      type: 'start',
      sr: 16000,
      ch: 1,
      fmt: 'pcm_s16le',
      encoding: 'pcm',
      session_id: sessionId || undefined,
    };
    ws.send(JSON.stringify(start));
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return;
    const msg = JSON.parse(ev.data);
    if (msg.type === 'ack') {
      console.log('ack', msg);
    } else if (msg.type === 'partial') {
      console.log('partial', msg.text, msg.seq, msg.bytes, msg.timestamps);
    } else if (msg.type === 'final') {
      console.log('final', msg.text, msg.timestamps);
    } else if (msg.type === 'error') {
      console.error('error', msg.code, msg.message);
    }
  };

  ws.onerror = (e) => console.error('ws error', e);
  ws.onclose = () => console.log('ws closed');
  return ws;
}

function endWs(ws, sessionId) {
  if (ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: 'end', session_id: sessionId }));
}
```

### 4) OfflineAudioContext 重采样到 16k
```js
async function decodeAudioFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
  ctx.close();
  return audioBuffer;
}

async function resampleTo16kMono(audioBuffer) {
  if (audioBuffer.sampleRate === 16000 && audioBuffer.numberOfChannels === 1) {
    return audioBuffer;
  }
  const targetRate = 16000;
  const frameCount = Math.ceil(audioBuffer.duration * targetRate);
  const offline = new OfflineAudioContext(1, frameCount, targetRate);
  const source = offline.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offline.destination);
  source.start(0);
  return offline.startRendering();
}

function floatTo16BitPCM(float32Array) {
  const out = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}
```

### 5) 文件流式发送（PCM 16k, 16-bit, mono）
```js
function sendPcm16InChunks(ws, pcm16, opts = {}) {
  const sampleRate = opts.sampleRate || 16000;
  const channels = opts.channels || 1;
  const chunkMs = opts.chunkMs || 100;
  const realtime = opts.realtime !== false;

  const bytes = new Uint8Array(pcm16.buffer);
  const bytesPerSample = 2;
  const chunkBytes = Math.floor(sampleRate * channels * bytesPerSample * chunkMs / 1000);

  return new Promise((resolve) => {
    let offset = 0;
    const sendNext = () => {
      if (ws.readyState !== WebSocket.OPEN) return resolve();
      if (offset >= bytes.length) return resolve();
      const end = Math.min(offset + chunkBytes, bytes.length);
      ws.send(bytes.slice(offset, end));
      offset = end;
      if (realtime) {
        setTimeout(sendNext, chunkMs);
      } else {
        queueMicrotask(sendNext);
      }
    };
    sendNext();
  });
}

async function streamFileToWs(file) {
  const sessionId = `web-${Date.now()}`;
  const ws = createAsrWs({ url: 'ws://127.0.0.1:8000/asr/stream', sessionId });

  ws.onopen = async () => {
    const audioBuffer = await decodeAudioFile(file);
    const resampled = await resampleTo16kMono(audioBuffer);
    const pcm16 = floatTo16BitPCM(resampled.getChannelData(0));
    await sendPcm16InChunks(ws, pcm16, { chunkMs: 100, realtime: true });
    endWs(ws, sessionId);
  };
}
```

### 6) 麦克风实时流
```js
function startMicStreaming() {
  const sessionId = `web-${Date.now()}`;
  const ws = createAsrWs({ url: 'ws://127.0.0.1:8000/asr/stream', sessionId });

  navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } })
    .then((stream) => {
      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(1024, 1, 1);

      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const pcm16 = floatTo16BitPCM(input);
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(pcm16.buffer);
        }
      };

      source.connect(processor);
      processor.connect(ctx.destination);

      // Return a stop function
      window.stopMicStreaming = () => {
        endWs(ws, sessionId);
        processor.disconnect();
        source.disconnect();
        stream.getTracks().forEach(t => t.stop());
        ctx.close();
      };
    });
}
```

Notes:
- `MediaRecorder` outputs compressed data and cannot be sent directly.
- If the browser does not output 16k PCM, you need a real-time resampler (AudioWorklet). The file example above uses OfflineAudioContext for correct 16k conversion.
