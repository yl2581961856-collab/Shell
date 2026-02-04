# Streaming WS Checklist

This checklist is for implementing and validating a streaming ASR WebSocket path.

## 1) Protocol & Data Contract
- Define a JSON start frame: `type=start`, `sr`, `fmt`, `ch`, optional `session_id`.
- Define binary audio frames: raw PCM bytes.
- Define a JSON end frame: `type=end`.
- Define server responses: `partial`, `final`, `error`, optional `seq`, `ts`.
- Enumerate error codes (e.g., `INVALID_FORMAT`, `UNSUPPORTED_SR`, `FRAME_TOO_LARGE`).

## 2) Server WS Flow (Python)
- Accept connection; wait for `start`.
- Validate fields (sr/fmt/ch) and reject invalid clients.
- On binary frames: process or buffer; respond `partial`.
- On `end`: finalize and respond `final`.
- Implement idle timeout to close stale sessions.

## 3) Client WS Flow (C++)
- Connect and send `start` JSON.
- Stream binary frames in fixed chunks.
- Receive `partial/final` responses; print or forward.
- Send `end`; close after final.

## 4) Error Handling
- JSON parse failure: respond `error` and close.
- Missing/invalid `start` fields: respond `error` and close.
- Oversized binary frame: respond `error` and drop or close.
- Server exception: log stack and close session.
- Client disconnect: server cleans up session state.

## 5) IO Concurrency & Flow Control
- Server: avoid heavy compute in WS callbacks.
- Server: per-connection session state (buffer, counters, timestamps).
- Client: queue sends; cap queue size.
- Client: throttle by bytes-per-second.
- Server: enforce max message size.

## 6) Heartbeat & Reconnect
- Client: ping every 20-30s.
- Server: handle pong; close on timeout.
- Client: reconnect with exponential backoff.

## 7) Observability
- Connection open/close logs.
- Session total bytes and duration.
- Error code + reason.
- Optional RTT from send to partial.

## 8) Local Validation Steps
- Mock mode: server replies `partial` with total bytes.
- Single stream test.
- Multi-stream (5+) test.
- Fault tests: invalid JSON, oversized frames, disconnect mid-stream.
