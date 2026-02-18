# C Cursor Daemon — Real-Time Cursor Detection via libav

## Motivation

The Python cursor detection pipeline has a fundamental limitation: it uses
ffmpeg as a subprocess to record video to a file, then spawns another ffmpeg
process to decode that file into raw frames for analysis. This two-stage
approach works well for **offline** analysis (record → stop → analyze), but
cannot do true real-time detection because:

1. Reading a growing MKV file is unreliable (no container trailer yet)
2. Python's `subprocess.Popen` + pipe overhead adds latency
3. numpy frame processing in Python is CPU-bound on the GIL

A C daemon using libav directly can read frames from the V4L2 capture device
in real-time, perform frame differencing in-place, and expose cursor position
via shared memory or a Unix socket.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              cursor-daemon (C)                   │
│                                                  │
│  V4L2 capture (/dev/video0)                      │
│       │                                          │
│       ▼                                          │
│  libavformat  ──→  raw YUV frame                 │
│       │                                          │
│       ▼                                          │
│  Frame differencing (prev vs current)            │
│       │                                          │
│       ▼                                          │
│  Blob extraction (connected components)          │
│       │                                          │
│       ▼                                          │
│  Curvature matching (Lissajous signature)        │
│       │                                          │
│       ▼                                          │
│  Shared memory: cursor_x, cursor_y, timestamp    │
│  Unix socket: /tmp/cursor-daemon.sock            │
└─────────────────────────────────────────────────┘
         │
         ▼
  Python CursorLocator reads position from socket
```

## Key Design Decisions

### Frame capture
- Use V4L2 directly (or libavdevice v4l2) for zero-copy frame access
- MJPEG input format → decode to YUV420P (hardware-accelerated on most cards)
- Target: 320x240 @ 60fps for motion detection (downscaled from 1920x1080)
- Full-res capture only when cursor position is confirmed and navigation needs it

### Frame differencing
- Work in YUV space (only need Y channel for motion)
- Simple absolute difference: `|Y_curr[i] - Y_prev[i]| > threshold`
- Connected component labeling for blob extraction (two-pass algorithm)
- No numpy, no Python — pure C with SIMD intrinsics if beneficial

### Curvature matching
- Port the Lissajous angle-matching algorithm from `cursor_detector.py`
- Sliding window of blob centroids → compute angles → correlate with expected curve
- This is the same math, just in C: atan2, dot product, correlation coefficient

### IPC
- Primary: shared memory (`/dev/shm/cursor-daemon`) — zero-copy, minimal latency
  - struct: `{ int32_t x, y; float correlation; uint64_t timestamp_ns; uint8_t valid; }`
- Secondary: Unix socket for commands (start sweep, stop, query status)

### Integration with Python
- `CursorLocator.locate()` sends "sweep" command to daemon via socket
- Daemon triggers ESP32 mouse movement (serial write) and starts analyzing
- Python polls shared memory for result
- Fallback: if daemon not running, use the current offline Python approach

## Dependencies

- `libavformat`, `libavcodec`, `libavutil` (from ffmpeg)
- `libv4l2` (optional, for direct V4L2 access)
- Standard POSIX: `shm_open`, `mmap`, Unix sockets, `pthread`

## Build

```bash
cc -O2 -o cursor-daemon cursor-daemon.c \
   $(pkg-config --cflags --libs libavformat libavcodec libavutil) \
   -lpthread -lrt
```

## Status

**Future work** — the current Python offline approach works (0.92 correlation
proven). This C daemon would enable:
1. True real-time detection with early termination
2. Always-on cursor tracking (the "lightweight daemon" concept)
3. Sub-millisecond frame processing instead of Python's ~160ms/frame

Priority: implement after the K380 pairing experience is working end-to-end.
