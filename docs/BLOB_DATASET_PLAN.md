# Blob Dataset: Per-Frame JSONL Logging in cursor-daemon

> **Status:** Plan (not implemented)
> **Created:** 2026-02-20
> **Context:** cursor-daemon extracts motion blobs at 30fps via connected-component labeling
> on frame-differenced Y-channel data. This data is currently written ONLY to shared memory
> (`/dev/shm/cursor-daemon`) and overwritten every frame (~33ms). No blob data persists to disk.
> The video persists (MKV chunks in `data/recordings/`), but the structured detection layer on
> top of it is lost.

---

## Problem

The daemon produces rich per-frame blob data (up to 32 blobs × 6 fields each) at 30fps.
This is a high-resolution time-series of all visual motion on the captured screen. Currently:

| Data | Persisted? | Location |
|------|-----------|----------|
| Raw video (720p H.264) | Yes | `data/recordings/kvm_*.mkv` |
| Per-frame blobs (centroids, bbox, angle, pixel count) | **No** — shared memory only | `/dev/shm/cursor-daemon` |
| Cursor position (Python-derived) | Partially | `data/state_log.jsonl` (blob **count** only, no details) |
| Vision inference results | Yes (sparse) | `data/vision_dataset/log.jsonl` (7 entries total) |

The blob data is the structured representation of what the video shows. Without it, replaying
cursor detection requires re-processing video through the full pipeline. With it, you can
query "what moved, where, when" directly.

## Design

### Output format: JSONL

One JSON object per frame, one file per recording chunk (aligned with MKV rotation).

**File location:** `data/blob_log/blobs_YYYYMMDD_HHMMSS.jsonl` (mirrors MKV naming)

**Schema per line:**

```json
{
  "ts": 1771587600123456789,
  "fn": 45021,
  "bc": 3,
  "b": [
    {"cx": 512, "cy": 340, "a": 0.785, "px": 87, "bx": [500, 330, 524, 350]},
    {"cx": 1400, "cy": 900, "a": -1.2, "px": 210, "bx": [1380, 880, 1420, 920]},
    {"cx": 100, "cy": 50, "a": 0.0, "px": 15, "bx": [95, 45, 105, 55]}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `ts` | uint64 | `clock_gettime(CLOCK_REALTIME)` in nanoseconds |
| `fn` | uint64 | Frame number (`g_shm->frame_count`) |
| `bc` | int | Blob count this frame |
| `b` | array | Blob array (omitted when `bc == 0` to save space) |
| `b[].cx` | int16 | Centroid X |
| `b[].cy` | int16 | Centroid Y |
| `b[].a` | float | Principal axis angle (radians), rounded to 3 decimals |
| `b[].px` | int32 | Pixel count |
| `b[].bx` | [4]int16 | Bounding box `[x_min, y_min, x_max, y_max]` |

**Short keys** to minimize file size at 30fps write rate.

### Estimated data rate

- **Idle screen** (0 blobs most frames): ~40 bytes/line × 30fps = ~1.2 KB/s = **~100 MB/day**
- **Active use** (2-5 blobs): ~150 bytes/line × 30fps = ~4.5 KB/s = **~380 MB/day**
- **Heavy motion** (10+ blobs): ~400 bytes/line × 30fps = ~12 KB/s = **~1 GB/day**

Compresses extremely well (gzip ~10:1 on repetitive JSON). Compared to video at 3-20 GB/day,
this is a modest addition.

## Implementation (C daemon changes)

### New CLI flag

```
--blob-log-dir PATH    Directory for blob JSONL files (default: data/blob_log/)
--no-blob-log          Disable blob logging
```

### New config fields

```c
// In Config struct (cursor_daemon.c line ~50)
const char *blob_log_dir;
int         no_blob_log;
```

### Blob logging thread

**Do NOT write from the capture loop** — file I/O in the hot path would add jitter.
Use the same producer/consumer pattern as the recording thread:

```
Capture loop (hot)           Blob log thread (background)
─────────────────           ────────────────────────────
extract_blobs()
  │
  ▼
enqueue_blob_snapshot()  →  ring buffer  →  dequeue + fprintf()
  (lock-free, ~20ns)                        (batched, buffered I/O)
```

**Ring buffer:** Fixed array of `BlobSnapshot` structs (64 slots). Atomic head/tail.
Each snapshot is ~800 bytes (frame metadata + 32 blobs). Total ring: ~50 KB.

```c
typedef struct {
    uint64_t timestamp_ns;   // CLOCK_REALTIME
    uint64_t frame_number;
    int      blob_count;
    ShmBlob  blobs[MAX_BLOBS];
} BlobSnapshot;

#define BLOB_RING_SIZE 64
static BlobSnapshot blob_ring[BLOB_RING_SIZE];
static _Atomic uint32_t blob_ring_head = 0;  // writer
static _Atomic uint32_t blob_ring_tail = 0;  // reader
```

**Writer (capture loop, after extract_blobs):**
```c
// ~20ns: copy blob data into ring slot, advance head
uint32_t h = atomic_load_explicit(&blob_ring_head, memory_order_relaxed);
uint32_t next = (h + 1) % BLOB_RING_SIZE;
if (next != atomic_load_explicit(&blob_ring_tail, memory_order_acquire)) {
    blob_ring[h].timestamp_ns = realtime_ns();
    blob_ring[h].frame_number = g_shm->frame_count;
    blob_ring[h].blob_count   = blob_count;
    memcpy(blob_ring[h].blobs, local_blobs, blob_count * sizeof(ShmBlob));
    atomic_store_explicit(&blob_ring_head, next, memory_order_release);
}
// If ring full: drop frame silently (logging can't keep up — non-critical)
```

**Reader (blob log thread):**
```c
static void *blob_log_thread(void *arg) {
    FILE *fp = NULL;
    time_t chunk_start = 0;
    char buf[8192];  // setvbuf for batched writes

    while (g_shm->daemon_running) {
        // Rotate file every CHUNK_DURATION_S (same cadence as MKV rotation)
        // ...open new file if needed...

        uint32_t t = atomic_load_explicit(&blob_ring_tail, memory_order_relaxed);
        uint32_t h = atomic_load_explicit(&blob_ring_head, memory_order_acquire);

        if (t == h) {
            usleep(10000);  // 10ms — drain at ~100Hz, well ahead of 30fps input
            continue;
        }

        // Drain all available snapshots
        while (t != h) {
            BlobSnapshot *snap = &blob_ring[t];
            // Format JSON line and fwrite to fp
            // ...
            t = (t + 1) % BLOB_RING_SIZE;
        }
        atomic_store_explicit(&blob_ring_tail, t, memory_order_release);
        // Don't fflush every iteration — rely on setvbuf buffering
    }
    if (fp) fclose(fp);
    return NULL;
}
```

### File rotation

Rotate blob log files on the same schedule as MKV chunks (`CHUNK_DURATION_S = 300` = 5 min).
This keeps blob files 1:1 with video chunks for easy correlation:

```
data/recordings/kvm_20260220_152041.mkv     ←→  data/blob_log/blobs_20260220_152041.jsonl
data/recordings/kvm_20260220_152541.mkv     ←→  data/blob_log/blobs_20260220_152541.jsonl
```

### JSON formatting (in C)

Use `snprintf` directly — no JSON library needed. The schema is fixed:

```c
// For each snapshot:
int off = 0;
off += snprintf(line+off, sizeof(line)-off,
    "{\"ts\":%lu,\"fn\":%lu,\"bc\":%d",
    snap->timestamp_ns, snap->frame_number, snap->blob_count);
if (snap->blob_count > 0) {
    off += snprintf(line+off, sizeof(line)-off, ",\"b\":[");
    for (int i = 0; i < snap->blob_count; i++) {
        ShmBlob *b = &snap->blobs[i];
        if (i > 0) line[off++] = ',';
        off += snprintf(line+off, sizeof(line)-off,
            "{\"cx\":%d,\"cy\":%d,\"a\":%.3f,\"px\":%d,\"bx\":[%d,%d,%d,%d]}",
            b->centroid_x, b->centroid_y, b->angle, b->pixel_count,
            b->bbox_x_min, b->bbox_y_min, b->bbox_x_max, b->bbox_y_max);
    }
    off += snprintf(line+off, sizeof(line)-off, "]");
}
off += snprintf(line+off, sizeof(line)-off, "}\n");
fwrite(line, 1, off, fp);
```

### Integration points in cursor_daemon.c

| Location | Change |
|----------|--------|
| `Config` struct (~line 50) | Add `blob_log_dir`, `no_blob_log` |
| `parse_args()` (~line 694) | Add `--blob-log-dir`, `--no-blob-log` options |
| `print_usage()` (~line 680) | Document new flags |
| After `extract_blobs()` (~line 921) | Call `enqueue_blob_snapshot()` |
| `main()` after recording thread start (~line 836) | Start blob log thread |
| Shutdown (~line 971) | Join blob log thread |

### Performance budget

| Operation | Cost | Where |
|-----------|------|-------|
| `enqueue_blob_snapshot()` | ~20ns (memcpy + atomic store) | Capture loop (hot path) |
| Ring buffer memory | ~50 KB fixed | Static allocation |
| `blob_log_thread` drain | ~100μs per batch | Background thread |
| Disk I/O | Buffered, ~4 KB/s typical | Background thread |

**Zero impact on capture loop latency** — the ring buffer enqueue is a memcpy + atomic,
same pattern as the existing recording frame enqueue.

## Retention

Blob log files should follow the same lifecycle as MKV recordings (see `RECORDING_LIFECYCLE_PLAN.md`):

- Same age/size retention policy
- Same purge rules (safe purge only after sync/archive)
- Add to `RECORDING_LIFECYCLE_PLAN.md` schema: include `blob_log` as a data source
- Index in datalake alongside video segments

## Relation to existing data

```
┌─────────────────────────────────────────────────────────────┐
│ cursor-daemon capture loop (30fps)                          │
│                                                             │
│  Frame ──→ MKV recording     (data/recordings/)  ← exists  │
│    │                                                        │
│    ▼                                                        │
│  extract_blobs()                                            │
│    │                                                        │
│    ├──→ Shared memory        (/dev/shm/cursor-daemon) ← exists (ephemeral) │
│    │                                                        │
│    └──→ Blob JSONL log       (data/blob_log/)     ← NEW    │
│                                                             │
│  Python dashboard reads shm ──→ state_log.jsonl   ← exists (sparse, no blob detail) │
│  Python Vision calls ──→ vision_dataset/log.jsonl ← exists (very sparse) │
└─────────────────────────────────────────────────────────────┘
```

## Open questions

- **Compression:** Should the blob log thread gzip-compress on the fly (`.jsonl.gz`)? Saves ~90% disk but adds CPU. Could use a separate compression pass via cron instead.
- **Zero-blob frames:** Log them (for completeness — proves "nothing moved") or skip (saves ~70% of lines on idle screens)? Current plan: log them with just `{"ts":...,"fn":...,"bc":0}` — cheap and preserves the complete timeline.
- **Datalake integration:** Should blob logs be indexed in the same `video_segments` table or a new `blob_logs` table? Leaning toward a new table since the lifecycle is the same but the data type differs.
