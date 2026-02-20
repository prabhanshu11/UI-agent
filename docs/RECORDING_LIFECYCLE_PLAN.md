# Recording Lifecycle: Sync, Purge & Dashboard

> **Status:** Plan (not implemented)
> **Created:** 2026-02-20
> **Context:** cursor-daemon writes 720p@30fps H.264 MKV chunks indefinitely with zero cleanup.
> At current rates (~3-12 GB/day depending on screen activity), the 247 GB free disk fills in ~20 days.

---

## Problem

Two data sources grow without bound on the desktop:

| Source | Location | Mechanism | Rate | Cleanup |
|--------|----------|-----------|------|---------|
| KVM recordings | `data/recordings/kvm_*.mkv` | cursor-daemon (C), 5-min H.264 chunks | 3-20 GB/day | **None** |
| Loss events | `data/loss_events/*/` | kvm_dashboard (Python), frames+patches on cursor loss | 0-9 GB/day | **None** |

Nothing syncs to the datalake. Nothing gets archived. Nothing gets purged.

## Existing Pieces (reusable)

| Component | Where | What it does |
|-----------|-------|--------------|
| `enforceBufferLimits()` | `tapo-c210-monitor/ringbuffer/main.go` | Time+size-based segment deletion (Go) |
| `DatalakeConnector` | `UI-agent/src/agents/datalake_connector.py` | Stages screenshots to `~/Programs/datalake/data/screenshots/` with metadata sidecars |
| `LogManager.prune_logs()` | `UI-agent/src/core/log_manager.py` | Deletes oldest files when count/size exceeded |
| Datalake schema v2 | `datalake/schema_v2.sql` | Has `audio`, `screenshots`, `sync_log` — no `video_segments` table |
| kvm_dashboard | `UI-agent/src/web/kvm_dashboard.py` | Already serves a web UI at localhost — can host new controls |

## Design

### 1. Datalake Schema Extension

Add a `video_segments` table to `datalake/schema_v2.sql`:

```sql
CREATE TABLE IF NOT EXISTS video_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,              -- Original path (data/recordings/kvm_*.mkv)
    filename TEXT NOT NULL,
    source TEXT NOT NULL,                 -- 'cursor-daemon', 'tapo-ringbuffer', 'orchestrator'
    source_device TEXT NOT NULL,          -- 'desktop', 'laptop'
    resolution TEXT,                      -- '1280x720', '1920x1080'
    fps INTEGER,
    codec TEXT,                           -- 'h264_nvenc', 'libx264'
    duration_seconds REAL,
    size_bytes INTEGER NOT NULL,
    segment_start_unix INTEGER NOT NULL,  -- Unix timestamp of segment start
    segment_end_unix INTEGER,             -- Unix timestamp of segment end
    content_hash TEXT,                    -- SHA256 for dedup
    sync_status TEXT DEFAULT 'local',     -- 'local', 'synced', 'archived', 'purged'
    archived_path TEXT,                   -- Path after archival (external drive, etc.)
    tags TEXT,                            -- JSON array
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    purged_at TEXT,                       -- When local copy was deleted
    metadata TEXT                         -- JSON (encoder settings, screen activity level, etc.)
);

CREATE INDEX IF NOT EXISTS idx_video_segments_start ON video_segments(segment_start_unix DESC);
CREATE INDEX IF NOT EXISTS idx_video_segments_source ON video_segments(source, source_device);
CREATE INDEX IF NOT EXISTS idx_video_segments_sync ON video_segments(sync_status);
CREATE INDEX IF NOT EXISTS idx_video_segments_size ON video_segments(size_bytes);
```

Similarly for loss events:

```sql
CREATE TABLE IF NOT EXISTS loss_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_dir TEXT NOT NULL,              -- e.g., '20260220_000003'
    source_device TEXT NOT NULL,
    frame_count INTEGER,
    patch_count INTEGER,
    size_bytes INTEGER NOT NULL,
    event_start_unix INTEGER NOT NULL,
    has_manifest INTEGER DEFAULT 0,
    sync_status TEXT DEFAULT 'local',
    tags TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    purged_at TEXT,
    metadata TEXT                         -- JSON (manifest contents, analysis results)
);
```

### 2. Ingestion Script

`UI-agent/src/storage/segment_indexer.py` — runs periodically (or on-demand from dashboard):

1. Scan `data/recordings/` for `kvm_*.mkv` files not yet in datalake
2. Extract metadata: size, duration (ffprobe), start timestamp from filename
3. Insert into `video_segments` table with `sync_status='local'`
4. Scan `data/loss_events/*/` similarly for loss events
5. Idempotent — uses filename as dedup key

### 3. Retention Policy Engine

`UI-agent/src/storage/retention.py`:

```
Retention rules (configurable via dashboard):
┌─────────────────────────────────────────────────────┐
│ KVM Recordings                                      │
│  - Max local age: 7 days (default)                  │
│  - Max local size: 50 GB (default)                  │
│  - Purge: delete local file, set sync_status='purged'│
│  - Pre-purge: must be 'synced' or 'archived' first  │
│    OR user explicitly enables "purge without backup" │
│                                                     │
│ Loss Events                                         │
│  - Max local age: 3 days (default)                  │
│  - Max local size: 10 GB (default)                  │
│  - Same purge rules                                 │
│                                                     │
│ Global                                              │
│  - Disk usage warning: at 80% full                  │
│  - Emergency purge: at 90% full, oldest first       │
│    regardless of sync status (with notification)    │
└─────────────────────────────────────────────────────┘
```

Two modes:
- **Safe purge** (default): only delete files that are `synced` or `archived` in datalake
- **Aggressive purge**: delete by age/size regardless — for when disk is critical

### 4. Archive/Backup Targets

Pluggable backends:

| Target | Method | When |
|--------|--------|------|
| Datalake metadata | SQLite insert (metadata only, not video bytes) | Always — every segment gets indexed |
| External drive | `rsync` to Samsung T7 / WD Elements when mounted | Manual trigger or on-connect |
| Remote (laptop) | `rsync` over Tailscale | Manual trigger from dashboard |
| Future: NAS/S3 | Configurable | When infra exists |

Video files are large — the datalake stores **metadata** (path, size, timestamps, tags), not the actual bytes. Archival copies go to external storage.

### 5. Dashboard Integration

Add a **Storage** tab to the existing kvm_dashboard (`localhost:PORT`):

```
┌─ Storage ──────────────────────────────────────────────────────┐
│                                                                │
│  Disk: ████████████░░░░░░░░░░ 198/446 GB (45%)               │
│  Est. fill: ~20 days at current rate                          │
│                                                                │
│  ┌─ Recordings ────────────────────────────────────────────┐  │
│  │ Total: 8.4 GB (459 files)  Rate: ~132 MB/hr            │  │
│  │ Oldest: Feb 18 19:02   Newest: Feb 20 09:55             │  │
│  │ Indexed: 420/459   Synced: 0/459   Archived: 0/459     │  │
│  │                                                          │  │
│  │ [Purge older than ▼ 7 days]  [Archive to ▼ T7 SSD]     │  │
│  │ [Index Now]  [Emergency Purge]                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌─ Loss Events ───────────────────────────────────────────┐  │
│  │ Total: 1.6 GB (103 events)  Rate: ~9 GB/day            │  │
│  │ Oldest: Feb 19 23:56   Newest: Feb 20 04:11             │  │
│  │                                                          │  │
│  │ [Purge older than ▼ 3 days]  [Purge All]                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌─ Retention Policy ──────────────────────────────────────┐  │
│  │ Recordings max age: [7 days ▼]  max size: [50 GB ▼]    │  │
│  │ Loss events max age: [3 days ▼]  max size: [10 GB ▼]   │  │
│  │ Auto-purge: [●] enabled    Safe mode: [●] enabled      │  │
│  │ (safe = only purge if synced/archived)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 6. API Endpoints

Add to kvm_dashboard Flask routes:

```
GET  /storage/status        — disk usage, recording counts, rates, fill estimate
GET  /storage/segments      — list segments with sync_status, filterable
POST /storage/index         — trigger indexing scan
POST /storage/purge         — purge by criteria (age, size, status)
POST /storage/archive       — trigger archive to specified target
GET  /storage/retention     — current retention policy
POST /storage/retention     — update retention policy
```

## Implementation Order

1. **Schema** — Add `video_segments` + `loss_events` tables to datalake
2. **Indexer** — Scan existing files, populate metadata in datalake
3. **Retention engine** — Age/size-based purge with safe-mode guard
4. **API endpoints** — Wire into kvm_dashboard
5. **Dashboard tab** — Storage UI with controls
6. **Auto-purge** — Background thread in cursor-daemon or dashboard, runs retention on interval
7. **Archive** — External drive rsync integration

## Immediate Stopgap (before full implementation)

If disk pressure becomes urgent before this is built, a simple cron job can prevent disaster:

```bash
# Delete recordings older than 7 days
find ~/Programs/UI-agent/data/recordings/ -name "kvm_*.mkv" -mtime +7 -delete

# Delete loss events older than 3 days
find ~/Programs/UI-agent/data/loss_events/ -maxdepth 1 -type d -mtime +3 -exec rm -rf {} +
```

This loses data permanently (no sync), but prevents disk-full.

## Open Questions

- Should the cursor-daemon itself enforce a ring buffer (like the tapo Go service), or should cleanup stay external?
  - Pro daemon-internal: guaranteed, no race conditions, works even if dashboard is down
  - Pro external: more flexible, dashboard can show "about to purge" warnings, user can intervene
- How much history is actually useful? Are week-old cursor recordings ever reviewed?
- Should loss events be summarized (keep manifest.json, delete frames/patches) instead of fully purged?
