"""Client for the C cursor daemon's shared memory interface.

Reads cursor position, motion blobs, and JPEG frames from
/dev/shm/cursor-daemon with lock-free sequence counter validation.

Usage:
    from src.hardware.daemon_client import DaemonClient

    client = DaemonClient()
    if client.is_running():
        state = client.read_state()
        jpeg = client.read_jpeg()
"""

import mmap
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

from .cursor_detector import MotionBlob


# ── Constants matching daemon_types.h ────────────────────────────

SHM_NAME = "/cursor-daemon"
SHM_MAGIC = 0xCDA0CDA0
SHM_VERSION = 1
MAX_BLOBS = 32
JPEG_BUF_SIZE = 512 * 1024

# Byte offsets computed from daemon_types.h packed struct layout
OFF_MAGIC         = 0
OFF_VERSION       = 4
OFF_SEQ_BEGIN     = 8
OFF_CURSOR_X      = 16
OFF_CURSOR_Y      = 20
OFF_CURSOR_AGE    = 24
OFF_CURSOR_METHOD = 28
OFF_BLOB_COUNT    = 32
OFF_BLOBS         = 36     # 32 * 24 = 768 bytes
OFF_FPS           = 804
OFF_FRAME_COUNT   = 808
OFF_TIMESTAMP_NS  = 816
OFF_DAEMON_RUN    = 824
OFF_RECORDING     = 825
OFF_SEQ_END       = 832
OFF_JPEG0_SIZE    = 848
OFF_JPEG0_DATA    = 852
OFF_JPEG1_SIZE    = 852 + JPEG_BUF_SIZE          # 525140
OFF_JPEG1_DATA    = 852 + JPEG_BUF_SIZE + 4      # 525144
OFF_JPEG_ACTIVE   = 852 + 2 * JPEG_BUF_SIZE + 8  # 1049436

# Total shm size (must match C sizeof(CursorDaemonShm))
# Approximate: OFF_JPEG_ACTIVE + 1 rounded up
SHM_SIZE = OFF_JPEG_ACTIVE + 8  # Small overestimate is fine for mmap

BLOB_SIZE = 24  # sizeof(ShmBlob)

CURSOR_METHODS = {
    0: "unknown",
    1: "motion_track",
    2: "shape_track",
    3: "micro_shake",
    4: "macro_shake",
    5: "lissajous",
    6: "api_probe",
    7: "cnn_reacquire",
}


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class DaemonState:
    """Snapshot of daemon shared memory state."""
    cursor_x: int = 0
    cursor_y: int = 0
    cursor_age_s: float = 999.0
    cursor_method: str = "unknown"
    blobs: list[MotionBlob] = field(default_factory=list)
    blob_count: int = 0
    fps: float = 0.0
    frame_count: int = 0
    timestamp_ns: int = 0
    daemon_running: bool = False
    recording_active: bool = False


# ── Client ───────────────────────────────────────────────────────

class DaemonClient:
    """Reads cursor-daemon state from POSIX shared memory.

    Lock-free: uses sequence counter for torn-read detection.
    Double-buffered JPEG: reads from the active buffer atomically.
    """

    def __init__(self):
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None
        self._attach()

    def _attach(self):
        """Attach to shared memory. Silently fails if daemon not running."""
        shm_path = f"/dev/shm{SHM_NAME}"
        try:
            self._fd = os.open(shm_path, os.O_RDONLY)
            file_size = os.fstat(self._fd).st_size
            if file_size < OFF_JPEG_ACTIVE:
                # File too small — daemon may be initializing
                os.close(self._fd)
                self._fd = None
                return
            self._mm = mmap.mmap(self._fd, file_size, access=mmap.ACCESS_READ)
        except (FileNotFoundError, OSError):
            self._fd = None
            self._mm = None

    def is_running(self) -> bool:
        """Check if daemon is running (shm exists + magic valid + flag set)."""
        if self._mm is None:
            self._attach()
        if self._mm is None:
            return False
        try:
            magic = struct.unpack_from("<I", self._mm, OFF_MAGIC)[0]
            if magic != SHM_MAGIC:
                return False
            return self._mm[OFF_DAEMON_RUN] != 0
        except (struct.error, IndexError):
            return False

    def read_state(self, max_retries: int = 3) -> Optional[DaemonState]:
        """Read a consistent snapshot of daemon state.

        Uses sequence counter for torn-read detection:
        read seq_begin, copy fields, read seq_end. Retry if mismatched.
        """
        if self._mm is None:
            return None

        for _ in range(max_retries):
            try:
                seq_begin = struct.unpack_from("<Q", self._mm, OFF_SEQ_BEGIN)[0]

                cursor_x = struct.unpack_from("<i", self._mm, OFF_CURSOR_X)[0]
                cursor_y = struct.unpack_from("<i", self._mm, OFF_CURSOR_Y)[0]
                cursor_age = struct.unpack_from("<f", self._mm, OFF_CURSOR_AGE)[0]
                cursor_method_id = self._mm[OFF_CURSOR_METHOD]
                blob_count = struct.unpack_from("<i", self._mm, OFF_BLOB_COUNT)[0]
                fps = struct.unpack_from("<f", self._mm, OFF_FPS)[0]
                frame_count = struct.unpack_from("<Q", self._mm, OFF_FRAME_COUNT)[0]
                timestamp_ns = struct.unpack_from("<Q", self._mm, OFF_TIMESTAMP_NS)[0]
                daemon_running = self._mm[OFF_DAEMON_RUN] != 0
                recording = self._mm[OFF_RECORDING] != 0

                # Read blobs
                blobs = []
                n = min(blob_count, MAX_BLOBS)
                for i in range(n):
                    off = OFF_BLOBS + i * BLOB_SIZE
                    cx, cy = struct.unpack_from("<hh", self._mm, off)
                    angle = struct.unpack_from("<f", self._mm, off + 4)[0]
                    px_count = struct.unpack_from("<i", self._mm, off + 8)[0]
                    bx0, by0, bx1, by1 = struct.unpack_from("<hhhh", self._mm, off + 12)
                    blobs.append(MotionBlob(
                        centroid=(int(cx), int(cy)),
                        angle=float(angle),
                        pixel_count=int(px_count),
                        bbox=(int(bx0), int(by0), int(bx1), int(by1)),
                    ))

                seq_end = struct.unpack_from("<Q", self._mm, OFF_SEQ_END)[0]

                if seq_begin == seq_end:
                    return DaemonState(
                        cursor_x=cursor_x,
                        cursor_y=cursor_y,
                        cursor_age_s=cursor_age,
                        cursor_method=CURSOR_METHODS.get(cursor_method_id, "unknown"),
                        blobs=blobs,
                        blob_count=len(blobs),
                        fps=fps,
                        frame_count=frame_count,
                        timestamp_ns=timestamp_ns,
                        daemon_running=daemon_running,
                        recording_active=recording,
                    )
            except (struct.error, IndexError):
                return None

        return None  # All retries failed

    def read_jpeg(self) -> Optional[bytes]:
        """Read the latest JPEG frame from the double buffer.

        Reads from the active buffer — no tearing since daemon writes
        to the inactive buffer and atomically flips.
        """
        if self._mm is None:
            return None

        try:
            active = self._mm[OFF_JPEG_ACTIVE]
            if active == 0:
                size = struct.unpack_from("<I", self._mm, OFF_JPEG0_SIZE)[0]
                if size == 0 or size > JPEG_BUF_SIZE:
                    return None
                return bytes(self._mm[OFF_JPEG0_DATA:OFF_JPEG0_DATA + size])
            else:
                size = struct.unpack_from("<I", self._mm, OFF_JPEG1_SIZE)[0]
                if size == 0 or size > JPEG_BUF_SIZE:
                    return None
                return bytes(self._mm[OFF_JPEG1_DATA:OFF_JPEG1_DATA + size])
        except (struct.error, IndexError):
            return None

    def read_frame_raw(self) -> Optional[bytes]:
        """Read latest JPEG and return as raw bytes (for MCP screenshot tool)."""
        return self.read_jpeg()

    def close(self):
        """Release shared memory mapping."""
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
