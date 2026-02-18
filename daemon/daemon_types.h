/*
 * daemon_types.h — Shared memory layout for cursor-daemon <-> Python IPC.
 *
 * This header defines the exact byte layout of /dev/shm/cursor-daemon.
 * Both the C daemon and Python's daemon_client.py (via struct/ctypes)
 * must agree on these offsets. All structs are packed, no hidden padding.
 *
 * Lock-free read protocol:
 *   1. Read seq_begin
 *   2. Copy all fields
 *   3. Read seq_end
 *   4. If seq_begin == seq_end, snapshot is consistent. Else retry.
 *
 * Double-buffered JPEG protocol:
 *   - Daemon writes to inactive buffer, then atomically flips jpeg_active.
 *   - Reader always reads from the active buffer. No tearing possible.
 */

#ifndef DAEMON_TYPES_H
#define DAEMON_TYPES_H

#include <stdint.h>
#include <stdatomic.h>

#define SHM_NAME          "/cursor-daemon"
#define SHM_MAGIC         0xCDA0CDA0U
#define SHM_VERSION       1
#define MAX_BLOBS         32
#define JPEG_BUF_SIZE     (512 * 1024)   /* 512 KB per JPEG buffer */
#define FRAME_WIDTH       1920
#define FRAME_HEIGHT      1080
#define RECORD_WIDTH      1280
#define RECORD_HEIGHT     720
#define TARGET_FPS        30
#define CHUNK_DURATION_S  300            /* 5 minutes per recording chunk */
#define RING_SIZE         4              /* Recording ring buffer slots */

/* Cursor tracking method — matches Python CURSOR_METHODS dict */
enum CursorMethod {
    METHOD_UNKNOWN       = 0,
    METHOD_MOTION_TRACK  = 1,
    METHOD_SHAPE_TRACK   = 2,
    METHOD_MICRO_SHAKE   = 3,
    METHOD_MACRO_SHAKE   = 4,
    METHOD_LISSAJOUS     = 5,
    METHOD_API_PROBE     = 6,
    METHOD_CNN_REACQUIRE = 7,
};

/* Single motion blob — matches Python MotionBlob dataclass */
typedef struct __attribute__((packed)) {
    int16_t  centroid_x;        /*  0 */
    int16_t  centroid_y;        /*  2 */
    float    angle;             /*  4: principal axis angle (radians) */
    int32_t  pixel_count;       /*  8: number of changed pixels */
    int16_t  bbox_x_min;       /* 12 */
    int16_t  bbox_y_min;       /* 14 */
    int16_t  bbox_x_max;       /* 16 */
    int16_t  bbox_y_max;       /* 18 */
    int16_t  _pad[2];          /* 20: align to 24 bytes total */
} ShmBlob;                     /* sizeof = 24 */

/* Main shared memory region — fixed layout, ~1 MB total */
typedef struct __attribute__((packed)) {
    /* Header */
    uint32_t magic;                    /*    0: SHM_MAGIC for validity check */
    uint32_t version;                  /*    4: SHM_VERSION */

    /* Lock-free sequence counter (begin) */
    _Atomic uint64_t seq_begin;        /*    8 */

    /* Cursor state */
    int32_t  cursor_x;                 /*   16 */
    int32_t  cursor_y;                 /*   20 */
    float    cursor_age_s;             /*   24 */
    uint8_t  cursor_method;            /*   28: enum CursorMethod */
    uint8_t  _pad1[3];                /*   29 */

    /* Blob data */
    int32_t  blob_count;               /*   32 */
    ShmBlob  blobs[MAX_BLOBS];         /*   36: 32 * 24 = 768 bytes */

    /* Performance */
    float    fps;                      /*  804 */
    uint64_t frame_count;              /*  808 */
    uint64_t timestamp_ns;             /*  816: clock_gettime CLOCK_MONOTONIC */

    /* Daemon status */
    uint8_t  daemon_running;           /*  824 */
    uint8_t  recording_active;         /*  825 */
    uint8_t  _pad2[6];               /*  826 */

    /* Lock-free sequence counter (end) */
    _Atomic uint64_t seq_end;          /*  832 */

    uint8_t  _pad3[8];               /*  840: alignment */

    /* Double-buffered JPEG frames for dashboard streaming */
    _Atomic uint32_t jpeg0_size;       /*  848 */
    uint8_t  jpeg0_data[JPEG_BUF_SIZE]; /* 852 .. 852+524287 = 525139 */

    _Atomic uint32_t jpeg1_size;       /* 525140 */
    uint8_t  jpeg1_data[JPEG_BUF_SIZE]; /* 525144 .. 1049431 */

    /* Which buffer is latest: 0 or 1 */
    _Atomic uint8_t jpeg_active;       /* 1049432 */

} CursorDaemonShm;

#define SHM_SIZE sizeof(CursorDaemonShm)

/* Byte offsets for Python struct.unpack_from — keep in sync with above */
/*
 * OFF_MAGIC          =    0
 * OFF_VERSION        =    4
 * OFF_SEQ_BEGIN      =    8
 * OFF_CURSOR_X       =   16
 * OFF_CURSOR_Y       =   20
 * OFF_CURSOR_AGE     =   24
 * OFF_CURSOR_METHOD  =   28
 * OFF_BLOB_COUNT     =   32
 * OFF_BLOBS          =   36   (each 24 bytes, 32 entries)
 * OFF_FPS            =  804
 * OFF_FRAME_COUNT    =  808
 * OFF_TIMESTAMP_NS   =  816
 * OFF_DAEMON_RUN     =  824
 * OFF_RECORDING      =  825
 * OFF_SEQ_END        =  832
 * OFF_JPEG0_SIZE     =  848
 * OFF_JPEG0_DATA     =  852
 * OFF_JPEG1_SIZE     =  852 + JPEG_BUF_SIZE
 * OFF_JPEG1_DATA     =  852 + JPEG_BUF_SIZE + 4
 * OFF_JPEG_ACTIVE    =  852 + JPEG_BUF_SIZE + 4 + JPEG_BUF_SIZE  (= 1049432)
 */

#endif /* DAEMON_TYPES_H */
