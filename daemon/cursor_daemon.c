/*
 * cursor_daemon.c — Real-time HDMI cursor detection daemon.
 *
 * Captures frames from a V4L2 HDMI capture card via libavdevice,
 * performs frame differencing + connected component labeling to
 * extract motion blobs, tracks the cursor, encodes JPEG for the
 * dashboard, and records time-chunked MKV video.
 *
 * All state is exposed via POSIX shared memory for zero-copy reads
 * by the Python KVM dashboard (daemon_client.py).
 *
 * Build:
 *   make   (or see Makefile)
 *
 * Usage:
 *   ./cursor-daemon [--device /dev/video0] [--threshold 12]
 *                   [--record-dir /tmp/kvm_recordings]
 *                   [--no-record] [--verbose]
 */

/* _GNU_SOURCE defined via -D in Makefile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <getopt.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <turbojpeg.h>

#include "daemon_types.h"

/* ── Configuration ────────────────────────────────────────────── */

typedef struct {
    const char *device;
    int         threshold;
    int         min_pixels;
    int         max_pixels;
    const char *record_dir;
    int         no_record;
    int         verbose;
} Config;

static Config cfg = {
    .device     = "/dev/video0",
    .threshold  = 12,
    .min_pixels = 5,
    .max_pixels = 3000,
    .record_dir = NULL,  /* set in main() based on project root */
    .no_record  = 0,
    .verbose    = 0,
};

/* ── Globals ──────────────────────────────────────────────────── */

static volatile sig_atomic_t g_running = 1;
static CursorDaemonShm      *g_shm    = NULL;
static int                    g_shm_fd = -1;

/* Recording ring buffer */
typedef struct {
    AVFrame       *frames[RING_SIZE];
    _Atomic int    write_idx;
    _Atomic int    read_idx;
    _Atomic int    active;       /* 0 = stopped, 1 = running */
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
} FrameRing;

static FrameRing g_ring;

/* ── Signal Handling ──────────────────────────────────────────── */

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

/* ── Utility ──────────────────────────────────────────────────── */

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        mkdir(path, 0755);
    }
}

/* ── Union-Find for Connected Component Labeling ──────────────── */

#define MAX_LABELS 16384

typedef struct {
    int parent[MAX_LABELS];
    int rank[MAX_LABELS];
    int next_label;
} UnionFind;

static void uf_init(UnionFind *uf) {
    uf->next_label = 1;  /* 0 = background */
    for (int i = 0; i < MAX_LABELS; i++) {
        uf->parent[i] = i;
        uf->rank[i] = 0;
    }
}

static int uf_find(UnionFind *uf, int x) {
    while (uf->parent[x] != x) {
        uf->parent[x] = uf->parent[uf->parent[x]]; /* Path halving */
        x = uf->parent[x];
    }
    return x;
}

static void uf_union(UnionFind *uf, int a, int b) {
    a = uf_find(uf, a);
    b = uf_find(uf, b);
    if (a == b) return;
    if (uf->rank[a] < uf->rank[b]) { int t = a; a = b; b = t; }
    uf->parent[b] = a;
    if (uf->rank[a] == uf->rank[b]) uf->rank[a]++;
}

static int uf_new_label(UnionFind *uf) {
    if (uf->next_label >= MAX_LABELS) return MAX_LABELS - 1;
    int l = uf->next_label++;
    uf->parent[l] = l;
    uf->rank[l] = 0;
    return l;
}

/* ── Frame Differencing ───────────────────────────────────────── */

static void frame_diff_y(
    const uint8_t *y_prev,
    const uint8_t *y_curr,
    uint8_t       *mask,
    int w, int h, int threshold)
{
    int n = w * h;
    for (int i = 0; i < n; i++) {
        int d = (int)y_curr[i] - (int)y_prev[i];
        mask[i] = (d > threshold || d < -threshold) ? 1 : 0;
    }
}

/* ── Blob Extraction (Two-Pass CCL) ──────────────────────────── */

/* Per-component accumulator for stats */
typedef struct {
    int64_t sum_x, sum_y;
    double  sum_xx, sum_yy, sum_xy;
    int     min_x, min_y, max_x, max_y;
    int     count;
} BlobAccum;

static float compute_principal_angle(const BlobAccum *a) {
    if (a->count < 3) return 0.0f;

    double mx = (double)a->sum_x / a->count;
    double my = (double)a->sum_y / a->count;
    double cxx = a->sum_xx / a->count - mx * mx;
    double cxy = a->sum_xy / a->count - mx * my;
    double cyy = a->sum_yy / a->count - my * my;

    double trace = cxx + cyy;
    double det   = cxx * cyy - cxy * cxy;
    double disc  = trace * trace / 4.0 - det;
    if (disc < 0.0) disc = 0.0;

    double lambda1 = trace / 2.0 + sqrt(disc);

    if (fabs(cxy) > 1e-10)
        return (float)atan2(lambda1 - cxx, cxy);
    else if (cxx >= cyy)
        return 0.0f;
    else
        return (float)(M_PI / 2.0);
}

static int extract_blobs(
    const uint8_t *mask,
    int           *labels,
    int            w, int h,
    ShmBlob       *out_blobs,
    int            max_out,
    int            min_px, int max_px)
{
    UnionFind uf;
    uf_init(&uf);
    memset(labels, 0, w * h * sizeof(int));

    /* Pass 1: Assign labels with 8-connectivity */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (!mask[idx]) continue;

            /* Check 4 already-visited neighbors (up-left, up, up-right, left) */
            int neighbors[4];
            int nn = 0;

            if (y > 0 && x > 0 && labels[(y-1)*w + (x-1)])
                neighbors[nn++] = labels[(y-1)*w + (x-1)];
            if (y > 0 && labels[(y-1)*w + x])
                neighbors[nn++] = labels[(y-1)*w + x];
            if (y > 0 && x < w-1 && labels[(y-1)*w + (x+1)])
                neighbors[nn++] = labels[(y-1)*w + (x+1)];
            if (x > 0 && labels[y*w + (x-1)])
                neighbors[nn++] = labels[y*w + (x-1)];

            if (nn == 0) {
                labels[idx] = uf_new_label(&uf);
            } else {
                /* Find minimum root label */
                int min_label = uf_find(&uf, neighbors[0]);
                for (int i = 1; i < nn; i++) {
                    int root = uf_find(&uf, neighbors[i]);
                    if (root < min_label) min_label = root;
                }
                labels[idx] = min_label;
                /* Union all neighbors */
                for (int i = 0; i < nn; i++) {
                    uf_union(&uf, min_label, neighbors[i]);
                }
            }
        }
    }

    /* Pass 2: Resolve labels + accumulate stats */
    int num_labels = uf.next_label;
    BlobAccum *accum = calloc(num_labels, sizeof(BlobAccum));
    if (!accum) return 0;

    /* Initialize bboxes */
    for (int i = 0; i < num_labels; i++) {
        accum[i].min_x = w;
        accum[i].min_y = h;
        accum[i].max_x = 0;
        accum[i].max_y = 0;
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (!labels[idx]) continue;

            int root = uf_find(&uf, labels[idx]);
            labels[idx] = root;

            BlobAccum *a = &accum[root];
            a->sum_x += x;
            a->sum_y += y;
            a->sum_xx += (double)x * x;
            a->sum_yy += (double)y * y;
            a->sum_xy += (double)x * y;
            if (x < a->min_x) a->min_x = x;
            if (y < a->min_y) a->min_y = y;
            if (x > a->max_x) a->max_x = x;
            if (y > a->max_y) a->max_y = y;
            a->count++;
        }
    }

    /* Collect blobs that pass size filter */
    typedef struct { int pixel_count; int label; } BlobSort;
    BlobSort *sortable = malloc(num_labels * sizeof(BlobSort));
    int n_valid = 0;

    for (int i = 1; i < num_labels; i++) {
        if (accum[i].count >= min_px && accum[i].count <= max_px) {
            sortable[n_valid].pixel_count = accum[i].count;
            sortable[n_valid].label = i;
            n_valid++;
        }
    }

    /* Sort by pixel_count descending (simple insertion sort, max 32) */
    for (int i = 1; i < n_valid; i++) {
        BlobSort tmp = sortable[i];
        int j = i - 1;
        while (j >= 0 && sortable[j].pixel_count < tmp.pixel_count) {
            sortable[j+1] = sortable[j];
            j--;
        }
        sortable[j+1] = tmp;
    }

    int out_count = n_valid < max_out ? n_valid : max_out;
    for (int i = 0; i < out_count; i++) {
        BlobAccum *a = &accum[sortable[i].label];
        out_blobs[i].centroid_x = (int16_t)(a->sum_x / a->count);
        out_blobs[i].centroid_y = (int16_t)(a->sum_y / a->count);
        out_blobs[i].angle      = compute_principal_angle(a);
        out_blobs[i].pixel_count = a->count;
        out_blobs[i].bbox_x_min = (int16_t)a->min_x;
        out_blobs[i].bbox_y_min = (int16_t)a->min_y;
        out_blobs[i].bbox_x_max = (int16_t)a->max_x;
        out_blobs[i].bbox_y_max = (int16_t)a->max_y;
        out_blobs[i]._pad[0]    = 0;
        out_blobs[i]._pad[1]    = 0;
    }

    free(accum);
    free(sortable);
    return out_count;
}

/* ── Recording Thread ─────────────────────────────────────────── */

typedef struct {
    AVFormatContext *fmt_ctx;
    AVCodecContext  *enc_ctx;
    AVStream        *stream;
    struct SwsContext *sws;
    AVFrame         *scaled_frame;
    int64_t          pts;
    time_t           chunk_start;
    char             current_path[512];
} RecState;

static int open_recording(RecState *rs) {
    char path[512];
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    snprintf(path, sizeof(path), "%s/kvm_%04d%02d%02d_%02d%02d%02d.mkv",
             cfg.record_dir,
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);

    int ret = avformat_alloc_output_context2(&rs->fmt_ctx, NULL, "matroska", path);
    if (ret < 0) {
        fprintf(stderr, "record: failed to alloc output context: %d\n", ret);
        return -1;
    }

    /* Try NVENC (GPU) first, fall back to libx264 (CPU) */
    const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
    int using_nvenc = (codec != NULL);
    if (!codec) {
        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    }
    if (!codec) {
        fprintf(stderr, "record: H.264 encoder not found\n");
        avformat_free_context(rs->fmt_ctx);
        rs->fmt_ctx = NULL;
        return -1;
    }

    rs->stream = avformat_new_stream(rs->fmt_ctx, codec);
    rs->enc_ctx = avcodec_alloc_context3(codec);
    rs->enc_ctx->width       = RECORD_WIDTH;
    rs->enc_ctx->height      = RECORD_HEIGHT;
    rs->enc_ctx->time_base   = (AVRational){1, TARGET_FPS};
    rs->enc_ctx->framerate   = (AVRational){TARGET_FPS, 1};
    rs->enc_ctx->pix_fmt     = AV_PIX_FMT_YUV420P;
    rs->enc_ctx->gop_size    = TARGET_FPS;   /* Keyframe every 1s */
    rs->enc_ctx->max_b_frames = 0;

    if (using_nvenc) {
        /* NVENC: use hardware-accelerated preset + constant quality */
        av_opt_set(rs->enc_ctx->priv_data, "preset", "p1", 0);  /* fastest */
        av_opt_set(rs->enc_ctx->priv_data, "rc", "constqp", 0);
        av_opt_set(rs->enc_ctx->priv_data, "qp", "28", 0);
        printf("record: using NVENC (GPU) encoder\n");
    } else {
        av_opt_set(rs->enc_ctx->priv_data, "preset", "ultrafast", 0);
        av_opt_set(rs->enc_ctx->priv_data, "crf", "23", 0);
        printf("record: using libx264 (CPU) encoder\n");
    }

    if (rs->fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        rs->enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    ret = avcodec_open2(rs->enc_ctx, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "record: failed to open encoder: %d\n", ret);
        avcodec_free_context(&rs->enc_ctx);
        avformat_free_context(rs->fmt_ctx);
        rs->fmt_ctx = NULL;
        return -1;
    }

    avcodec_parameters_from_context(rs->stream->codecpar, rs->enc_ctx);
    rs->stream->time_base = rs->enc_ctx->time_base;

    ret = avio_open(&rs->fmt_ctx->pb, path, AVIO_FLAG_WRITE);
    if (ret < 0) {
        fprintf(stderr, "record: failed to open file %s: %d\n", path, ret);
        avcodec_free_context(&rs->enc_ctx);
        avformat_free_context(rs->fmt_ctx);
        rs->fmt_ctx = NULL;
        return -1;
    }

    ret = avformat_write_header(rs->fmt_ctx, NULL);
    if (ret < 0) {
        fprintf(stderr, "record: failed to write header: %d\n", ret);
        avio_closep(&rs->fmt_ctx->pb);
        avcodec_free_context(&rs->enc_ctx);
        avformat_free_context(rs->fmt_ctx);
        rs->fmt_ctx = NULL;
        return -1;
    }

    rs->pts = 0;
    rs->chunk_start = now;
    snprintf(rs->current_path, sizeof(rs->current_path), "%s", path);

    if (cfg.verbose)
        printf("record: opened %s\n", path);

    return 0;
}

static void close_recording(RecState *rs) {
    if (!rs->fmt_ctx) return;

    /* Flush encoder */
    avcodec_send_frame(rs->enc_ctx, NULL);
    AVPacket *pkt = av_packet_alloc();
    while (avcodec_receive_packet(rs->enc_ctx, pkt) == 0) {
        av_packet_rescale_ts(pkt, rs->enc_ctx->time_base, rs->stream->time_base);
        pkt->stream_index = rs->stream->index;
        av_interleaved_write_frame(rs->fmt_ctx, pkt);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);

    av_write_trailer(rs->fmt_ctx);
    avio_closep(&rs->fmt_ctx->pb);
    avcodec_free_context(&rs->enc_ctx);
    avformat_free_context(rs->fmt_ctx);
    rs->fmt_ctx = NULL;

    if (cfg.verbose)
        printf("record: closed %s\n", rs->current_path);
}

static void *recording_thread(void *arg) {
    (void)arg;

    RecState rs;
    memset(&rs, 0, sizeof(rs));

    /* Allocate scaled frame for 1280x720 */
    rs.scaled_frame = av_frame_alloc();
    rs.scaled_frame->format = AV_PIX_FMT_YUV420P;
    rs.scaled_frame->width  = RECORD_WIDTH;
    rs.scaled_frame->height = RECORD_HEIGHT;
    av_frame_get_buffer(rs.scaled_frame, 0);

    rs.sws = NULL;  /* Initialized on first frame */

    ensure_dir(cfg.record_dir);

    if (open_recording(&rs) < 0) {
        fprintf(stderr, "record: failed to start recording, thread exiting\n");
        av_frame_free(&rs.scaled_frame);
        return NULL;
    }
    g_shm->recording_active = 1;

    while (g_running || atomic_load(&g_ring.read_idx) < atomic_load(&g_ring.write_idx)) {
        /* Wait for frame */
        int ri = atomic_load(&g_ring.read_idx);
        int wi = atomic_load(&g_ring.write_idx);

        if (ri >= wi) {
            /* No frames available — brief sleep */
            struct timespec ts = {0, 10000000}; /* 10ms */
            nanosleep(&ts, NULL);
            continue;
        }

        int slot = ri % RING_SIZE;
        AVFrame *frame = g_ring.frames[slot];
        if (!frame) {
            atomic_fetch_add(&g_ring.read_idx, 1);
            continue;
        }

        /* Initialize scaler on first frame (we learn input format here) */
        if (!rs.sws) {
            rs.sws = sws_getContext(
                frame->width, frame->height, frame->format,
                RECORD_WIDTH, RECORD_HEIGHT, AV_PIX_FMT_YUV420P,
                SWS_BILINEAR, NULL, NULL, NULL);
            if (!rs.sws) {
                fprintf(stderr, "record: failed to create scaler\n");
                atomic_fetch_add(&g_ring.read_idx, 1);
                continue;
            }
        }

        /* Scale to 1280x720 */
        av_frame_make_writable(rs.scaled_frame);
        sws_scale(rs.sws,
                  (const uint8_t * const *)frame->data, frame->linesize,
                  0, frame->height,
                  rs.scaled_frame->data, rs.scaled_frame->linesize);

        rs.scaled_frame->pts = rs.pts++;

        /* Encode */
        avcodec_send_frame(rs.enc_ctx, rs.scaled_frame);
        AVPacket *pkt = av_packet_alloc();
        while (avcodec_receive_packet(rs.enc_ctx, pkt) == 0) {
            av_packet_rescale_ts(pkt, rs.enc_ctx->time_base, rs.stream->time_base);
            pkt->stream_index = rs.stream->index;
            av_interleaved_write_frame(rs.fmt_ctx, pkt);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);

        /* Release frame slot */
        av_frame_free(&g_ring.frames[slot]);
        g_ring.frames[slot] = NULL;
        atomic_fetch_add(&g_ring.read_idx, 1);

        /* Time-chunk rotation */
        time_t now = time(NULL);
        if (now - rs.chunk_start >= CHUNK_DURATION_S) {
            close_recording(&rs);
            if (open_recording(&rs) < 0) {
                fprintf(stderr, "record: failed to rotate chunk, stopping\n");
                break;
            }
        }
    }

    close_recording(&rs);
    g_shm->recording_active = 0;

    if (rs.sws) sws_freeContext(rs.sws);
    av_frame_free(&rs.scaled_frame);

    return NULL;
}

static void enqueue_frame(AVFrame *frame) {
    int wi = atomic_load(&g_ring.write_idx);
    int ri = atomic_load(&g_ring.read_idx);

    /* Drop frame if ring is full (recording can't keep up) */
    if (wi - ri >= RING_SIZE) {
        return;
    }

    int slot = wi % RING_SIZE;
    g_ring.frames[slot] = av_frame_clone(frame);
    atomic_fetch_add(&g_ring.write_idx, 1);
}

/* ── Shared Memory Setup ──────────────────────────────────────── */

static int setup_shm(void) {
    g_shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (g_shm_fd < 0) {
        perror("shm_open");
        return -1;
    }

    if (ftruncate(g_shm_fd, SHM_SIZE) < 0) {
        perror("ftruncate");
        close(g_shm_fd);
        shm_unlink(SHM_NAME);
        return -1;
    }

    g_shm = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
    if (g_shm == MAP_FAILED) {
        perror("mmap");
        close(g_shm_fd);
        shm_unlink(SHM_NAME);
        return -1;
    }

    memset(g_shm, 0, SHM_SIZE);
    g_shm->magic   = SHM_MAGIC;
    g_shm->version = SHM_VERSION;
    g_shm->daemon_running = 1;

    return 0;
}

static void cleanup_shm(void) {
    if (g_shm) {
        g_shm->daemon_running = 0;
        munmap(g_shm, SHM_SIZE);
        g_shm = NULL;
    }
    if (g_shm_fd >= 0) {
        close(g_shm_fd);
        g_shm_fd = -1;
    }
    shm_unlink(SHM_NAME);
}

/* ── JPEG Encoding ────────────────────────────────────────────── */

static int encode_jpeg_to_shm(
    tjhandle compressor,
    const AVFrame *frame,
    struct SwsContext *sws_rgb)
{
    /* Convert frame to RGB for JPEG encoding */
    uint8_t *rgb_data[1];
    int rgb_linesize[1];
    int w = frame->width;
    int h = frame->height;

    rgb_data[0] = malloc(w * h * 3);
    if (!rgb_data[0]) return -1;
    rgb_linesize[0] = w * 3;

    sws_scale(sws_rgb,
              (const uint8_t * const *)frame->data, frame->linesize,
              0, h,
              rgb_data, rgb_linesize);

    /* Compress to JPEG */
    unsigned char *jpeg_buf = NULL;
    unsigned long jpeg_size = 0;

    int ret = tjCompress2(compressor, rgb_data[0], w, 0, h,
                          TJPF_RGB, &jpeg_buf, &jpeg_size,
                          TJSAMP_420, 80, TJFLAG_FASTDCT);
    free(rgb_data[0]);

    if (ret != 0 || jpeg_size == 0 || jpeg_size > JPEG_BUF_SIZE) {
        if (jpeg_buf) tjFree(jpeg_buf);
        return -1;
    }

    /* Write to inactive double-buffer slot */
    int active = atomic_load(&g_shm->jpeg_active);
    int target = 1 - active;

    if (target == 0) {
        memcpy(g_shm->jpeg0_data, jpeg_buf, jpeg_size);
        uint32_t sz = (uint32_t)jpeg_size;
        memcpy((void *)&g_shm->jpeg0_size, &sz, sizeof(sz));
    } else {
        memcpy(g_shm->jpeg1_data, jpeg_buf, jpeg_size);
        uint32_t sz = (uint32_t)jpeg_size;
        memcpy((void *)&g_shm->jpeg1_size, &sz, sizeof(sz));
    }

    /* Atomically flip active buffer */
    atomic_store(&g_shm->jpeg_active, (uint8_t)target);

    tjFree(jpeg_buf);
    return 0;
}

/* ── Command-Line Parsing ─────────────────────────────────────── */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --device PATH      V4L2 device (default: /dev/video0)\n"
        "  --threshold N      Motion threshold on Y channel (default: 12)\n"
        "  --record-dir PATH  Recording output directory\n"
        "  --no-record        Disable recording\n"
        "  --verbose          Print per-frame stats\n"
        "  -h, --help         Show this help\n",
        prog);
}

static void parse_args(int argc, char **argv) {
    static struct option long_opts[] = {
        {"device",     required_argument, NULL, 'd'},
        {"threshold",  required_argument, NULL, 't'},
        {"record-dir", required_argument, NULL, 'r'},
        {"no-record",  no_argument,       NULL, 'n'},
        {"verbose",    no_argument,       NULL, 'v'},
        {"help",       no_argument,       NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:t:r:nvh", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'd': cfg.device     = optarg; break;
        case 't': cfg.threshold  = atoi(optarg); break;
        case 'r': cfg.record_dir = optarg; break;
        case 'n': cfg.no_record  = 1; break;
        case 'v': cfg.verbose    = 1; break;
        case 'h': print_usage(argv[0]); exit(0);
        default:  print_usage(argv[0]); exit(1);
        }
    }
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    parse_args(argc, argv);

    /* Default record dir: relative to daemon binary location */
    char default_record_dir[512];
    if (!cfg.record_dir) {
        /* Try to find project root (one level up from daemon/) */
        char *exe_dir = realpath("/proc/self/exe", NULL);
        if (exe_dir) {
            char *slash = strrchr(exe_dir, '/');
            if (slash) *slash = '\0';  /* Remove binary name */
            slash = strrchr(exe_dir, '/');
            if (slash) *slash = '\0';  /* Remove daemon/ */
            snprintf(default_record_dir, sizeof(default_record_dir),
                     "%s/data/recordings", exe_dir);
            free(exe_dir);
        } else {
            strncpy(default_record_dir, "/tmp/kvm_recordings",
                    sizeof(default_record_dir) - 1);
        }
        cfg.record_dir = default_record_dir;
    }

    printf("cursor-daemon starting\n");
    printf("  device:     %s\n", cfg.device);
    printf("  threshold:  %d\n", cfg.threshold);
    printf("  record-dir: %s\n", cfg.record_dir);
    printf("  recording:  %s\n", cfg.no_record ? "disabled" : "enabled");

    /* Signal handlers */
    signal(SIGTERM, signal_handler);
    signal(SIGINT,  signal_handler);

    /* Register ffmpeg devices */
    avdevice_register_all();

    /* Setup shared memory */
    if (setup_shm() < 0) {
        fprintf(stderr, "Failed to setup shared memory\n");
        return 1;
    }

    /* Open V4L2 capture device via libavdevice */
    const AVInputFormat *ifmt = av_find_input_format("v4l2");
    if (!ifmt) {
        fprintf(stderr, "v4l2 input format not found\n");
        cleanup_shm();
        return 1;
    }

    AVDictionary *options = NULL;
    av_dict_set(&options, "video_size", "1920x1080", 0);
    av_dict_set(&options, "input_format", "mjpeg", 0);
    av_dict_set(&options, "framerate", "30", 0);

    AVFormatContext *input_ctx = NULL;
    int ret = avformat_open_input(&input_ctx, cfg.device, ifmt, &options);
    av_dict_free(&options);

    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "Failed to open %s: %s\n", cfg.device, errbuf);
        cleanup_shm();
        return 1;
    }

    avformat_find_stream_info(input_ctx, NULL);

    /* Find video stream */
    int video_idx = -1;
    for (unsigned i = 0; i < input_ctx->nb_streams; i++) {
        if (input_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = i;
            break;
        }
    }
    if (video_idx < 0) {
        fprintf(stderr, "No video stream found\n");
        avformat_close_input(&input_ctx);
        cleanup_shm();
        return 1;
    }

    /* Open decoder */
    AVCodecParameters *codecpar = input_ctx->streams[video_idx]->codecpar;
    const AVCodec *decoder = avcodec_find_decoder(codecpar->codec_id);
    AVCodecContext *dec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(dec_ctx, codecpar);
    avcodec_open2(dec_ctx, decoder, NULL);

    printf("  input: %dx%d %s @ %dfps\n",
           dec_ctx->width, dec_ctx->height,
           avcodec_get_name(dec_ctx->codec_id),
           TARGET_FPS);

    /* SwsContext for decoded frame -> RGB (for JPEG) */
    struct SwsContext *sws_rgb = NULL;

    /* TurboJPEG compressor */
    tjhandle jpeg_compressor = tjInitCompress();
    if (!jpeg_compressor) {
        fprintf(stderr, "Failed to init turbojpeg\n");
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&input_ctx);
        cleanup_shm();
        return 1;
    }

    /* Start recording thread */
    pthread_t rec_tid = 0;
    memset(&g_ring, 0, sizeof(g_ring));
    atomic_store(&g_ring.write_idx, 0);
    atomic_store(&g_ring.read_idx, 0);

    if (!cfg.no_record) {
        if (pthread_create(&rec_tid, NULL, recording_thread, NULL) != 0) {
            fprintf(stderr, "Failed to create recording thread\n");
            rec_tid = 0;
        }
    }

    /* Allocate work buffers */
    int w = dec_ctx->width;
    int h = dec_ctx->height;
    uint8_t *y_prev    = calloc(w * h, 1);
    uint8_t *y_curr    = calloc(w * h, 1);
    uint8_t *diff_mask = calloc(w * h, 1);
    int     *labels    = calloc(w * h, sizeof(int));

    if (!y_prev || !y_curr || !diff_mask || !labels) {
        fprintf(stderr, "Failed to allocate work buffers\n");
        goto cleanup;
    }

    int has_prev = 0;

    /* FPS tracking */
    uint64_t fps_start = now_ns();
    int fps_count = 0;

    printf("  capture loop starting...\n");

    /* ── Main capture loop ────────────────────────────────────── */

    AVPacket *pkt = av_packet_alloc();
    AVFrame  *frame = av_frame_alloc();

    while (g_running) {
        ret = av_read_frame(input_ctx, pkt);
        if (ret < 0) {
            if (ret == AVERROR(EAGAIN)) {
                usleep(1000);
                continue;
            }
            fprintf(stderr, "av_read_frame error: %d\n", ret);
            break;
        }

        if (pkt->stream_index != video_idx) {
            av_packet_unref(pkt);
            continue;
        }

        /* Decode MJPEG -> YUV frame */
        ret = avcodec_send_packet(dec_ctx, pkt);
        av_packet_unref(pkt);
        if (ret < 0) continue;

        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret < 0) continue;

        /* Initialize SwsContext on first frame (learn actual pixel format) */
        if (!sws_rgb) {
            sws_rgb = sws_getContext(
                frame->width, frame->height, frame->format,
                frame->width, frame->height, AV_PIX_FMT_RGB24,
                SWS_BILINEAR, NULL, NULL, NULL);
        }

        /* Extract Y plane (luma) for motion detection */
        for (int row = 0; row < h; row++) {
            memcpy(y_curr + row * w,
                   frame->data[0] + row * frame->linesize[0],
                   w);
        }

        /* Enqueue frame for recording (before we process - recording gets raw) */
        if (!cfg.no_record) {
            enqueue_frame(frame);
        }

        /* Motion detection */
        ShmBlob local_blobs[MAX_BLOBS];
        int blob_count = 0;

        if (has_prev) {
            frame_diff_y(y_prev, y_curr, diff_mask, w, h, cfg.threshold);
            blob_count = extract_blobs(diff_mask, labels, w, h,
                                       local_blobs, MAX_BLOBS,
                                       cfg.min_pixels, cfg.max_pixels);
            /* Cursor identification is NOT done here — the daemon only
             * reports raw blobs. Python (which controls the mouse and
             * knows when jitter/Lissajous fires) identifies which blob
             * is the cursor by correlating with known movements. */
        }

        /* Encode JPEG to shared memory */
        if (sws_rgb) {
            encode_jpeg_to_shm(jpeg_compressor, frame, sws_rgb);
        }

        /* Update shared memory (seq counter protocol) */
        uint64_t seq;
        memcpy(&seq, (void *)&g_shm->seq_begin, sizeof(seq));
        seq++;
        memcpy((void *)&g_shm->seq_begin, &seq, sizeof(seq));
        atomic_thread_fence(memory_order_release);

        /* Cursor x/y/age/method are NOT set by daemon — Python owns cursor
         * identification via jitter correlation + CNN validation. */
        g_shm->blob_count    = blob_count;
        memcpy(g_shm->blobs, local_blobs, blob_count * sizeof(ShmBlob));
        g_shm->frame_count++;
        g_shm->timestamp_ns  = now_ns();

        atomic_thread_fence(memory_order_release);
        memcpy((void *)&g_shm->seq_end, &seq, sizeof(seq));

        /* FPS calculation */
        fps_count++;
        uint64_t elapsed = now_ns() - fps_start;
        if (elapsed >= 2000000000ULL) { /* Every 2 seconds */
            g_shm->fps = (float)fps_count * 1e9f / (float)elapsed;
            if (cfg.verbose) {
                printf("  fps=%.1f blobs=%d frames=%lu\n",
                       g_shm->fps, blob_count,
                       (unsigned long)g_shm->frame_count);
            }
            fps_count = 0;
            fps_start = now_ns();
        }

        /* Swap Y buffers */
        uint8_t *tmp = y_prev;
        y_prev = y_curr;
        y_curr = tmp;
        has_prev = 1;
    }

    printf("cursor-daemon shutting down...\n");

cleanup:
    /* Wait for recording thread to finish */
    if (rec_tid) {
        pthread_join(rec_tid, NULL);
    }

    /* Free resources */
    av_packet_free(&pkt);
    av_frame_free(&frame);
    if (sws_rgb) sws_freeContext(sws_rgb);
    tjDestroy(jpeg_compressor);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&input_ctx);

    free(y_prev);
    free(y_curr);
    free(diff_mask);
    free(labels);

    cleanup_shm();
    printf("cursor-daemon stopped\n");
    return 0;
}
