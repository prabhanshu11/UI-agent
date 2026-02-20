/* profiler.js — Timeline scrubber (video-editor style) + pipeline profiler. */

var serverUtc = '';
var httpRt = 0;
var barMax = 50;
var history = [];
var MAX_HISTORY = 100;
var testRunning = false;
var testPollId = null;
var previewSeq = 0;

// Timeline data: ring buffer of per-frame timing
var timelineData = [];
var MAX_TIMELINE = 200;
var timelineHoverIdx = -1;

var STAGE_COLORS = {
    decode: '#3498db', track: '#2ecc71', overlay: '#9b59b6',
    encode: '#e67e22', total: '#e74c3c', http: '#f1c40f',
};
var STAGE_KEYS = {
    decode: 'jpeg_decode_ms', track: 'silhouette_ms',
    overlay: 'overlay_draw_ms', encode: 'jpeg_encode_ms',
    total: 'frame_total_ms', http: 'http_ms',
};
var STAGE_NAMES = {
    jpeg_decode_ms: 'JPEG Decode', silhouette_ms: 'Sil. Track',
    overlay_draw_ms: 'Overlay Draw', jpeg_encode_ms: 'JPEG Encode',
    frame_total_ms: 'Frame Total',
};

// UTC clock
function tickClock() {
    document.getElementById('browser-utc').textContent = new Date().toISOString().slice(11, 23);
    if (serverUtc) {
        var d = Date.now() - new Date(serverUtc).getTime();
        document.getElementById('utc-delta').textContent = d + 'ms';
    }
}
setInterval(tickClock, 100);

function setBar(name, ms) {
    var b = document.getElementById('bar-' + name);
    var v = document.getElementById('val-' + name);
    if (!b || !v) return;
    b.style.width = Math.min(100, (ms / barMax) * 100) + '%';
    v.textContent = ms.toFixed(1) + 'ms';
}

// ── Timeline scrubber ────────────────────────────────────────
function addTimelineFrame(data) {
    var entry = {
        decode: data.pipeline.jpeg_decode_ms,
        track: data.pipeline.silhouette_ms,
        overlay: data.pipeline.overlay_draw_ms,
        encode: data.pipeline.jpeg_encode_ms,
        total: data.pipeline.frame_total_ms,
        http: httpRt,
        ts: data.utc,
    };
    timelineData.push(entry);
    if (timelineData.length > MAX_TIMELINE) timelineData.shift();
    renderTimeline();
}

function renderTimeline() {
    var container = document.getElementById('timeline-container');
    if (!container) return;
    var W = container.offsetWidth;
    var stages = ['decode', 'track', 'overlay', 'encode', 'total', 'http'];

    // Remove old blocks (keep tracks, playhead, tooltip)
    container.querySelectorAll('.timeline-block').forEach(function(b) { b.remove(); });

    var n = timelineData.length;
    if (n === 0) return;

    var blockW = Math.max(2, (W - 90) / MAX_TIMELINE); // leave room for labels

    for (var fi = 0; fi < n; fi++) {
        var frame = timelineData[fi];
        var x = 90 + fi * blockW;

        for (var si = 0; si < stages.length; si++) {
            var stage = stages[si];
            var ms = frame[stage] || 0;
            // Width proportional to ms (max ~50ms = full track width isn't useful,
            // so we just show a fixed-width colored block per frame)
            var block = document.createElement('div');
            block.className = 'timeline-block';
            block.style.left = x + 'px';
            block.style.top = (si * 24 + 2) + 'px';
            block.style.width = Math.max(2, blockW - 1) + 'px';
            // Color intensity based on ms value (darker = less, brighter = more)
            var intensity = Math.min(1, ms / 20);
            block.style.backgroundColor = STAGE_COLORS[stage];
            block.style.opacity = 0.3 + intensity * 0.7;
            block.setAttribute('data-frame', fi);
            container.appendChild(block);
        }
    }

    // Update playhead to latest
    var playhead = document.getElementById('playhead');
    playhead.style.left = (90 + (n - 1) * blockW + blockW / 2) + 'px';
}

// Timeline hover interaction
(function() {
    var container = document.getElementById('timeline-container');
    if (!container) return;

    container.addEventListener('mousemove', function(e) {
        var rect = container.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var blockW = Math.max(2, (container.offsetWidth - 90) / MAX_TIMELINE);
        var idx = Math.floor((x - 90) / blockW);

        if (idx < 0 || idx >= timelineData.length) {
            document.getElementById('timeline-tooltip').style.display = 'none';
            return;
        }

        var frame = timelineData[idx];
        var tooltip = document.getElementById('timeline-tooltip');
        tooltip.style.display = 'block';
        tooltip.style.left = Math.min(x, container.offsetWidth - 200) + 'px';
        tooltip.textContent =
            'Decode: ' + frame.decode.toFixed(1) + 'ms | ' +
            'Track: ' + frame.track.toFixed(1) + 'ms | ' +
            'Overlay: ' + frame.overlay.toFixed(1) + 'ms | ' +
            'Encode: ' + frame.encode.toFixed(1) + 'ms | ' +
            'Total: ' + frame.total.toFixed(1) + 'ms | ' +
            'HTTP: ' + frame.http.toFixed(0) + 'ms';

        // Move playhead
        var playhead = document.getElementById('playhead');
        playhead.style.left = (90 + idx * blockW + blockW / 2) + 'px';
    });

    container.addEventListener('mouseleave', function() {
        document.getElementById('timeline-tooltip').style.display = 'none';
    });
})();

// ── Main poll ────────────────────────────────────────────────
async function poll() {
    try {
        var t0 = Date.now();
        var r = await fetch('/api/profiler');
        httpRt = Date.now() - t0;
        var d = await r.json();
        serverUtc = d.utc;
        document.getElementById('server-utc').textContent = d.utc.slice(11, 23);
        document.getElementById('prof-status').textContent = 'Live';
        document.getElementById('prof-status').style.color = '#2ecc71';

        setBar('decode', d.pipeline.jpeg_decode_ms);
        setBar('track', d.pipeline.silhouette_ms);
        setBar('overlay', d.pipeline.overlay_draw_ms);
        setBar('encode', d.pipeline.jpeg_encode_ms);
        setBar('total', d.pipeline.frame_total_ms);
        setBar('http', httpRt);

        // Daemon stats
        document.getElementById('daemon-fps').textContent = d.daemon.fps.toFixed(1);
        document.getElementById('blob-count').textContent = d.daemon.blob_count;
        document.getElementById('frame-count').textContent = d.daemon.frame_count;

        // Backend metrics
        document.getElementById('m-method').textContent = d.tracking.method;
        document.getElementById('m-hz').textContent = d.tracking.hz;
        var conf = d.tracking.confidence;
        var confEl = document.getElementById('m-conf');
        confEl.textContent = (conf * 100).toFixed(1) + '%';
        confEl.className = 'metric-val ' + (conf > 0.7 ? 'good' : conf > 0.3 ? 'warn' : 'bad');
        document.getElementById('m-misses').textContent = d.tracking.misses;
        document.getElementById('m-age').textContent = d.tracking.cursor_age_s + 's';
        document.getElementById('m-validated').textContent = d.cursor.validated ? 'Yes' : 'No';
        document.getElementById('m-ts').textContent = d.daemon.timestamp_ns;

        // Cursor for preview
        document.getElementById('pv-cursor').textContent = '(' + d.cursor.x + ', ' + d.cursor.y + ')';
        document.getElementById('pv-method').textContent = d.cursor.method;
        document.getElementById('pv-age').textContent = d.cursor.age_s;

        // Sparkline
        history.push(d.pipeline.frame_total_ms);
        if (history.length > MAX_HISTORY) history.shift();
        drawSparkline();

        // Timeline
        addTimelineFrame(d);
    } catch (e) {
        document.getElementById('prof-status').textContent = 'Disconnected';
        document.getElementById('prof-status').style.color = '#e74c3c';
    }
}
setInterval(poll, 200);

// Preview frame at 2Hz
function updatePreview() {
    document.getElementById('preview-frame').src = '/api/frame?' + previewSeq++;
}
setInterval(updatePreview, 500);

// Sparkline
function drawSparkline() {
    var canvas = document.getElementById('spark-canvas');
    var ctx = canvas.getContext('2d');
    var W = canvas.offsetWidth, H = canvas.offsetHeight;
    canvas.width = W * (window.devicePixelRatio || 1);
    canvas.height = H * (window.devicePixelRatio || 1);
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

    ctx.clearRect(0, 0, W, H);
    if (history.length < 2) return;

    var min = Math.min.apply(null, history);
    var max = Math.max.apply(null, history);
    var avg = history.reduce(function(a,b){return a+b;}, 0) / history.length;
    var range = Math.max(max - min, 1);

    document.getElementById('spark-min').textContent = 'min: ' + min.toFixed(1) + 'ms';
    document.getElementById('spark-avg').textContent = 'avg: ' + avg.toFixed(1) + 'ms';
    document.getElementById('spark-max').textContent = 'max: ' + max.toFixed(1) + 'ms';

    var avgY = H - ((avg - min) / range) * (H - 10) - 5;
    ctx.strokeStyle = '#333';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, avgY); ctx.lineTo(W, avgY); ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (var i = 0; i < history.length; i++) {
        var x = (i / (MAX_HISTORY - 1)) * W;
        var y = H - ((history[i] - min) / range) * (H - 10) - 5;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.lineTo((history.length - 1) / (MAX_HISTORY - 1) * W, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fillStyle = 'rgba(231, 76, 60, 0.1)';
    ctx.fill();
}

// 3-minute test
async function startTest() {
    await fetch('/api/profiler/start', {method:'POST'});
    testRunning = true;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('test-status').textContent = 'Running...';
    document.getElementById('test-status').style.color = '#2ecc71';
    testPollId = setInterval(pollResults, 1000);
}

async function stopTest() {
    await fetch('/api/profiler/stop', {method:'POST'});
    testRunning = false;
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('test-status').textContent = 'Complete';
    document.getElementById('test-status').style.color = '#f1c40f';
    if (testPollId) { clearInterval(testPollId); testPollId = null; }
    pollResults();
}

async function pollResults() {
    try {
        var r = await fetch('/api/profiler/results');
        var d = await r.json();
        if (d.status === 'no_data') return;
        document.getElementById('test-progress').textContent =
            'Samples: ' + d.samples + ' | ' + d.elapsed_s + 's / ' + d.duration_s + 's';
        if (d.status === 'complete' && testRunning) {
            testRunning = false;
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-stop').disabled = true;
            document.getElementById('test-status').textContent = 'Complete';
            document.getElementById('test-status').style.color = '#f1c40f';
            if (testPollId) { clearInterval(testPollId); testPollId = null; }
        }
        if (d.stages) {
            var tbody = document.getElementById('results-body');
            while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
            var keys = Object.keys(STAGE_NAMES);
            for (var i = 0; i < keys.length; i++) {
                if (d.stages[keys[i]]) {
                    var s = d.stages[keys[i]];
                    var tr = document.createElement('tr');
                    [STAGE_NAMES[keys[i]], s.min.toFixed(1), s.avg.toFixed(1), s.max.toFixed(1), s.p95.toFixed(1)].forEach(function(v) {
                        var td = document.createElement('td');
                        td.textContent = v;
                        tr.appendChild(td);
                    });
                    tbody.appendChild(tr);
                }
            }
        }
    } catch(e) {}
}

// OCR Latency measurement
async function measureLatency() {
    var btn = document.getElementById('btn-measure');
    var st = document.getElementById('measure-status');
    var res = document.getElementById('measure-result');
    btn.disabled = true;
    st.textContent = 'Capturing + OCR...';
    st.style.color = '#f1c40f';
    res.textContent = '';
    try {
        var r = await fetch('/api/profiler/measure_latency', {method:'POST'});
        var d = await r.json();
        btn.disabled = false;
        if (d.error) { st.textContent = 'Error'; st.style.color = '#e74c3c'; res.textContent = d.error; return; }
        st.textContent = 'Done'; st.style.color = '#2ecc71';
        var lines = [];
        lines.push('Capture: ' + d.capture_ms.toFixed(1) + 'ms');
        lines.push('Inference: ' + d.inference_ms.toFixed(0) + 'ms');
        lines.push('Local UTC: ' + d.capture_utc);
        lines.push('Screen UTC: ' + (d.displayed_utc || 'not detected'));
        if (d.delta_ms !== null && d.delta_ms !== undefined) lines.push('Delta: ' + d.delta_ms + 'ms');
        if (d.ocr_result && d.ocr_result.notes) lines.push('Notes: ' + d.ocr_result.notes);
        res.textContent = lines.join('\n');
    } catch(e) { btn.disabled = false; st.textContent = 'Failed'; st.style.color = '#e74c3c'; }
}

poll();
