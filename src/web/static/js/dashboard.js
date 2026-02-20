/* dashboard.js — Sidebar state polling, tag system, controls. */

var paused = false;
var frozenState = null;
var tagCount = 0;
var lastState = null;
var tagState = {};

// ── Initialize ───────────────────────────────────────────────
function initDashboard() {
    // Init canvas feed
    CanvasFeed.init('canvas-container');

    // Build overlay toggle panel
    buildOverlayToggles();

    // Wire tag buttons
    document.querySelectorAll('.tag-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            if (!paused) togglePause();
            var dim = this.dataset.dim;
            var val = this.dataset.val;
            this.parentElement.querySelectorAll('.tag-btn').forEach(function(b) {
                b.classList.remove('selected');
            });
            this.classList.add('selected');
            tagState[dim] = val;
        });
    });

    // Start polling
    scheduleStateUpdate();
}

// ── Overlay toggles ──────────────────────────────────────────
function buildOverlayToggles() {
    var panel = document.getElementById('overlay-toggles');
    if (!panel) return;
    var names = {
        motionBlobs: 'Motion Blobs',
        yoloBbox: 'YOLO Box',
        silhouetteRoi: 'Sil. ROI',
        cursorCrosshair: 'Crosshair',
        trackingHud: 'HUD Text',
        visionDot: 'Vision Dot',
    };
    for (var key in names) {
        var btn = document.createElement('button');
        btn.className = 'overlay-toggle' + (CanvasOverlays.getOverlay(key) ? ' active' : '');
        btn.textContent = names[key];
        btn.setAttribute('data-overlay', key);
        btn.addEventListener('click', function() {
            var k = this.getAttribute('data-overlay');
            var newVal = !CanvasOverlays.getOverlay(k);
            CanvasOverlays.setOverlay(k, newVal);
            this.classList.toggle('active', newVal);
        });
        panel.appendChild(btn);
    }
}

// ── State polling (200ms = 5Hz) ──────────────────────────────
function scheduleStateUpdate() {
    if (!paused) updateState();
    setTimeout(scheduleStateUpdate, 200);
}

async function updateState() {
    try {
        var res = await fetch('/api/state');
        var s = await res.json();
        lastState = s;

        // Feed the canvas overlay system
        CanvasFeed.updateState(s);

        document.getElementById('fps-display').textContent = s.fps + ' FPS';

        setDot('dot-mouse', s.hardware.mouse_connected ? 'green' : 'red');
        setDot('dot-heartbeat', s.hardware.heartbeat_active ? 'green' : 'gray');
        setDot('dot-jitter', s.hardware.anti_sleep_active ? 'green' : 'gray');
        setDot('dot-keyboard',
            s.pi_keyboard === true ? 'green' :
            s.pi_keyboard === false ? 'yellow' : 'gray');

        var coordsEl = document.getElementById('cursor-coords');
        coordsEl.textContent = '(' + s.cursor.x + ', ' + s.cursor.y + ')';
        coordsEl.style.color = s.cursor.quality === 'good' ? '#2ecc71' :
            s.cursor.quality === 'stale' ? '#f1c40f' : '#e74c3c';
        document.getElementById('cursor-method').textContent =
            s.cursor.method + ' [' + s.cursor.quality + ']';
        document.getElementById('cursor-age').textContent = s.cursor.age_s + 's';

        var m = s.cursor.method;
        document.getElementById('tier-1').classList.toggle('active',
            m === 'micro_shake' || m === 'motion_track' || m === 'shape_track');
        document.getElementById('tier-2').classList.toggle('active',
            m === 'macro_shake' || m === 'cnn_reacquire');
        document.getElementById('tier-3').classList.toggle('active', m === 'lissajous');

        document.getElementById('blob-count').textContent = s.blob_count;
        var tbody = document.getElementById('blob-list');
        while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
        if (s.blobs.length > 0) {
            s.blobs.slice(0, 8).forEach(function(b, i) {
                var tr = document.createElement('tr');
                if (i === 0) tr.style.color = '#2ecc71';
                var td1 = document.createElement('td');
                td1.textContent = '(' + b.centroid[0] + ', ' + b.centroid[1] + ')';
                var td2 = document.createElement('td');
                td2.textContent = b.pixel_count + 'px';
                var td3 = document.createElement('td');
                td3.textContent = Math.round(b.angle * 180 / Math.PI) + '\u00B0';
                tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3);
                tbody.appendChild(tr);
            });
        } else {
            var tr = document.createElement('tr');
            var td = document.createElement('td');
            td.colSpan = 3; td.style.color = '#444'; td.textContent = 'No motion';
            tr.appendChild(td); tbody.appendChild(tr);
        }

        // Jitter panel
        var jitterStatus = document.getElementById('jitter-status');
        var jitterBar = document.getElementById('jitter-bar');
        var jitterDesc = document.getElementById('jitter-desc');
        if (s.jitter && s.jitter.reacquire_pending) {
            jitterStatus.textContent = 'Re-acquiring';
            jitterStatus.className = 'hw-value warn';
            jitterBar.classList.add('active');
            jitterBar.style.background = '#e67e22';
            jitterDesc.textContent = 'Low confidence ' +
                s.jitter.low_confidence_s.toFixed(0) + 's \u2014 next jitter will probe';
            jitterDesc.style.color = '#e67e22';
        } else if (s.hardware.anti_sleep_active) {
            jitterStatus.textContent = 'Active';
            jitterStatus.className = 'hw-value ok';
            jitterBar.classList.add('active');
            jitterBar.style.background = '';
            jitterDesc.textContent = '\u00B13px every 30s \u2014 prevents Windows sleep';
            jitterDesc.style.color = '#444';
        } else {
            jitterStatus.textContent = 'Inactive';
            jitterStatus.className = 'hw-value warn';
            jitterBar.classList.remove('active');
            jitterDesc.style.color = '#444';
        }

        setHW('hw-mouse', s.hardware.mouse_connected, 'Connected', 'Disconnected');
        setHW('hw-heartbeat', s.hardware.heartbeat_active, 'Active (100ms)', 'Inactive');
        setHW('hw-keyboard',
            s.pi_keyboard === true, 'BT Connected',
            s.pi_keyboard === false ? 'Discoverable' : 'Unreachable');
        document.getElementById('hw-frames').textContent = s.frame_count;

        // Passthrough live stats
        var ptStats = document.getElementById('pt-stats');
        if (s.passthrough && s.passthrough.active && s.passthrough.session) {
            var ps = s.passthrough.session;
            ptStats.style.display = 'block';
            ptStats.textContent = '\u25CF REC ' + ps.label + ' | ' +
                ps.duration_s + 's | ' + ps.events + ' events | ' +
                ps.screenshots + ' shots | ' + ps.cnn_samples + ' CNN';
            ptStats.style.color = '#e74c3c';
        } else {
            ptStats.style.display = 'none';
        }

        // YOLO panel
        if (s.yolo) {
            var yoloStatus = document.getElementById('yolo-status');
            yoloStatus.textContent = s.yolo.active ? 'Detecting' : 'No detection';
            yoloStatus.className = 'hw-value ' + (s.yolo.active ? 'ok' : 'warn');
            document.getElementById('yolo-position').textContent =
                s.yolo.active ? '(' + s.yolo.x + ', ' + s.yolo.y + ')' : '\u2014';
            var yoloConf = document.getElementById('yolo-confidence');
            yoloConf.textContent = s.yolo.active ? (s.yolo.confidence * 100).toFixed(1) + '%' : '\u2014';
            yoloConf.className = 'hw-value ' + (s.yolo.confidence > 0.7 ? 'ok' :
                s.yolo.confidence > 0.4 ? 'warn' : 'err');
            document.getElementById('yolo-inference').textContent =
                s.yolo.inference_ms.toFixed(1) + 'ms';
            document.getElementById('yolo-count').textContent = s.yolo.detection_count;
        }

        // Silhouette panel
        if (s.silhouette) {
            var silStatus = document.getElementById('sil-status');
            silStatus.textContent = s.silhouette.active ? 'Active' : 'Inactive';
            silStatus.className = 'hw-value ' + (s.silhouette.active ? 'ok' : 'warn');
            var silMethod = document.getElementById('sil-method');
            silMethod.textContent = s.silhouette.method;
            silMethod.style.color = s.silhouette.method === 'template' ? '#2ecc71' :
                s.silhouette.method === 'motion_roi' ? '#3498db' :
                s.silhouette.method === 'lost' ? '#e74c3c' : '#888';
            document.getElementById('sil-hz').textContent = s.silhouette.hz + ' Hz';
            var roiEl = document.getElementById('sil-roi');
            roiEl.textContent = s.silhouette.roi_size + 'px';
            roiEl.style.color = s.silhouette.roi_size <= 200 ? '#2ecc71' :
                s.silhouette.roi_size <= 400 ? '#f1c40f' : '#e74c3c';
            var silConf = document.getElementById('sil-confidence');
            silConf.textContent = (s.silhouette.confidence * 100).toFixed(1) + '%';
            silConf.className = 'hw-value ' + (s.silhouette.confidence > 0.5 ? 'ok' :
                s.silhouette.confidence > 0.3 ? 'warn' : 'err');
            document.getElementById('sil-latency').textContent =
                s.silhouette.latency_ms.toFixed(2) + 'ms';
            var trusted = s.cnn.cursor_validated ||
                s.cnn.confidence > 0.8 ||
                s.silhouette.method === 'motion_roi' ||
                s.silhouette.method === 'jitter_confirm' ||
                (s.yolo && s.yolo.active);
            var trustEl = document.getElementById('sil-trusted');
            trustEl.textContent = trusted ? 'Yes' : 'Gated';
            trustEl.className = 'hw-value ' + (trusted ? 'ok' : 'warn');
        }

        // CNN panel
        if (s.cnn) {
            var conf = s.cnn.confidence;
            var confEl = document.getElementById('cnn-confidence');
            confEl.textContent = (conf * 100).toFixed(1) + '%';
            confEl.className = 'hw-value ' + (conf > 0.7 ? 'ok' : conf > 0.3 ? 'warn' : 'err');
            var bar = document.getElementById('cnn-bar');
            bar.style.width = (conf * 100) + '%';
            bar.style.background = conf > 0.7 ? '#2ecc71' : conf > 0.3 ? '#f1c40f' : '#e74c3c';
            document.getElementById('cnn-model').textContent = s.cnn.model_version;
            var valEl = document.getElementById('cnn-validated');
            valEl.textContent = s.cnn.cursor_validated ? 'Yes' : 'No';
            valEl.className = 'hw-value ' + (s.cnn.cursor_validated ? 'ok' : 'warn');
            document.getElementById('cnn-inference').textContent = s.cnn.inference_ms + 'ms';
            document.getElementById('cnn-samples').textContent =
                s.cnn.samples.pos + ' pos / ' + s.cnn.samples.neg + ' neg';
        }

        // Claude Vision panel
        if (s.vision) {
            var v = s.vision;
            if (v.age_s >= 0) {
                document.getElementById('vision-pos').textContent =
                    '(' + v.x + ', ' + v.y + ')';
                var vconf = document.getElementById('vision-confidence');
                vconf.textContent = v.confidence;
                vconf.className = 'hw-value ' + (
                    v.confidence === 'high' ? 'ok' :
                    v.confidence === 'medium' ? 'warn' : 'err');
                document.getElementById('vision-type').textContent = v.cursor_type;
                document.getElementById('vision-latency').textContent =
                    (v.latency_ms / 1000).toFixed(1) + 's';
                var ageStr = v.age_s < 60 ? v.age_s.toFixed(0) + 's ago' :
                    (v.age_s / 60).toFixed(1) + 'm ago';
                document.getElementById('vision-age').textContent = ageStr;
                var dx = Math.abs(s.cursor.x - v.x);
                var dy = Math.abs(s.cursor.y - v.y);
                var delta = Math.round(Math.sqrt(dx*dx + dy*dy));
                var deltaEl = document.getElementById('vision-delta');
                deltaEl.textContent = delta + 'px';
                deltaEl.className = 'hw-value ' + (
                    delta < 30 ? 'ok' : delta < 100 ? 'warn' : 'err');
            }
        }

        var errPanel = document.getElementById('error-panel');
        if (s.error) {
            errPanel.style.display = 'block';
            document.getElementById('error-text').textContent = s.error;
        } else {
            errPanel.style.display = 'none';
        }
    } catch (e) {
        console.error('State update failed:', e);
    }
}

// ── Pause / freeze for tagging ───────────────────────────────
function togglePause() {
    paused = !paused;
    var btn = document.getElementById('pause-btn');
    var overlay = document.getElementById('pause-overlay');
    if (paused) {
        btn.textContent = 'RESUME (LIVE)';
        btn.style.background = '#e74c3c';
        overlay.style.display = 'flex';
        frozenState = lastState;
        CanvasFeed.pause();
    } else {
        btn.textContent = 'PAUSE (TAG)';
        btn.style.background = '#f39c12';
        overlay.style.display = 'none';
        frozenState = null;
        CanvasFeed.resume();
        document.querySelectorAll('.tag-btn').forEach(function(b) {
            b.classList.remove('selected');
        });
        tagState = {};
        document.getElementById('tag-notes').value = '';
        document.getElementById('tag-status').textContent = '';
    }
}

// ── YOLO manual detect ──────────────────────────────────────
function runYoloDetect() {
    var btn = document.getElementById('yolo-btn');
    btn.textContent = 'Detecting...';
    btn.disabled = true;
    fetch('/api/yolo_detect_now')
        .then(function(r) { return r.json(); })
        .then(function(d) {
            btn.textContent = d.active ? 'Detected!' : 'No detection';
            btn.disabled = false;
            setTimeout(function() { btn.textContent = 'Run Detection'; }, 2000);
        })
        .catch(function() {
            btn.textContent = 'Run Detection';
            btn.disabled = false;
        });
}

// ── Claude Vision detect ─────────────────────────────────────
function runVisionDetect() {
    var btn = document.getElementById('vision-btn');
    btn.textContent = 'Detecting...';
    btn.disabled = true;
    fetch('/api/claude_detect')
        .then(function(r) { return r.json(); })
        .then(function(d) {
            btn.textContent = 'Run Detection';
            btn.disabled = false;
        })
        .catch(function() {
            btn.textContent = 'Run Detection';
            btn.disabled = false;
        });
}

// ── Tagging ──────────────────────────────────────────────────
async function submitTag() {
    var notes = document.getElementById('tag-notes').value;
    tagState.notes = notes;
    var statusEl = document.getElementById('tag-status');
    statusEl.textContent = 'Saving...';
    try {
        var res = await fetch('/api/tag', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(tagState)
        });
        var data = await res.json();
        if (data.status === 'tagged') {
            tagCount++;
            statusEl.textContent = 'Tagged #' + tagCount + ' (' + data.frame + ')';
            statusEl.style.color = '#27ae60';
            setTimeout(function() { togglePause(); }, 800);
        } else {
            statusEl.textContent = 'Error: ' + (data.error || 'unknown');
            statusEl.style.color = '#e74c3c';
        }
    } catch(e) {
        statusEl.textContent = 'Error: ' + e.message;
        statusEl.style.color = '#e74c3c';
    }
}

// ── Mouse Passthrough ────────────────────────────────────────
var passthroughActive = false;
var ptBatchDx = 0;
var ptBatchDy = 0;
var ptFlushInterval = null;
var ptSensitivity = 1.0;

function _ptToggleFetch() {
    var label = document.getElementById('pt-label').value;
    return fetch('/api/passthrough/toggle', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({label: label})
    }).then(function(r) { return r.json(); });
}

function togglePassthrough() {
    _ptToggleFetch().then(function(d) {
        passthroughActive = d.active;
        updatePassthroughUI();
        if (!passthroughActive) exitPassthrough();
    });
}

function enterPassthrough() {
    if (!passthroughActive) {
        _ptToggleFetch().then(function(d) {
            passthroughActive = d.active;
            updatePassthroughUI();
            if (passthroughActive) requestPointerLock();
        });
    } else {
        requestPointerLock();
    }
}

function requestPointerLock() {
    var zone = document.getElementById('passthrough-zone');
    zone.requestPointerLock = zone.requestPointerLock || zone.mozRequestPointerLock;
    zone.requestPointerLock();
}

function exitPassthrough() {
    if (document.pointerLockElement) {
        document.exitPointerLock();
    }
    if (ptFlushInterval) {
        clearInterval(ptFlushInterval);
        ptFlushInterval = null;
    }
    flushPassthroughBatch();
}

function setZoneContent(zone, captured) {
    while (zone.firstChild) zone.removeChild(zone.firstChild);
    var main = document.createElement('span');
    if (captured) {
        main.style.color = '#e74c3c';
        main.style.fontWeight = 'bold';
        main.textContent = 'MOUSE CAPTURED';
    } else {
        main.textContent = 'Click here to control remote mouse';
    }
    zone.appendChild(main);
    zone.appendChild(document.createElement('br'));
    var sub = document.createElement('span');
    sub.style.fontSize = '0.6rem';
    sub.style.color = captured ? '#888' : '#444';
    sub.textContent = 'Press Esc to release';
    zone.appendChild(sub);
}

function onPointerLockChange() {
    var zone = document.getElementById('passthrough-zone');
    var statusEl = document.getElementById('passthrough-status');
    if (document.pointerLockElement === zone) {
        document.addEventListener('mousemove', onPassthroughMove);
        document.addEventListener('mousedown', onPassthroughClick);
        document.addEventListener('wheel', onPassthroughScroll, {passive: false});
        document.addEventListener('keydown', onPassthroughKey);
        ptBatchDx = 0;
        ptBatchDy = 0;
        ptFlushInterval = setInterval(flushPassthroughBatch, 20);
        zone.style.borderColor = '#e74c3c';
        zone.style.background = '#2a1015';
        setZoneContent(zone, true);
        statusEl.textContent = 'Pointer locked \u2014 input piped to remote';
        statusEl.style.color = '#e74c3c';
        // Auto-start browser recording if Recorder is available
        if (typeof Recorder !== 'undefined' && !Recorder.isRecording()) {
            try { Recorder.start(); } catch(e) {}
        }
    } else {
        document.removeEventListener('mousemove', onPassthroughMove);
        document.removeEventListener('mousedown', onPassthroughClick);
        document.removeEventListener('wheel', onPassthroughScroll);
        document.removeEventListener('keydown', onPassthroughKey);
        if (ptFlushInterval) {
            clearInterval(ptFlushInterval);
            ptFlushInterval = null;
        }
        flushPassthroughBatch();
        zone.style.borderColor = '#444';
        zone.style.background = '#1a1a25';
        setZoneContent(zone, false);
        statusEl.textContent = 'Released';
        statusEl.style.color = '#555';
        // Auto-stop browser recording
        if (typeof Recorder !== 'undefined' && Recorder.isRecording()) {
            try { Recorder.stop(); } catch(e) {}
        }
    }
}

function onPassthroughMove(e) {
    ptBatchDx += Math.round(e.movementX * ptSensitivity);
    ptBatchDy += Math.round(e.movementY * ptSensitivity);
}

function onPassthroughClick(e) {
    if (!passthroughActive) return;
    var btn = e.button === 2 ? 'right' : e.button === 1 ? 'middle' : 'left';
    fetch('/api/passthrough/click', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({button: btn})
    });
    e.preventDefault();
}

function onPassthroughScroll(e) {
    if (!passthroughActive) return;
    // Normalize deltaY to discrete scroll clicks (most mice: 100-120px per notch)
    var clicks = Math.round(e.deltaY / 100) || (e.deltaY > 0 ? 1 : -1);
    fetch('/api/passthrough/scroll', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({clicks: clicks})
    });
    e.preventDefault();
}

var _ptKeyMap = {'Enter': '\n', 'Tab': '\t', 'Backspace': '\b'};

function onPassthroughKey(e) {
    if (!passthroughActive) return;
    // Escape is reserved for releasing pointer lock
    if (e.key === 'Escape') return;
    var text = _ptKeyMap[e.key] || (e.key.length === 1 ? e.key : null);
    if (!text) return;
    fetch('/api/passthrough/key', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
    });
    e.preventDefault();
}

function flushPassthroughBatch() {
    if (ptBatchDx === 0 && ptBatchDy === 0) return;
    var dx = ptBatchDx;
    var dy = ptBatchDy;
    ptBatchDx = 0;
    ptBatchDy = 0;
    fetch('/api/passthrough/move', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({dx: dx, dy: dy})
    });
}

function updatePassthroughUI() {
    var btn = document.getElementById('passthrough-btn');
    if (passthroughActive) {
        btn.textContent = 'DEACTIVATE';
        btn.style.background = '#c0392b';
    } else {
        btn.textContent = 'ACTIVATE';
        btn.style.background = '#e74c3c';
    }
}

document.addEventListener('pointerlockchange', onPointerLockChange);
document.addEventListener('mozpointerlockchange', onPointerLockChange);

document.addEventListener('contextmenu', function(e) {
    if (document.pointerLockElement) e.preventDefault();
});

// ── Init on DOM ready ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', initDashboard);
