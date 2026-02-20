/* recorder.js — MediaRecorder wrapper for canvas stream recording.
 *
 * Records the composite view (frame + overlays) as WebM VP9 at 5fps.
 * Chunks are uploaded to the server via /api/recording/chunk (no browser download).
 * The recording captures exactly what the user sees, including active overlay toggles.
 */

var Recorder = (function() {
    var mediaRecorder = null;
    var recording = false;
    var startTime = 0;
    var timerInterval = null;
    var compositeCanvas = null;
    var compositeCtx = null;
    var drawInterval = null;
    var recordingId = null;

    function toggle() {
        if (recording) {
            stop();
        } else {
            start();
        }
    }

    function start() {
        if (recording) return;

        var frameCanvas = CanvasFeed.getFrameCanvas();
        var overlayCanvas = CanvasFeed.getOverlayCanvas();
        if (!frameCanvas || !overlayCanvas) {
            updateStatus('No canvas available');
            return;
        }

        var size = CanvasFeed.getFrameSize();

        // Create a composite canvas that merges both layers
        compositeCanvas = document.createElement('canvas');
        compositeCanvas.width = size.w;
        compositeCanvas.height = size.h;
        compositeCtx = compositeCanvas.getContext('2d');

        // Draw both layers onto composite at 5fps
        drawInterval = setInterval(function() {
            compositeCtx.drawImage(frameCanvas, 0, 0);
            compositeCtx.drawImage(overlayCanvas, 0, 0);
        }, 200);

        // Capture stream from composite canvas
        var stream = compositeCanvas.captureStream(5);

        var codec = 'video/webm;codecs=vp9';
        try {
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: codec,
                videoBitsPerSecond: 2000000,
            });
        } catch (e) {
            codec = 'video/webm';
            try {
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: codec,
                    videoBitsPerSecond: 2000000,
                });
            } catch (e2) {
                updateStatus('MediaRecorder not supported');
                return;
            }
        }

        updateStatus('Starting server recording...');

        // Start server-side recording, then hook up MediaRecorder
        fetch('/api/recording/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                codec: codec,
                resolution: size.w + 'x' + size.h,
                fps: 5,
                overlays: typeof CanvasOverlays !== 'undefined' ?
                    CanvasOverlays.getAll() : {},
            })
        })
        .then(function(r) { return r.json(); })
        .then(function(d) {
            recordingId = d.recording_id;

            // Stream each chunk to server
            mediaRecorder.ondataavailable = function(e) {
                if (e.data.size > 0 && recordingId) {
                    fetch('/api/recording/chunk?id=' +
                        encodeURIComponent(recordingId), {
                        method: 'POST',
                        body: e.data,
                    });
                }
            };

            // On stop, finalize server-side
            mediaRecorder.onstop = function() {
                if (recordingId) {
                    fetch('/api/recording/stop?id=' +
                        encodeURIComponent(recordingId), {
                        method: 'POST',
                    })
                    .then(function(r) { return r.json(); })
                    .then(function(d) {
                        if (d.ok) {
                            updateStatus('Saved: ' + d.filename +
                                ' (' + (d.bytes / 1024 / 1024).toFixed(1) + 'MB)');
                        } else {
                            updateStatus('Error: ' + (d.error || 'unknown'));
                        }
                        setTimeout(function() { updateStatus(''); }, 5000);
                    })
                    .catch(function() {
                        updateStatus('Upload error');
                        setTimeout(function() { updateStatus(''); }, 3000);
                    });
                }
                recordingId = null;
                cleanup();
            };

            mediaRecorder.start(1000);
            recording = true;
            startTime = Date.now();

            var btn = document.getElementById('rec-btn');
            if (btn) {
                btn.textContent = 'STOP';
                btn.classList.add('recording');
            }
            updateStatus('Recording (server)...');
            timerInterval = setInterval(updateTimer, 100);
        })
        .catch(function(err) {
            updateStatus('Server error: ' + err.message);
            cleanup();
        });
    }

    function stop() {
        if (!recording || !mediaRecorder) return;
        mediaRecorder.stop();
        recording = false;

        var btn = document.getElementById('rec-btn');
        if (btn) {
            btn.textContent = 'REC';
            btn.classList.remove('recording');
        }
        updateStatus('Saving...');
    }

    function cleanup() {
        if (drawInterval) { clearInterval(drawInterval); drawInterval = null; }
        if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
        compositeCanvas = null;
        compositeCtx = null;
        var dur = document.getElementById('rec-duration');
        if (dur) dur.textContent = '';
    }

    function updateTimer() {
        if (!recording) return;
        var elapsed = Math.floor((Date.now() - startTime) / 1000);
        var min = Math.floor(elapsed / 60);
        var sec = elapsed % 60;
        var dur = document.getElementById('rec-duration');
        if (dur) dur.textContent = min + ':' + (sec < 10 ? '0' : '') + sec;
    }

    function updateStatus(msg) {
        var el = document.getElementById('rec-status');
        if (el) el.textContent = msg;
    }

    return {
        toggle: toggle,
        start: start,
        stop: stop,
        isRecording: function() { return recording; },
    };
})();
