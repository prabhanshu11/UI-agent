/* recorder.js — MediaRecorder wrapper for canvas stream recording.
 *
 * Records the composite view (frame + overlays) as WebM VP9 at 5fps.
 * The recording captures exactly what the user sees, including active overlay toggles.
 */

var Recorder = (function() {
    var mediaRecorder = null;
    var chunks = [];
    var recording = false;
    var startTime = 0;
    var timerInterval = null;
    var compositeCanvas = null;
    var compositeCtx = null;
    var drawInterval = null;

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
        chunks = [];

        try {
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp9',
                videoBitsPerSecond: 2000000,
            });
        } catch (e) {
            // Fallback to default codec
            try {
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'video/webm',
                    videoBitsPerSecond: 2000000,
                });
            } catch (e2) {
                updateStatus('MediaRecorder not supported');
                return;
            }
        }

        mediaRecorder.ondataavailable = function(e) {
            if (e.data.size > 0) chunks.push(e.data);
        };

        mediaRecorder.onstop = function() {
            var blob = new Blob(chunks, { type: 'video/webm' });
            downloadBlob(blob);
            cleanup();
        };

        mediaRecorder.start(1000); // Collect data every second
        recording = true;
        startTime = Date.now();

        // UI
        var btn = document.getElementById('rec-btn');
        if (btn) {
            btn.textContent = 'STOP';
            btn.classList.add('recording');
        }
        updateStatus('Recording...');

        timerInterval = setInterval(updateTimer, 100);
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
        updateStatus('');
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

    function downloadBlob(blob) {
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        var ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        a.href = url;
        a.download = 'kvm-recording-' + ts + '.webm';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        updateStatus('Downloaded');
        setTimeout(function() { updateStatus(''); }, 3000);
    }

    return {
        toggle: toggle,
        start: start,
        stop: stop,
        isRecording: function() { return recording; },
    };
})();
