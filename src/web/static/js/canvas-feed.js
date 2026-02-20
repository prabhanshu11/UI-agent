/* canvas-feed.js — Raw frame fetch + dual-canvas rendering loop.
 *
 * Architecture:
 *   <div id="canvas-container">
 *     <canvas id="frame-canvas">   ← raw JPEG frame (bottom layer)
 *     <canvas id="overlay-canvas"> ← transparent overlay drawings (top layer)
 *   </div>
 *
 * The feed fetches /api/frame_raw at ~5Hz, draws onto frame-canvas,
 * then calls drawOverlays() from canvas-overlays.js on overlay-canvas.
 */

var CanvasFeed = (function() {
    var frameCanvas, overlayCanvas, frameCtx, overlayCtx;
    var containerEl;
    var feedRunning = false;
    var feedPaused = false;
    var seq = 0;
    var frameWidth = 1920;
    var frameHeight = 1080;
    var feedFps = 5;
    var lastState = null;

    function init(containerId) {
        containerEl = document.getElementById(containerId);
        if (!containerEl) return;

        // Create dual canvas stack
        frameCanvas = document.createElement('canvas');
        frameCanvas.id = 'frame-canvas';
        frameCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%';

        overlayCanvas = document.createElement('canvas');
        overlayCanvas.id = 'overlay-canvas';
        overlayCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none';

        containerEl.style.position = 'relative';
        containerEl.appendChild(frameCanvas);
        containerEl.appendChild(overlayCanvas);

        frameCtx = frameCanvas.getContext('2d');
        overlayCtx = overlayCanvas.getContext('2d');

        feedRunning = true;
        scheduleFrame();
    }

    function setCanvasSize(w, h) {
        frameWidth = w;
        frameHeight = h;
        frameCanvas.width = w;
        frameCanvas.height = h;
        overlayCanvas.width = w;
        overlayCanvas.height = h;
    }

    function scheduleFrame() {
        if (!feedRunning) return;
        if (!feedPaused) {
            fetchFrame();
        }
        setTimeout(scheduleFrame, 1000 / feedFps);
    }

    function fetchFrame() {
        var img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = function() {
            // Resize canvases to match frame dimensions (first frame or size change)
            if (img.naturalWidth !== frameWidth || img.naturalHeight !== frameHeight) {
                setCanvasSize(img.naturalWidth, img.naturalHeight);
            }
            frameCtx.drawImage(img, 0, 0);

            // Draw overlays if we have state
            if (lastState && typeof CanvasOverlays !== 'undefined') {
                overlayCtx.clearRect(0, 0, frameWidth, frameHeight);
                CanvasOverlays.draw(overlayCtx, lastState, frameWidth, frameHeight);
            }
        };
        img.src = '/api/frame_raw?' + seq++;
    }

    function updateState(state) {
        lastState = state;
    }

    function pause() { feedPaused = true; }
    function resume() { feedPaused = false; }
    function isPaused() { return feedPaused; }
    function stop() { feedRunning = false; }

    /** Get the composite canvas (frame + overlays) for MediaRecorder. */
    function getCompositeCanvas() {
        var composite = document.createElement('canvas');
        composite.width = frameWidth;
        composite.height = frameHeight;
        var ctx = composite.getContext('2d');
        ctx.drawImage(frameCanvas, 0, 0);
        ctx.drawImage(overlayCanvas, 0, 0);
        return composite;
    }

    function getFrameCanvas() { return frameCanvas; }
    function getOverlayCanvas() { return overlayCanvas; }
    function getFrameSize() { return { w: frameWidth, h: frameHeight }; }

    return {
        init: init,
        updateState: updateState,
        pause: pause,
        resume: resume,
        isPaused: isPaused,
        stop: stop,
        getCompositeCanvas: getCompositeCanvas,
        getFrameCanvas: getFrameCanvas,
        getOverlayCanvas: getOverlayCanvas,
        getFrameSize: getFrameSize,
    };
})();
