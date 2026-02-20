/* canvas-overlays.js — All overlay drawing (ported from Python cv2 to JS canvas).
 *
 * Each overlay can be toggled independently via localStorage.
 * Overlay state is read from /api/state JSON.
 */

var CanvasOverlays = (function() {
    // Default overlay visibility (persisted to localStorage)
    var overlayDefaults = {
        motionBlobs: true,
        yoloBbox: true,
        silhouetteRoi: true,
        cursorCrosshair: true,
        trackingHud: true,
        visionDot: true,
    };

    var overlays = {};

    function loadToggles() {
        var saved = localStorage.getItem('kvm_overlays');
        if (saved) {
            try { overlays = JSON.parse(saved); } catch(e) {}
        }
        // Fill in defaults for any missing keys
        for (var k in overlayDefaults) {
            if (overlays[k] === undefined) overlays[k] = overlayDefaults[k];
        }
    }

    function saveToggles() {
        localStorage.setItem('kvm_overlays', JSON.stringify(overlays));
    }

    function setOverlay(name, enabled) {
        overlays[name] = enabled;
        saveToggles();
    }

    function getOverlay(name) {
        return overlays[name] !== false;
    }

    function getAll() { return overlays; }

    loadToggles();

    /** Main draw entry — called each frame with overlay canvas ctx + state. */
    function draw(ctx, state, w, h) {
        if (overlays.silhouetteRoi) drawSilhouetteRoi(ctx, state);
        if (overlays.motionBlobs)   drawMotionBlobs(ctx, state);
        if (overlays.yoloBbox)      drawYoloBbox(ctx, state);
        if (overlays.cursorCrosshair) drawCursorCrosshair(ctx, state, w, h);
        if (overlays.trackingHud)   drawTrackingHud(ctx, state, w, h);
        if (overlays.visionDot)     drawVisionDot(ctx, state);
    }

    /** Green/blue motion blob rectangles + centroids. */
    function drawMotionBlobs(ctx, state) {
        var blobs = state.blobs || [];
        for (var i = 0; i < blobs.length; i++) {
            var b = blobs[i];
            var bbox = b.bbox;
            var cx = b.centroid[0], cy = b.centroid[1];
            var isPointer = (i === 0);

            ctx.strokeStyle = isPointer ? '#00ff00' : 'rgba(80, 160, 255, 0.8)';
            ctx.lineWidth = isPointer ? 2 : 1;
            ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);

            // Centroid dot
            ctx.fillStyle = ctx.strokeStyle;
            ctx.beginPath();
            ctx.arc(cx, cy, 4, 0, Math.PI * 2);
            ctx.fill();

            // Pixel count label
            ctx.font = '11px JetBrains Mono, monospace';
            ctx.fillText(b.pixel_count + 'px', bbox[0], bbox[1] - 5);
        }
    }

    /** Cyan YOLO bounding box with confidence label. */
    function drawYoloBbox(ctx, state) {
        if (!state.yolo || !state.yolo.active) return;
        var bbox = state.yolo.bbox;
        if (!bbox || bbox.length < 4) return;
        var conf = state.yolo.confidence;
        if (conf < 0.25) return;

        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);

        ctx.fillStyle = '#00ffff';
        ctx.font = '11px JetBrains Mono, monospace';
        ctx.fillText('YOLO ' + Math.round(conf * 100) + '%', bbox[0], bbox[1] - 5);
    }

    /** Yellow dashed ROI rectangle for silhouette tracking. */
    function drawSilhouetteRoi(ctx, state) {
        if (!state.silhouette || !state.silhouette.active) return;
        // Compute ROI bounds from cursor position + roi_size
        var cx = state.cursor.x, cy = state.cursor.y;
        var half = Math.floor(state.silhouette.roi_size / 2);
        var rx0 = cx - half, ry0 = cy - half;
        var rx1 = cx + half, ry1 = cy + half;

        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 1;
        ctx.setLineDash([10, 10]);
        ctx.strokeRect(rx0, ry0, rx1 - rx0, ry1 - ry0);
        ctx.setLineDash([]);
    }

    /** Green/yellow/red crosshair at cursor position. */
    function drawCursorCrosshair(ctx, state, w, h) {
        var cx = state.cursor.x, cy = state.cursor.y;
        if (cx <= 0 && cy <= 0) return;

        var age = state.cursor.age_s;
        var color = age < 5 ? '#00cc00' : age < 30 ? '#ffcc00' : '#ff3333';
        var size = 20;

        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        // Horizontal line
        ctx.beginPath();
        ctx.moveTo(cx - size, cy);
        ctx.lineTo(cx + size, cy);
        ctx.stroke();
        // Vertical line
        ctx.beginPath();
        ctx.moveTo(cx, cy - size);
        ctx.lineTo(cx, cy + size);
        ctx.stroke();
    }

    /** TRACKING/STALE/LOST text in bottom-left. */
    function drawTrackingHud(ctx, state, w, h) {
        var age = state.cursor.age_s;
        var cx = state.cursor.x, cy = state.cursor.y;
        var color, label;

        if (age < 5) {
            color = '#00cc00';
            label = 'TRACKING (' + cx + ',' + cy + ')';
        } else if (age < 30) {
            color = '#ffcc00';
            label = 'STALE ' + Math.round(age) + 's (' + cx + ',' + cy + ')';
        } else {
            color = '#ff3333';
            label = 'LOST ' + Math.round(age) + 's';
        }

        ctx.font = '14px JetBrains Mono, monospace';
        // Shadow
        ctx.fillStyle = '#000';
        ctx.fillText(label, 12, h - 10);
        ctx.fillText(label, 10, h - 12);
        // Text
        ctx.fillStyle = color;
        ctx.fillText(label, 10, h - 12);
    }

    /** Purple dot for Claude Vision position (fades with age). */
    function drawVisionDot(ctx, state) {
        if (!state.vision || state.vision.age_s < 0) return;
        var vx = state.vision.x, vy = state.vision.y;
        if (vx <= 0 || vy <= 0) return;

        var age = state.vision.age_s;
        var opacity = age < 30 ? 1.0 : age < 120 ? 1.0 - (age - 30) / 90 : 0;
        if (opacity <= 0.05) return;

        ctx.globalAlpha = opacity;
        ctx.fillStyle = '#a855f7';
        ctx.beginPath();
        ctx.arc(vx, vy, 9, 0, Math.PI * 2);
        ctx.fill();

        // Bright center
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(vx, vy, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1.0;
    }

    return {
        draw: draw,
        setOverlay: setOverlay,
        getOverlay: getOverlay,
        getAll: getAll,
        loadToggles: loadToggles,
    };
})();
