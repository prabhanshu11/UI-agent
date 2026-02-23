/* sub-panels.js — Three sub-panel canvases below the main HDMI feed.
 *
 * 1. Motion Blob Map: minimap of detected blobs (green primary, blue secondary)
 * 2. Model Inference: reduced HDMI frame + YOLO/CNN/Vision/Silhouette overlays
 * 3. Mouse Trails: detected (cyan) vs ESP32 (green) vs passthrough (orange)
 *
 * Hooks into the existing 5Hz state poll via SubPanels.update(state).
 */

var SubPanels = (function() {
    // ── Canvas refs ──────────────────────────────────────────
    var blobCanvas, blobCtx;
    var inferCanvas, inferCtx;
    var trailsCanvas, trailsCtx;
    var SCREEN_W = 1920, SCREEN_H = 1080;

    // ── Mouse trail ring buffers ─────────────────────────────
    var TRAIL_SIZE = 100;  // 20 seconds at 5Hz
    var detectedTrail = [];
    var esp32Trail = [];
    var passthroughTrail = [];

    // ── Passthrough tracking ─────────────────────────────────
    var ptAccumX = 0, ptAccumY = 0;
    var ptActive = false;

    // ── Model inference state ────────────────────────────────
    var lastFrameImg = null;
    var frameSeq = 0;

    // ── Motion trail API data ─────────────────────────────────
    var lastTrailData = null;

    function init() {
        blobCanvas = document.getElementById('blob-map-canvas');
        inferCanvas = document.getElementById('inference-canvas');
        trailsCanvas = document.getElementById('trails-canvas');
        if (!blobCanvas || !inferCanvas || !trailsCanvas) return;

        blobCtx = blobCanvas.getContext('2d');
        inferCtx = inferCanvas.getContext('2d');
        trailsCtx = trailsCanvas.getContext('2d');

        sizeCanvases();
        window.addEventListener('resize', sizeCanvases);

        // Fetch reduced frame for inference panel at 1Hz
        setInterval(fetchInferenceFrame, 1000);
    }

    function sizeCanvases() {
        var canvases = [blobCanvas, inferCanvas, trailsCanvas];
        for (var i = 0; i < canvases.length; i++) {
            var c = canvases[i];
            if (c && c.parentElement) {
                var rect = c.parentElement.getBoundingClientRect();
                c.width = Math.floor(rect.width);
                c.height = Math.floor(rect.height);
            }
        }
    }

    function fetchInferenceFrame() {
        var img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = function() { lastFrameImg = img; };
        img.src = '/api/frame_raw?sub=' + (frameSeq++);
    }

    // ── Trail helpers ────────────────────────────────────────
    function pushTrail(trail, x, y) {
        trail.push({x: x, y: y});
        if (trail.length > TRAIL_SIZE) trail.shift();
    }

    // ── Passthrough delta accumulation (called from dashboard.js at 50Hz) ──
    function recordPassthroughDelta(dx, dy) {
        ptAccumX += dx;
        ptAccumY += dy;
        // Clamp to screen bounds
        ptAccumX = Math.max(0, Math.min(SCREEN_W, ptAccumX));
        ptAccumY = Math.max(0, Math.min(SCREEN_H, ptAccumY));
    }

    // ── Main update (called from dashboard.js at 5Hz) ────────
    function update(state) {
        if (!blobCtx || !inferCtx || !trailsCtx) return;

        // Accumulate trail points
        if (state.cursor && (state.cursor.x > 0 || state.cursor.y > 0)) {
            pushTrail(detectedTrail, state.cursor.x, state.cursor.y);
        }
        if (state.esp32) {
            pushTrail(esp32Trail, state.esp32.x, state.esp32.y);
        }

        // Passthrough trail: init base position when passthrough activates
        var nowActive = !!(state.passthrough && state.passthrough.active);
        if (nowActive && !ptActive && state.esp32) {
            // Initialize accumulator from ESP32 position when entering passthrough
            ptAccumX = state.esp32.x;
            ptAccumY = state.esp32.y;
        }
        ptActive = nowActive;
        if (ptActive && (ptAccumX > 0 || ptAccumY > 0)) {
            pushTrail(passthroughTrail, ptAccumX, ptAccumY);
        }

        // Capture trail API data for velocity visualization
        if (state.trail) {
            lastTrailData = state.trail;
        }

        drawBlobMap(state);
        drawInference(state);
        drawTrails();
    }

    // ── Draw: Motion Blob Map ─────────────────────────────────
    function drawBlobMap(state) {
        var w = blobCanvas.width, h = blobCanvas.height;
        if (w === 0 || h === 0) return;
        var ctx = blobCtx;
        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = '#0a0a12';
        ctx.fillRect(0, 0, w, h);

        var sx = w / SCREEN_W;
        var sy = h / SCREEN_H;

        // Screen border
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        ctx.strokeRect(1, 1, w - 2, h - 2);

        var blobs = state.blobs || [];
        if (blobs.length === 0) {
            ctx.font = '10px monospace';
            ctx.fillStyle = '#333';
            ctx.textAlign = 'center';
            ctx.fillText('No motion', w / 2, h / 2);
            ctx.textAlign = 'left';
            return;
        }

        for (var i = 0; i < blobs.length; i++) {
            var b = blobs[i];
            var color = i === 0 ? '#2ecc71' : '#3498db';
            var alpha = i === 0 ? 0.7 : 0.5;

            // Bounding box
            if (b.bbox && b.bbox.length >= 4) {
                ctx.globalAlpha = alpha;
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                var bx = b.bbox[0] * sx;
                var by = b.bbox[1] * sy;
                var bw = (b.bbox[2] - b.bbox[0]) * sx;
                var bh = (b.bbox[3] - b.bbox[1]) * sy;
                ctx.strokeRect(bx, by, bw, bh);
            }

            // Centroid dot (centroid is [x, y] tuple from backend)
            if (b.centroid && b.centroid.length >= 2) {
                ctx.globalAlpha = 0.9;
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(b.centroid[0] * sx, b.centroid[1] * sy, i === 0 ? 4 : 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        ctx.globalAlpha = 1.0;

        // Blob count label
        ctx.font = '9px monospace';
        ctx.fillStyle = '#666';
        ctx.fillText(blobs.length + ' blob' + (blobs.length !== 1 ? 's' : ''), 6, h - 6);
    }

    // ── Draw: Mouse Trails ───────────────────────────────────
    function drawTrails() {
        var w = trailsCanvas.width, h = trailsCanvas.height;
        if (w === 0 || h === 0) return;
        var ctx = trailsCtx;
        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = '#0a0a12';
        ctx.fillRect(0, 0, w, h);

        var sx = w / SCREEN_W;
        var sy = h / SCREEN_H;

        // Screen border
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        ctx.strokeRect(1, 1, w - 2, h - 2);

        // Detected trail (cyan)
        drawTrailLine(ctx, detectedTrail, '#00e5ff', sx, sy);
        // ESP32 trail (green)
        drawTrailLine(ctx, esp32Trail, '#2ecc71', sx, sy);
        // Passthrough trail (orange)
        drawTrailLine(ctx, passthroughTrail, '#ff6b35', sx, sy);

        // Current position dots
        drawTrailDot(ctx, detectedTrail, '#00e5ff', sx, sy);
        drawTrailDot(ctx, esp32Trail, '#2ecc71', sx, sy);
        drawTrailDot(ctx, passthroughTrail, '#ff6b35', sx, sy);

        // Velocity arrow from trail API
        var trail = lastTrailData;
        if (trail && trail.moving && trail.last_x != null) {
            var tx = trail.last_x * sx;
            var ty = trail.last_y * sy;
            var vx = trail.velocity.vx;
            var vy = trail.velocity.vy;
            var speed = trail.speed;
            // Scale arrow length: 100 px/s = 30px arrow
            var arrowLen = Math.min(60, speed * 0.3) * sx;
            if (arrowLen > 3) {
                var angle = Math.atan2(vy, vx);
                var ex = tx + Math.cos(angle) * arrowLen;
                var ey = ty + Math.sin(angle) * arrowLen;
                ctx.strokeStyle = '#ff0';
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.9;
                ctx.beginPath();
                ctx.moveTo(tx, ty);
                ctx.lineTo(ex, ey);
                ctx.stroke();
                // Arrowhead
                var headLen = 5;
                ctx.beginPath();
                ctx.moveTo(ex, ey);
                ctx.lineTo(ex - headLen * Math.cos(angle - 0.4),
                           ey - headLen * Math.sin(angle - 0.4));
                ctx.moveTo(ex, ey);
                ctx.lineTo(ex - headLen * Math.cos(angle + 0.4),
                           ey - headLen * Math.sin(angle + 0.4));
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }
        }

        // Speed + source HUD (top-right)
        if (trail) {
            ctx.font = '9px monospace';
            ctx.fillStyle = '#ff0';
            ctx.textAlign = 'right';
            ctx.fillText(Math.round(trail.speed) + ' px/s', w - 4, 12);
            ctx.fillStyle = '#888';
            ctx.fillText(trail.total_points + ' pts', w - 4, 22);
            ctx.textAlign = 'left';
        }

        // Legend
        ctx.font = '9px monospace';
        var legendY = h - 22;
        ctx.fillStyle = '#00e5ff';
        ctx.fillText('\u25CF Detected', 6, legendY);
        ctx.fillStyle = '#2ecc71';
        ctx.fillText('\u25CF ESP32', 6, legendY + 10);
        ctx.fillStyle = '#ff6b35';
        ctx.fillText('\u25CF Passthrough', 80, legendY + 10);
        if (trail && trail.moving) {
            ctx.fillStyle = '#ff0';
            ctx.fillText('\u25CF Velocity', 80, legendY);
        }
    }

    function drawTrailDot(ctx, trail, color, sx, sy) {
        if (trail.length === 0) return;
        var last = trail[trail.length - 1];
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(last.x * sx, last.y * sy, 4, 0, Math.PI * 2);
        ctx.fill();
    }

    function drawTrailLine(ctx, trail, color, sx, sy) {
        if (trail.length < 2) return;
        for (var i = 1; i < trail.length; i++) {
            var alpha = i / trail.length;
            ctx.strokeStyle = color;
            ctx.globalAlpha = alpha * 0.8 + 0.1;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(trail[i - 1].x * sx, trail[i - 1].y * sy);
            ctx.lineTo(trail[i].x * sx, trail[i].y * sy);
            ctx.stroke();
        }
        ctx.globalAlpha = 1.0;
    }

    // ── Draw: Model Inference ────────────────────────────────
    function drawInference(state) {
        var w = inferCanvas.width, h = inferCanvas.height;
        if (w === 0 || h === 0) return;
        var ctx = inferCtx;
        ctx.clearRect(0, 0, w, h);

        ctx.fillStyle = '#0a0a12';
        ctx.fillRect(0, 0, w, h);

        var sx = w / SCREEN_W;
        var sy = h / SCREEN_H;

        // Draw reduced HDMI frame at low opacity
        if (lastFrameImg) {
            ctx.globalAlpha = 0.4;
            ctx.drawImage(lastFrameImg, 0, 0, w, h);
            ctx.globalAlpha = 1.0;
        }

        // YOLO bbox (cyan, opacity scaled by confidence, visible even at low conf)
        if (state.yolo && state.yolo.bbox && state.yolo.bbox.length >= 4) {
            var yAge = state.yolo.age_s >= 0 ? state.yolo.age_s : 999;
            var yConf = state.yolo.confidence;
            // Fade: full opacity when fresh, fade after 3s
            var yAlpha = yAge < 3 ? 1.0 : yAge < 8 ? 1.0 - (yAge - 3) / 5 : 0;
            // Scale by confidence
            yAlpha = yAlpha * Math.min(1.0, yConf + 0.3);
            if (yAlpha > 0.05) {
                ctx.globalAlpha = yAlpha;
                ctx.strokeStyle = '#00ffff';
                ctx.lineWidth = 2;
                var bx = state.yolo.bbox[0] * sx;
                var by = state.yolo.bbox[1] * sy;
                var bw = (state.yolo.bbox[2] - state.yolo.bbox[0]) * sx;
                var bh = (state.yolo.bbox[3] - state.yolo.bbox[1]) * sy;
                ctx.strokeRect(bx, by, bw, bh);
                // Label
                ctx.fillStyle = '#00ffff';
                ctx.font = '9px monospace';
                ctx.fillText('YOLO ' + Math.round(yConf * 100) + '%', bx, by - 3);
                // Center dot
                ctx.beginPath();
                ctx.arc(state.yolo.x * sx, state.yolo.y * sy, 3, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }
        }

        // Vision quadrant highlight (purple, fades after detection)
        if (state.vision && state.vision.age_s >= 0 && state.vision.age_s < 10) {
            var vAlpha = state.vision.age_s < 3 ? 0.25 :
                state.vision.age_s < 8 ? 0.25 * (1.0 - (state.vision.age_s - 3) / 5) : 0;
            if (vAlpha > 0.02) {
                var qx = state.vision.x < SCREEN_W / 2 ? 0 : w / 2;
                var qy = state.vision.y < SCREEN_H / 2 ? 0 : h / 2;
                ctx.globalAlpha = vAlpha;
                ctx.fillStyle = '#a855f7';
                ctx.fillRect(qx, qy, w / 2, h / 2);
                ctx.globalAlpha = 1.0;
                // Vision dot
                ctx.fillStyle = '#a855f7';
                ctx.beginPath();
                ctx.arc(state.vision.x * sx, state.vision.y * sy, 5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // CNN crop indicator (yellow 64x64 box at cursor position)
        if (state.cnn && state.cursor && state.cnn.confidence > 0) {
            var cAlpha = state.cnn.confidence > 0.5 ? 0.6 : 0.3;
            ctx.globalAlpha = cAlpha;
            ctx.strokeStyle = '#f1c40f';
            ctx.lineWidth = 1;
            var cropSize = 64 * sx;
            var ccx = state.cursor.x * sx - cropSize / 2;
            var ccy = state.cursor.y * sy - cropSize / 2;
            ctx.strokeRect(ccx, ccy, cropSize, cropSize);
            ctx.font = '8px monospace';
            ctx.fillStyle = '#f1c40f';
            ctx.fillText('CNN ' + Math.round(state.cnn.confidence * 100) + '%',
                ccx, ccy - 2);
            ctx.globalAlpha = 1.0;
        }

        // Silhouette ROI (yellow dashed)
        if (state.silhouette && state.silhouette.active && state.cursor) {
            ctx.strokeStyle = '#ffff00';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.globalAlpha = 0.4;
            var half = state.silhouette.roi_size / 2;
            var rx = (state.cursor.x - half) * sx;
            var ry = (state.cursor.y - half) * sy;
            var rw = state.silhouette.roi_size * sx;
            var rh = state.silhouette.roi_size * sy;
            ctx.strokeRect(rx, ry, rw, rh);
            ctx.setLineDash([]);
            ctx.globalAlpha = 1.0;
        }

        // Model status labels at bottom
        drawModelLabels(ctx, state, w, h);
    }

    function drawModelLabels(ctx, state, w, h) {
        ctx.font = '9px monospace';
        var y = h - 6;
        var labels = [];

        // YOLO
        if (state.yolo) {
            var yAgeStr = state.yolo.age_s >= 0 ? Math.round(state.yolo.age_s) + 's' : '--';
            labels.push({
                text: 'YOLO ' + (state.yolo.active ?
                    Math.round(state.yolo.confidence * 100) + '% ' : 'off ') + yAgeStr,
                color: state.yolo.active ? '#00ffff' : '#444',
            });
        }

        // CNN
        if (state.cnn) {
            labels.push({
                text: 'CNN ' + Math.round(state.cnn.confidence * 100) + '%',
                color: state.cnn.confidence > 0.7 ? '#f1c40f' :
                    state.cnn.confidence > 0.3 ? '#888' : '#444',
            });
        }

        // Silhouette
        if (state.silhouette) {
            labels.push({
                text: 'Sil ' + (state.silhouette.active ?
                    Math.round(state.silhouette.hz) + 'Hz' : 'off'),
                color: state.silhouette.active ? '#ffff00' : '#444',
            });
        }

        // Vision
        if (state.vision) {
            var vAgeStr = state.vision.age_s >= 0 && state.vision.age_s < 999 ?
                Math.round(state.vision.age_s) + 's' : '--';
            labels.push({
                text: 'Vis ' + (state.vision.confidence || '--') + ' ' + vAgeStr,
                color: state.vision.confidence === 'high' ? '#a855f7' :
                    state.vision.confidence === 'medium' ? '#888' : '#444',
            });
        }

        // Draw horizontally spaced
        var spacing = labels.length > 0 ? w / labels.length : w;
        for (var i = 0; i < labels.length; i++) {
            // Shadow
            ctx.fillStyle = '#000';
            ctx.fillText(labels[i].text, spacing * i + 5, y);
            // Text
            ctx.fillStyle = labels[i].color;
            ctx.fillText(labels[i].text, spacing * i + 4, y - 1);
        }
    }

    return {
        init: init,
        update: update,
        recordPassthroughDelta: recordPassthroughDelta,
    };
})();
