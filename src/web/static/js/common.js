/* common.js — Shared utilities for all KVM dashboard pages */

/** Set a status dot color by element ID. */
function setDot(id, color) {
    var el = document.getElementById(id);
    if (el) el.className = 'dot ' + color;
}

/** Set a hardware value row: green if ok, red if not. */
function setHW(id, ok, okText, failText) {
    var el = document.getElementById(id);
    if (!el) return;
    el.textContent = ok ? okText : failText;
    el.className = 'hw-value ' + (ok ? 'ok' : 'err');
}

/** Show a toast notification. */
function showToast(msg, duration) {
    var t = document.getElementById('toast');
    if (!t) return;
    t.textContent = msg;
    t.style.display = 'block';
    setTimeout(function() { t.style.display = 'none'; }, duration || 2000);
}

/** Fetch JSON with error handling. Returns parsed JSON or null on error. */
async function fetchJSON(url, opts) {
    try {
        var res = await fetch(url, opts);
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        console.error('fetchJSON error:', url, e);
        return null;
    }
}
