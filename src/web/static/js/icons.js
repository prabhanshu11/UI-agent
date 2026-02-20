/* icons.js — Waybar Icon Validation page logic */

let allResults = [];
let currentFilter = 'all';
let currentTestFilter = 'all';
let currentDeviceFilter = 'all';
let currentPage = 0;
const PAGE_SIZE = 20;
let focusedRow = -1;

// ── Data loading ───────────────────────────────────────────

async function loadResults() {
    const params = new URLSearchParams({
        page: currentPage,
        page_size: PAGE_SIZE,
        filter: currentFilter,
        test_filter: currentTestFilter,
        device_filter: currentDeviceFilter,
    });
    try {
        const resp = await fetch(`/api/icons/validations?${params}`);
        const data = await resp.json();
        allResults = data.results || [];
        renderTable(allResults);
        renderStats(data.counts || {});
        renderPagination(data.total || 0);
    } catch (err) {
        console.error('Failed to load results:', err);
    }
}

// ── Rendering ──────────────────────────────────────────────

function renderStats(counts) {
    const bar = document.getElementById('stats-bar');
    bar.textContent = `Total: ${counts.all || 0}  |  Passed: ${counts.passed || 0}  |  Failed: ${counts.failed || 0}  |  Unreviewed: ${counts.unreviewed || 0}`;
}

function renderTable(results) {
    const tbody = document.getElementById('results-body');
    // Clear existing rows
    while (tbody.firstChild) tbody.removeChild(tbody.firstChild);

    if (!results.length) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 10;
        td.style.textAlign = 'center';
        td.style.color = '#555';
        td.style.padding = '24px';
        td.textContent = 'No results found. Run tests first.';
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
    }

    results.forEach((r, i) => {
        const tr = document.createElement('tr');
        tr.dataset.idx = i;
        tr.addEventListener('dblclick', () => openDetail(i));
        if (focusedRow === i) tr.classList.add('focused');

        // Row number
        addCell(tr, String(currentPage * PAGE_SIZE + i + 1));

        // Timestamp
        const tsCell = addCell(tr, formatTimestamp(r.created_at));
        tsCell.className = 'timestamp';

        // Test name
        const testCell = addCell(tr, r.test_name.replace('test_', '').replace(/_/g, ' '));
        testCell.className = 'test-name';

        // Device
        addCell(tr, r.source_device || '?');

        // Model
        const modelCell = addCell(tr, (r.vision_model || '').replace('anthropic/', ''));
        modelCell.className = 'model-name';

        // Icons found
        const foundCell = document.createElement('td');
        const found = parseJsonArray(r.icons_found);
        found.forEach(ic => {
            const chip = document.createElement('span');
            chip.className = 'icon-chip found';
            chip.textContent = ic;
            foundCell.appendChild(chip);
        });
        if (!found.length) { foundCell.style.color = '#555'; foundCell.textContent = 'none'; }
        tr.appendChild(foundCell);

        // Icons missing
        const missCell = document.createElement('td');
        const missing = parseJsonArray(r.icons_missing);
        missing.forEach(ic => {
            const chip = document.createElement('span');
            chip.className = 'icon-chip missing';
            chip.textContent = ic;
            missCell.appendChild(chip);
        });
        if (!missing.length) { missCell.style.color = '#333'; missCell.textContent = '\u2014'; }
        tr.appendChild(missCell);

        // Result badge
        const resultCell = document.createElement('td');
        const resultBadge = document.createElement('span');
        resultBadge.className = r.passed ? 'badge badge-pass' : 'badge badge-fail';
        resultBadge.textContent = r.passed ? 'PASS' : 'FAIL';
        resultCell.appendChild(resultBadge);
        tr.appendChild(resultCell);

        // Review badge
        const reviewCell = document.createElement('td');
        const reviewBadge = document.createElement('span');
        if (r.human_review === 'ok') {
            reviewBadge.className = 'badge badge-ok';
            reviewBadge.textContent = 'OK';
        } else if (r.human_review === 'wrong') {
            reviewBadge.className = 'badge badge-wrong';
            reviewBadge.textContent = 'WRONG';
        } else {
            reviewBadge.className = 'badge badge-pending';
            reviewBadge.textContent = '\u2014';
        }
        reviewCell.appendChild(reviewBadge);
        tr.appendChild(reviewCell);

        // Action buttons
        const actCell = document.createElement('td');
        const okBtn = document.createElement('button');
        okBtn.className = 'action-btn ok';
        okBtn.textContent = 'OK';
        okBtn.addEventListener('click', (e) => { e.stopPropagation(); submitReview(r.id, 'ok'); });
        actCell.appendChild(okBtn);

        const wrongBtn = document.createElement('button');
        wrongBtn.className = 'action-btn wrong';
        wrongBtn.textContent = 'Wrong';
        wrongBtn.addEventListener('click', (e) => { e.stopPropagation(); submitReview(r.id, 'wrong'); });
        actCell.appendChild(wrongBtn);
        tr.appendChild(actCell);

        tbody.appendChild(tr);
    });
}

function addCell(tr, text) {
    const td = document.createElement('td');
    td.textContent = text;
    tr.appendChild(td);
    return td;
}

function renderPagination(total) {
    const pages = Math.ceil(total / PAGE_SIZE);
    const el = document.getElementById('pagination');
    while (el.firstChild) el.removeChild(el.firstChild);
    if (pages <= 1) return;
    for (let i = 0; i < pages; i++) {
        const btn = document.createElement('button');
        btn.className = 'page-btn' + (i === currentPage ? ' active' : '');
        btn.textContent = String(i + 1);
        btn.addEventListener('click', () => goToPage(i));
        el.appendChild(btn);
    }
}

// ── Filters ────────────────────────────────────────────────

function setFilter(f) {
    currentFilter = f;
    currentPage = 0;
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === f);
    });
    loadResults();
}

function setTestFilter(val) { currentTestFilter = val; currentPage = 0; loadResults(); }
function setDeviceFilter(val) { currentDeviceFilter = val; currentPage = 0; loadResults(); }
function goToPage(p) { currentPage = p; loadResults(); }

// ── Actions ────────────────────────────────────────────────

async function submitReview(id, action) {
    try {
        const resp = await fetch('/api/icons/review', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ id, action }),
        });
        if (resp.ok) loadResults();
    } catch (err) {
        console.error('Review failed:', err);
    }
}

async function runTest() {
    const btn = document.querySelector('.btn-run-test');
    btn.textContent = 'Running...';
    btn.disabled = true;
    try {
        const resp = await fetch('/api/icons/run_test', { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            loadResults();
        } else {
            alert('Test failed: ' + (data.error || 'unknown'));
        }
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.textContent = 'Run Test Now';
        btn.disabled = false;
    }
}

async function ingestResults() {
    const btn = document.querySelector('.btn-ingest');
    btn.textContent = 'Ingesting...';
    btn.disabled = true;
    try {
        const resp = await fetch('/api/icons/ingest', { method: 'POST' });
        const data = await resp.json();
        alert(`Ingested ${data.count || 0} new results`);
        loadResults();
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.textContent = 'Ingest to Datalake';
        btn.disabled = false;
    }
}

// ── Detail overlay ─────────────────────────────────────────

function openDetail(idx) {
    const r = allResults[idx];
    if (!r) return;
    document.getElementById('detail-title').textContent = r.test_name;

    const found = parseJsonArray(r.icons_found);
    const missing = parseJsonArray(r.icons_missing);

    // Meta
    const meta = document.getElementById('detail-meta');
    while (meta.firstChild) meta.removeChild(meta.firstChild);
    const metaLines = [
        `Timestamp: ${r.created_at}`,
        `Model: ${r.vision_model}`,
        `Device: ${r.source_device}`,
        `Result: ${r.passed ? 'PASS' : 'FAIL'}`,
    ];
    if (r.battery_color) metaLines.push(`Battery Color: ${r.battery_color}`);
    if (r.power_profile_text) metaLines.push(`Power Profile: ${r.power_profile_text}`);
    if (r.ac_power !== null && r.ac_power !== undefined) metaLines.push(`AC Power: ${r.ac_power ? 'On' : 'Off'}`);
    metaLines.forEach(line => {
        const div = document.createElement('div');
        div.textContent = line;
        meta.appendChild(div);
    });

    // Found icons
    const foundEl = document.getElementById('detail-found');
    while (foundEl.firstChild) foundEl.removeChild(foundEl.firstChild);
    const foundH = document.createElement('h4');
    foundH.textContent = `Found (${found.length})`;
    foundEl.appendChild(foundH);
    found.forEach(ic => {
        const chip = document.createElement('div');
        chip.className = 'icon-chip found';
        chip.textContent = ic;
        foundEl.appendChild(chip);
    });

    // Missing icons
    const missEl = document.getElementById('detail-missing');
    while (missEl.firstChild) missEl.removeChild(missEl.firstChild);
    const missH = document.createElement('h4');
    missH.textContent = `Missing (${missing.length})`;
    missEl.appendChild(missH);
    if (missing.length) {
        missing.forEach(ic => {
            const chip = document.createElement('div');
            chip.className = 'icon-chip missing';
            chip.textContent = ic;
            missEl.appendChild(chip);
        });
    } else {
        const none = document.createElement('span');
        none.style.color = '#555';
        none.textContent = 'None';
        missEl.appendChild(none);
    }

    // Notes
    document.getElementById('detail-notes').textContent = r.notes || 'No notes';

    document.getElementById('detail-overlay').classList.add('open');
}

function closeDetail() {
    document.getElementById('detail-overlay').classList.remove('open');
}

// ── Keyboard navigation ────────────────────────────────────

document.addEventListener('keydown', (e) => {
    if (document.getElementById('detail-overlay').classList.contains('open')) {
        if (e.key === 'Escape') closeDetail();
        return;
    }
    const rows = allResults.length;
    if (!rows) return;

    if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        focusedRow = Math.min(focusedRow + 1, rows - 1);
        renderTable(allResults);
    } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        focusedRow = Math.max(focusedRow - 1, 0);
        renderTable(allResults);
    } else if (e.key === '1' && focusedRow >= 0) {
        submitReview(allResults[focusedRow].id, 'ok');
    } else if (e.key === '2' && focusedRow >= 0) {
        submitReview(allResults[focusedRow].id, 'wrong');
    } else if (e.key === ' ' && focusedRow >= 0) {
        e.preventDefault();
        openDetail(focusedRow);
    }
});

// ── Helpers ────────────────────────────────────────────────

function parseJsonArray(val) {
    if (!val) return [];
    if (Array.isArray(val)) return val;
    try { return JSON.parse(val); } catch { return []; }
}

function formatTimestamp(ts) {
    if (!ts) return '\u2014';
    try {
        const d = new Date(ts);
        return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }) + ' ' +
               d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
    } catch { return ts; }
}

// ── Init ───────────────────────────────────────────────────
loadResults();
