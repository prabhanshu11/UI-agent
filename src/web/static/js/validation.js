/* validation.js — Cursor sample validation page logic */

var currentPage = 0;
var currentFilter = 'all';
var currentSort = 'priority';
var currentLabel = 'all';
var currentSource = 'all';
var totalPages = 1;
var selectedRow = -1;
var currentSamples = [];

function toggleInfo() {
    var bar = document.getElementById('info-bar');
    bar.classList.toggle('collapsed');
    localStorage.setItem('validation_info', bar.classList.contains('collapsed') ? '0' : '1');
}
// Restore info bar state
if (localStorage.getItem('validation_info') === '0') {
    document.getElementById('info-bar').classList.add('collapsed');
}

async function loadSamples() {
    var url = '/api/validation/samples?page=' + currentPage +
        '&sort=' + currentSort +
        '&filter=' + currentFilter +
        '&label=' + currentLabel +
        '&source_filter=' + currentSource;
    var res = await fetch(url);
    var data = await res.json();
    totalPages = Math.ceil(data.total / data.page_size) || 1;
    document.getElementById('page-info').textContent =
        'Page ' + (currentPage + 1) + '/' + totalPages + ' (' + data.total + ' total)';

    // Update filter counts in tabs
    var fc = data.filter_counts || {};
    ['all','unreviewed','reviewed','claude','disagreement','needs_review'].forEach(function(k) {
        var el = document.getElementById('cnt-' + k);
        if (el) el.textContent = '(' + (fc[k] || 0) + ')';
    });

    // Populate source dropdown (once)
    var srcSel = document.getElementById('source-select');
    if (fc.sources && srcSel.options.length <= 1) {
        Object.keys(fc.sources).sort().forEach(function(src) {
            var opt = document.createElement('option');
            opt.value = src;
            opt.textContent = src + ' (' + fc.sources[src] + ')';
            srcSel.appendChild(opt);
        });
    }

    currentSamples = data.samples;
    var tbody = document.getElementById('samples-body');
    tbody.replaceChildren();
    selectedRow = -1;

    data.samples.forEach(function(s, idx) {
        var tr = document.createElement('tr');
        if (s.review) tr.classList.add('reviewed');
        tr.id = 'row-' + s.filename;
        tr.dataset.idx = idx;

        // Patch image
        var tdImg = document.createElement('td');
        var img = document.createElement('img');
        img.className = 'patch-img';
        img.src = '/api/validation/patch/' + encodeURIComponent(s.filename);
        img.onclick = function() { showPreview(s.filename); };
        tdImg.appendChild(img);
        tr.appendChild(tdImg);

        // Label
        var tdLabel = document.createElement('td');
        var labelSpan = document.createElement('span');
        labelSpan.className = s.label === 'pos' ? 'label-pos' : 'label-neg';
        labelSpan.textContent = s.label;
        tdLabel.appendChild(labelSpan);
        if (s.review) {
            var reviewSpan = document.createElement('span');
            reviewSpan.style.cssText = 'color:#4CAF50;font-size:10px;margin-left:4px';
            reviewSpan.textContent = '[' + s.review.action + ']';
            tdLabel.appendChild(reviewSpan);
        }
        if (s.cnn_disagrees) {
            var dBadge = document.createElement('span');
            dBadge.className = 'disagree-badge';
            dBadge.textContent = 'DISAGREE';
            tdLabel.appendChild(dBadge);
        }
        if (s.needs_review) {
            var nrBadge = document.createElement('span');
            nrBadge.style.cssText = 'background:#FF5722;color:#fff;font-size:9px;padding:1px 4px;border-radius:3px;margin-left:4px;font-weight:bold';
            nrBadge.textContent = 'TAG ME';
            tdLabel.appendChild(nrBadge);
        }
        tr.appendChild(tdLabel);

        // Cursor type
        var tdType = document.createElement('td');
        var ctype = s.cursor_type || '';
        if (ctype) {
            var typeSpan = document.createElement('span');
            var typeClass = 'cursor-type ';
            if (ctype === 'arrow') typeClass += 'arrow';
            else if (ctype === 'hand') typeClass += 'hand';
            else if (ctype === 'ibeam' || ctype === 'I-beam') typeClass += 'ibeam';
            else typeClass += 'unknown';
            typeSpan.className = typeClass;
            typeSpan.textContent = ctype;
            tdType.appendChild(typeSpan);
        } else {
            tdType.style.color = '#333';
            tdType.textContent = '-';
        }
        tr.appendChild(tdType);

        // Source
        var tdSource = document.createElement('td');
        var srcSpan = document.createElement('span');
        var srcClass = 'source';
        if (s.source && s.source.indexOf('claude') >= 0) srcClass += ' claude';
        else if (s.source && s.source.indexOf('ground_truth') >= 0) srcClass += ' ground-truth';
        srcSpan.className = srcClass;
        srcSpan.textContent = s.source || '?';
        tdSource.appendChild(srcSpan);
        if (s.is_confusing) {
            var badge = document.createElement('span');
            badge.className = 'confusing-badge';
            badge.textContent = '\u26A0';
            tdSource.appendChild(badge);
        }
        tr.appendChild(tdSource);

        // CNN Confidence
        var tdConf = document.createElement('td');
        var bar = document.createElement('div');
        bar.className = 'conf-bar';
        var fill = document.createElement('div');
        var confVal = s.cnn_confidence != null ? s.cnn_confidence : 0;
        fill.className = 'conf-fill ' + (confVal > 0.7 ? 'conf-high' : confVal > 0.4 ? 'conf-mid' : 'conf-low');
        fill.style.width = Math.round(confVal * 100) + '%';
        bar.appendChild(fill);
        tdConf.appendChild(bar);
        tdConf.appendChild(document.createTextNode(s.cnn_confidence != null ? s.cnn_confidence.toFixed(3) : '?'));
        tr.appendChild(tdConf);

        // Priority
        var tdPri = document.createElement('td');
        tdPri.style.color = '#666';
        tdPri.textContent = s.priority.toFixed(1);
        tr.appendChild(tdPri);

        // Quick actions
        var tdAct = document.createElement('td');
        tdAct.className = 'actions-cell';
        ['OK:correct:btn-ok', 'X:wrong:btn-wrong', 'Del:delete:btn-del'].forEach(function(spec) {
            var parts = spec.split(':');
            var btn = document.createElement('button');
            btn.className = 'btn ' + parts[2];
            btn.textContent = parts[0];
            btn.addEventListener('click', function() { reviewSample(s.filename, parts[1]); });
            tdAct.appendChild(btn);
        });
        tr.appendChild(tdAct);

        // Feature tags
        if (window.tagSchema) {
            Object.keys(window.tagSchema).forEach(function(col) {
                var tdTag = document.createElement('td');
                tdTag.className = 'tag-cell';
                var currentVal = (s.tags && s.tags[col]) || '';
                var schema = window.tagSchema[col];
                schema.values.forEach(function(val) {
                    var chip = document.createElement('span');
                    chip.className = 'tag-chip' + (currentVal === val ? ' active' : '');
                    chip.textContent = val;
                    chip.style.borderColor = schema.color;
                    if (currentVal === val) chip.style.background = schema.color + '30';
                    chip.addEventListener('click', function() {
                        var newVal = currentVal === val ? '' : val;
                        setTag(s.filename, col, newVal);
                        tdTag.querySelectorAll('.tag-chip').forEach(function(c) {
                            c.classList.remove('active');
                            c.style.background = '';
                        });
                        if (newVal) {
                            chip.classList.add('active');
                            chip.style.background = schema.color + '30';
                        }
                        if (!s.tags) s.tags = {};
                        s.tags[col] = newVal;
                    });
                    tdTag.appendChild(chip);
                });
                tr.appendChild(tdTag);
            });
        }

        tbody.appendChild(tr);
    });
}

async function loadStats() {
    var res = await fetch('/api/validation/stats');
    var data = await res.json();
    var bar = document.getElementById('stats-bar');
    bar.replaceChildren();
    var items = [
        data.pos_count + ' pos', data.neg_count + ' neg',
        data.review_count + ' reviewed', 'Model: ' + data.model_version
    ];
    var sources = data.source_distribution || {};
    Object.keys(sources).forEach(function(k) { items.push(k + ': ' + sources[k]); });
    items.forEach(function(txt) {
        var span = document.createElement('span');
        span.textContent = txt;
        bar.appendChild(span);
    });
}

async function setTag(filename, column, value) {
    var tags = {};
    tags[column] = value;
    var res = await fetch('/api/validation/tag', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: filename, tags: tags})
    });
    if (res.ok) {
        showToast(column + '=' + (value || '(cleared)') + ' \u2192 ' + filename);
    }
}

async function loadTagSchema() {
    var res = await fetch('/api/validation/tag_schema');
    var data = await res.json();
    window.tagSchema = data.schema;
    var thead = document.querySelector('#samples-table thead tr');
    if (thead) {
        Object.keys(data.schema).forEach(function(col) {
            var th = document.createElement('th');
            th.textContent = col;
            th.style.color = data.schema[col].color;
            th.style.fontSize = '0.65rem';
            th.style.textTransform = 'uppercase';
            thead.appendChild(th);
        });
    }
}

async function reviewSample(filename, action) {
    var res = await fetch('/api/validation/review', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: filename, action: action})
    });
    if (res.ok) {
        var row = document.getElementById('row-' + filename);
        if (row) row.classList.add('reviewed');
        showToast(action + ': ' + filename);
    }
}

async function retrain() {
    if (!confirm('Start retraining TinyCursorNet?')) return;
    var res = await fetch('/api/validation/retrain', {method: 'POST'});
    var data = await res.json();
    showToast('Training started (PID: ' + data.pid + ')');
}

function setFilter(f) {
    currentFilter = f;
    document.querySelectorAll('.filter-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.filter === f);
    });
    currentPage = 0;
    loadSamples();
}

function setSort(s) {
    currentSort = s;
    document.querySelector('.sort-select').value = s;
    currentPage = 0;
    loadSamples();
}

function setLabel(l) { currentLabel = l; currentPage = 0; loadSamples(); }
function setSource(s) { currentSource = s; currentPage = 0; loadSamples(); }

function nextPage() { if (currentPage < totalPages - 1) { currentPage++; loadSamples(); } }
function prevPage() { if (currentPage > 0) { currentPage--; loadSamples(); } }

var previewFilename = null;

function showPreview(filename) {
    previewFilename = filename;
    var marker = document.getElementById('tip-marker');
    marker.style.display = 'none';
    document.getElementById('preview-img').src = '/api/validation/patch/' + encodeURIComponent(filename);
    document.getElementById('preview-overlay').classList.add('active');
    document.getElementById('preview-info').textContent = 'Click on the cursor tip to annotate. ESC to close.';
}

// Click-to-annotate cursor tip
document.getElementById('preview-img').addEventListener('click', async function(e) {
    if (!previewFilename) return;
    var img = e.target;
    var rect = img.getBoundingClientRect();
    var renderX = e.clientX - rect.left;
    var renderY = e.clientY - rect.top;
    var scaleX = 64 / rect.width;
    var scaleY = 64 / rect.height;
    var tipX = Math.round(renderX * scaleX);
    var tipY = Math.round(renderY * scaleY);
    tipX = Math.max(0, Math.min(63, tipX));
    tipY = Math.max(0, Math.min(63, tipY));

    var marker = document.getElementById('tip-marker');
    marker.style.left = renderX + 'px';
    marker.style.top = renderY + 'px';
    marker.style.display = 'block';

    var res = await fetch('/api/validation/review', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: previewFilename, action: 'annotate_tip', tip_x: tipX, tip_y: tipY})
    });
    if (res.ok) {
        document.getElementById('preview-info').textContent =
            'Tip annotated at (' + tipX + ', ' + tipY + ') \u2014 saved. Click again to adjust, ESC to close.';
        showToast('Tip: (' + tipX + ',' + tipY + ') \u2192 ' + previewFilename);
    }
});

function showToast(msg) {
    var t = document.getElementById('toast');
    t.textContent = msg;
    t.style.display = 'block';
    setTimeout(function() { t.style.display = 'none'; }, 2000);
}

// Keyboard shortcuts for fast tagging
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    var rows = document.querySelectorAll('#samples-body tr');
    if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        if (selectedRow < rows.length - 1) {
            if (selectedRow >= 0) rows[selectedRow].style.outline = '';
            selectedRow++;
            rows[selectedRow].style.outline = '1px solid #ff4444';
            rows[selectedRow].scrollIntoView({block: 'nearest'});
        }
    } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (selectedRow > 0) {
            rows[selectedRow].style.outline = '';
            selectedRow--;
            rows[selectedRow].style.outline = '1px solid #ff4444';
            rows[selectedRow].scrollIntoView({block: 'nearest'});
        }
    } else if (selectedRow >= 0 && selectedRow < currentSamples.length) {
        var fn = currentSamples[selectedRow].filename;
        if (e.key === '1') reviewSample(fn, 'correct');
        else if (e.key === '2') reviewSample(fn, 'wrong');
        else if (e.key === '3') reviewSample(fn, 'confusing');
        else if (e.key === '4') reviewSample(fn, 'delete');
        else if (e.key === ' ') { e.preventDefault(); showPreview(fn); }
    }
    if (e.key === 'Escape') document.getElementById('preview-overlay').classList.remove('active');
});

// Load tag schema first, then samples (schema needed to render tag columns)
loadTagSchema().then(function() { loadSamples(); });
loadStats();
