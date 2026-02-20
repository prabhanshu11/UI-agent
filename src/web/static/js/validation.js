/* validation.js — Cursor sample validation page logic */

var currentPage = 0;
var currentFilter = 'all';
var currentSort = 'priority';
var currentLabel = 'all';
var currentSource = 'all';
var totalPages = 1;
var selectedRow = -1;
var currentSamples = [];
var comparisonActive = false;
var runtimeModelVersion = '';
var comparisonModelVersion = '';

// Batch selection state
var selectedFiles = new Set();

// Auto-refresh state
var heartbeatInterval = null;
var lastHeartbeat = {};
var refreshTimer = null;

// Preview suggestion state
var pendingSuggestions = {};

function toggleInfo() {
    var bar = document.getElementById('info-bar');
    bar.classList.toggle('collapsed');
    localStorage.setItem('validation_info', bar.classList.contains('collapsed') ? '0' : '1');
}
// Restore info bar state
if (localStorage.getItem('validation_info') === '0') {
    document.getElementById('info-bar').classList.add('collapsed');
}

// ── Model comparison ──────────────────────────────────────────────

async function loadModels() {
    var res = await fetch('/api/validation/models');
    var data = await res.json();
    runtimeModelVersion = data.runtime || '--';
    comparisonModelVersion = data.comparison || '';

    // Update runtime display
    document.getElementById('runtime-version').textContent = runtimeModelVersion;
    var rtModel = data.models.find(function(m) {
        return runtimeModelVersion.indexOf(m.version) >= 0;
    });
    if (rtModel && rtModel.val_acc != null) {
        document.getElementById('runtime-acc').textContent =
            (rtModel.val_acc * 100).toFixed(1) + '% acc' +
            (rtModel.major ? ' (' + rtModel.major + ')' : '');
    }

    // Populate comparison dropdown
    var sel = document.getElementById('comparison-select');
    // Keep first option, remove rest
    while (sel.options.length > 1) sel.remove(1);
    data.models.forEach(function(m) {
        var opt = document.createElement('option');
        opt.value = m.version;
        var label = m.version;
        if (m.val_acc != null) label += ' (' + (m.val_acc * 100).toFixed(1) + '%)';
        if (m.major) label += ' [' + m.major + ']';
        opt.textContent = label;
        if (m.version === data.comparison) opt.selected = true;
        sel.appendChild(opt);
    });

    comparisonActive = !!data.comparison;
    updateComparisonAcc(data.models, data.comparison);
}

function updateComparisonAcc(models, version) {
    var el = document.getElementById('comparison-acc');
    if (!version) { el.textContent = ''; return; }
    var m = models.find(function(x) { return x.version === version; });
    if (m && m.val_acc != null) {
        el.textContent = (m.val_acc * 100).toFixed(1) + '% acc' +
            (m.major ? ' (' + m.major + ')' : '');
    }
}

async function setComparisonModel(version) {
    var res = await fetch('/api/validation/set_comparison', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({version: version || null})
    });
    var data = await res.json();
    if (data.ok) {
        comparisonActive = !!data.version;
        comparisonModelVersion = data.version || '';
        showToast(data.version ? 'Comparison: ' + data.version : 'Comparison cleared');
        loadModels();
        loadSamples();
    }
}

// ── YOLO status ──────────────────────────────────────────────────

async function loadYoloStatus() {
    try {
        var res = await fetch('/api/validation/yolo_status');
        var data = await res.json();
        var versionEl = document.getElementById('yolo-model-version');
        var statusEl = document.getElementById('yolo-enrich-status');
        if (versionEl) versionEl.textContent = data.yolo_model || 'none';
        if (statusEl) {
            var parts = [];
            if (data.total_cached > 0) {
                parts.push(data.pct_enriched + '% enriched');
                parts.push(data.detected_count + ' detected');
            }
            if (data.enrichment_running) parts.push('RUNNING...');
            statusEl.textContent = parts.join(' | ');
        }
    } catch(e) { /* ignore */ }
}

async function triggerYoloEnrich() {
    if (!confirm('Run YOLO batch enrichment on all samples?')) return;
    var res = await fetch('/api/validation/yolo_enrich', {method: 'POST'});
    var data = await res.json();
    if (data.status === 'already_running') {
        showToast('YOLO enrichment already running (PID: ' + data.pid + ')');
    } else {
        showToast('YOLO enrichment started (PID: ' + data.pid + ')');
    }
    loadYoloStatus();
}

// ── Confidence bar helper ─────────────────────────────────────────

function makeConfBar(confVal, label) {
    var row = document.createElement('div');
    row.className = 'conf-row ' + (label || '');
    if (label) {
        var lbl = document.createElement('span');
        lbl.className = 'conf-label';
        lbl.textContent = label === 'runtime' ? 'R' : 'C';
        lbl.title = label === 'runtime' ? 'Runtime model' : 'Comparison model';
        row.appendChild(lbl);
    }
    var bar = document.createElement('div');
    bar.className = 'conf-bar';
    var fill = document.createElement('div');
    var v = confVal != null ? confVal : 0;
    fill.className = 'conf-fill ' + (v > 0.7 ? 'conf-high' : v > 0.4 ? 'conf-mid' : 'conf-low');
    fill.style.width = Math.round(v * 100) + '%';
    bar.appendChild(fill);
    row.appendChild(bar);
    var txt = document.createElement('span');
    txt.className = 'conf-val';
    txt.textContent = confVal != null ? confVal.toFixed(3) : '?';
    row.appendChild(txt);
    return row;
}

function makeYoloBar(sample) {
    var td = document.createElement('td');
    td.className = 'yolo-cell';

    if (!sample.yolo_enriched) {
        var dash = document.createElement('span');
        dash.className = 'yolo-badge yolo-none';
        dash.textContent = '\u2014';
        dash.title = 'Not yet enriched with YOLO';
        td.appendChild(dash);
        return td;
    }

    // Detection badge
    var badge = document.createElement('span');
    if (sample.yolo_detected) {
        badge.className = 'yolo-badge yolo-yes';
        badge.textContent = '\u2713';
        badge.title = 'YOLO detected cursor';
    } else {
        badge.className = 'yolo-badge yolo-no';
        badge.textContent = '\u2717';
        badge.title = 'YOLO: no cursor detected';
    }
    td.appendChild(badge);

    // Confidence bar (blue tint)
    if (sample.yolo_conf != null) {
        var bar = document.createElement('div');
        bar.className = 'conf-bar yolo-bar';
        var fill = document.createElement('div');
        var v = sample.yolo_conf;
        fill.className = 'conf-fill yolo-fill';
        fill.style.width = Math.round(v * 100) + '%';
        bar.appendChild(fill);
        td.appendChild(bar);

        var txt = document.createElement('span');
        txt.className = 'conf-val';
        txt.textContent = v.toFixed(3);
        td.appendChild(txt);
    }

    // Disagree indicator
    if (sample.yolo_disagrees) {
        var dis = document.createElement('span');
        dis.className = 'yolo-disagree-badge';
        dis.textContent = '!';
        dis.title = 'YOLO disagrees with label';
        td.appendChild(dis);
    }

    return td;
}

// ── Samples table ─────────────────────────────────────────────────

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
    ['all','unreviewed','reviewed','claude','disagreement','needs_review','yolo_detected','yolo_disagrees'].forEach(function(k) {
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

        // Batch checkbox
        var tdCheck = document.createElement('td');
        tdCheck.className = 'batch-td';
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.className = 'batch-cb';
        cb.checked = selectedFiles.has(s.filename);
        cb.addEventListener('change', function() {
            toggleSelection(s.filename, cb.checked);
        });
        tdCheck.appendChild(cb);
        tr.appendChild(tdCheck);

        // Patch image — show context crop as thumbnail if available, 64x64 otherwise
        var tdImg = document.createElement('td');
        var img = document.createElement('img');
        img.className = 'patch-img';
        if (s.has_context) {
            img.src = '/api/validation/context/' + encodeURIComponent(s.filename);
            img.title = '256x256 context (click for full view)';
        } else {
            img.src = '/api/validation/patch/' + encodeURIComponent(s.filename);
        }
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
        // CNN agreement/disagreement indicator
        if (s.cnn_predicted) {
            var cnnBadge = document.createElement('span');
            if (s.cnn_disagrees) {
                cnnBadge.className = 'cnn-badge disagree';
                cnnBadge.textContent = 'CNN\u2192' + s.cnn_predicted.toUpperCase() +
                    ', Label: ' + s.label.toUpperCase();
                cnnBadge.title = 'Model predicts ' + s.cnn_predicted +
                    ' but sample is labeled ' + s.label;
            } else {
                cnnBadge.className = 'cnn-badge agree';
                cnnBadge.textContent = 'CNN \u2713';
                cnnBadge.title = 'Model agrees with label (' + s.label + ')';
            }
            tdLabel.appendChild(cnnBadge);
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

        // Source + origin badge
        var tdSource = document.createElement('td');
        var originBadge = document.createElement('span');
        originBadge.className = 'origin-badge ' + (s.origin_type || 'sensor');
        var originLabels = {ground_truth: 'GROUND TRUTH', sensor: 'SENSOR', prediction: 'PREDICTION'};
        originBadge.textContent = originLabels[s.origin_type] || 'SENSOR';
        tdSource.appendChild(originBadge);
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

        // CNN Confidence — dual bar when comparison active
        var tdConf = document.createElement('td');
        if (comparisonActive && s.comparison_confidence != null) {
            var dual = document.createElement('div');
            dual.className = 'dual-conf';
            dual.appendChild(makeConfBar(s.cnn_confidence, 'runtime'));
            var compRow = makeConfBar(s.comparison_confidence, 'comparison');
            if (s.models_disagree) {
                var dis = document.createElement('span');
                dis.className = 'models-disagree-badge';
                dis.textContent = '\u2716';
                dis.title = 'Models disagree: runtime=' + (s.cnn_predicted || '?') +
                    ' vs comparison=' + (s.comp_predicted || '?');
                compRow.appendChild(dis);
            }
            dual.appendChild(compRow);
            tdConf.appendChild(dual);
        } else {
            // Single bar (original behavior)
            tdConf.appendChild(makeConfBar(s.cnn_confidence));
        }
        tr.appendChild(tdConf);

        // YOLO column
        tr.appendChild(makeYoloBar(s));

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

    updateBatchBar();
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
    // Remove old dynamic headers (on refresh)
    document.querySelectorAll('.tag-th').forEach(function(el) { el.remove(); });
    var thead = document.querySelector('#samples-table thead tr');
    if (thead) {
        Object.keys(data.schema).forEach(function(col) {
            var th = document.createElement('th');
            th.className = 'tag-th';
            th.style.color = data.schema[col].color;
            th.style.fontSize = '0.65rem';
            th.style.textTransform = 'uppercase';
            var nameSpan = document.createElement('span');
            nameSpan.textContent = col;
            th.appendChild(nameSpan);
            var addBtn = document.createElement('button');
            addBtn.className = 'schema-add-btn';
            addBtn.textContent = '+';
            addBtn.title = 'Add new value to ' + col;
            addBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                promptAddValue(col, data.schema[col].color);
            });
            th.appendChild(addBtn);
            thead.appendChild(th);
        });
        var addColTh = document.createElement('th');
        addColTh.className = 'tag-th';
        var addColBtn = document.createElement('button');
        addColBtn.className = 'schema-add-col-btn';
        addColBtn.textContent = '+ Column';
        addColBtn.title = 'Add a new tag column';
        addColBtn.addEventListener('click', function() { promptAddColumn(); });
        addColTh.appendChild(addColBtn);
        thead.appendChild(addColTh);
    }
}

function promptAddValue(column, color) {
    var val = prompt('Add new value to "' + column + '":');
    if (!val || !val.trim()) return;
    fetch('/api/validation/schema/add_value', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({column: column, value: val.trim()})
    }).then(function(r) { return r.json(); }).then(function(d) {
        if (d.ok) {
            showToast('Added "' + val.trim() + '" to ' + column);
            loadTagSchema().then(function() { loadSamples(); });
        }
    });
}

function promptAddColumn() {
    var col = prompt('New column name (e.g. "scene_type"):');
    if (!col || !col.trim()) return;
    var vals = prompt('Comma-separated values (e.g. "static,scrolling,popup"):');
    if (!vals || !vals.trim()) return;
    var valArr = vals.split(',').map(function(v) { return v.trim(); }).filter(Boolean);
    if (valArr.length === 0) return;
    var colors = ['#e67e22', '#1abc9c', '#9b59b6', '#e74c3c', '#3498db', '#f39c12'];
    var color = colors[Object.keys(window.tagSchema || {}).length % colors.length];
    fetch('/api/validation/schema/add_column', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({column: col.trim(), values: valArr, color: color})
    }).then(function(r) { return r.json(); }).then(function(d) {
        if (d.ok) {
            showToast('Added column "' + col.trim() + '" with values: ' + valArr.join(', '));
            loadTagSchema().then(function() { loadSamples(); });
        }
    });
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

// ── Batch selection ───────────────────────────────────────────────

function toggleSelection(filename, checked) {
    if (checked) {
        selectedFiles.add(filename);
    } else {
        selectedFiles.delete(filename);
    }
    updateBatchBar();
}

function toggleSelectAll(checked) {
    currentSamples.forEach(function(s) {
        if (checked) selectedFiles.add(s.filename);
        else selectedFiles.delete(s.filename);
    });
    // Update checkboxes in DOM
    document.querySelectorAll('.batch-cb').forEach(function(cb) {
        cb.checked = checked;
    });
    updateBatchBar();
}

function clearSelection() {
    selectedFiles.clear();
    document.querySelectorAll('.batch-cb').forEach(function(cb) {
        cb.checked = false;
    });
    document.getElementById('select-all').checked = false;
    updateBatchBar();
}

function updateBatchBar() {
    var bar = document.getElementById('batch-bar');
    var count = selectedFiles.size;
    if (count > 0) {
        bar.style.display = 'flex';
        document.getElementById('batch-count').textContent = count + ' selected';
    } else {
        bar.style.display = 'none';
    }
}

async function batchReview(action) {
    if (selectedFiles.size === 0) return;
    var filenames = Array.from(selectedFiles);
    if (!confirm(action.toUpperCase() + ' ' + filenames.length + ' samples?')) return;
    var res = await fetch('/api/validation/batch_review', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filenames: filenames, action: action})
    });
    var data = await res.json();
    if (data.ok) {
        showToast('Batch ' + action + ': ' + data.count + ' samples');
        clearSelection();
        loadSamples();
    }
}

function batchTagPrompt() {
    if (selectedFiles.size === 0) return;
    var schema = window.tagSchema || {};
    var cols = Object.keys(schema);
    if (cols.length === 0) { showToast('No tag schema loaded'); return; }
    var col = prompt('Tag column (' + cols.join(', ') + '):');
    if (!col || !schema[col]) { showToast('Invalid column'); return; }
    var vals = schema[col].values;
    var val = prompt('Value (' + vals.join(', ') + '):');
    if (!val) return;

    var filenames = Array.from(selectedFiles);
    var tags = {};
    tags[col] = val;
    fetch('/api/validation/batch_tag', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filenames: filenames, tags: tags})
    }).then(function(r) { return r.json(); }).then(function(d) {
        if (d.ok) {
            showToast('Batch tagged ' + d.count + ' samples: ' + col + '=' + val);
            clearSelection();
            loadSamples();
        }
    });
}

// ── Preview overlay ───────────────────────────────────────────────

var previewFilename = null;

function showPreview(filename) {
    previewFilename = filename;
    var marker = document.getElementById('tip-marker');
    marker.style.display = 'none';
    document.getElementById('preview-img').src = '/api/validation/patch/' + encodeURIComponent(filename);
    document.getElementById('preview-overlay').classList.add('active');

    // Show context crop if available
    var ctxImg = document.getElementById('context-img');
    var ctxLabel = document.getElementById('context-label');
    var sample = currentSamples.find(function(s) { return s.filename === filename; });

    if (sample && sample.has_context) {
        ctxImg.src = '/api/validation/context/' + encodeURIComponent(filename);
        ctxImg.style.display = 'block';
        ctxLabel.style.display = 'block';
    } else {
        ctxImg.style.display = 'none';
        ctxLabel.style.display = 'none';
    }

    // Build info text with model confidence
    var info = 'Click on the cursor tip to annotate. ESC to close.';
    if (sample) {
        var parts = [];
        if (sample.cnn_confidence != null) {
            parts.push('Runtime: ' + sample.cnn_confidence.toFixed(3) +
                ' \u2192 ' + (sample.cnn_predicted || '?'));
        }
        if (comparisonActive && sample.comparison_confidence != null) {
            parts.push('Compare: ' + sample.comparison_confidence.toFixed(3) +
                ' \u2192 ' + (sample.comp_predicted || '?'));
            if (sample.models_disagree) parts.push('\u26A0 MODELS DISAGREE');
        }
        if (parts.length) info = parts.join(' | ') + '\n' + info;
    }
    document.getElementById('preview-info').textContent = info;

    // Fetch YOLO suggestions
    loadSuggestions(filename);
}

async function loadSuggestions(filename) {
    var sugBar = document.getElementById('suggestion-bar');
    var sugInfo = document.getElementById('suggestion-info');
    var sugTags = document.getElementById('suggestion-tags');
    var acceptBtn = document.getElementById('accept-suggestions-btn');

    sugBar.style.display = 'none';
    sugTags.replaceChildren();
    acceptBtn.style.display = 'none';
    pendingSuggestions = {};

    try {
        var res = await fetch('/api/validation/suggest/' + encodeURIComponent(filename));
        var data = await res.json();

        var parts = [];
        if (data.yolo && data.yolo.detected) {
            parts.push('YOLO: cursor detected (' + (data.yolo.conf ? (data.yolo.conf * 100).toFixed(0) + '%' : '?') + ' conf)');
        } else if (data.yolo && data.yolo.source) {
            parts.push('YOLO: no cursor detected');
        }
        if (data.cnn && data.cnn.conf != null) {
            parts.push('CNN: ' + data.cnn.predicted + ' (' + (data.cnn.conf * 100).toFixed(0) + '% conf)');
        }

        if (parts.length === 0) return;

        sugInfo.textContent = parts.join(' | ');
        sugBar.style.display = 'flex';

        // Show suggested tags
        var suggestions = data.suggestions || {};
        var sugKeys = Object.keys(suggestions);
        if (sugKeys.length > 0) {
            pendingSuggestions = suggestions;
            var label = document.createElement('span');
            label.className = 'suggestion-label';
            label.textContent = 'Suggested: ';
            sugTags.appendChild(label);
            sugKeys.forEach(function(k) {
                var chip = document.createElement('span');
                chip.className = 'tag-chip suggestion-chip';
                chip.textContent = k + '=' + suggestions[k];
                sugTags.appendChild(chip);
            });
            acceptBtn.style.display = 'inline-block';
        }
    } catch(e) { /* ignore fetch errors */ }
}

async function acceptSuggestions() {
    if (!previewFilename || Object.keys(pendingSuggestions).length === 0) return;
    var res = await fetch('/api/validation/tag', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: previewFilename, tags: pendingSuggestions})
    });
    if (res.ok) {
        showToast('Applied suggestions to ' + previewFilename);
        document.getElementById('accept-suggestions-btn').style.display = 'none';
        // Update suggestion chips to show applied
        document.querySelectorAll('.suggestion-chip').forEach(function(c) {
            c.classList.add('active');
        });
    }
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

// ── Auto-refresh polling ──────────────────────────────────────────

function startHeartbeat() {
    if (heartbeatInterval) return;
    heartbeatInterval = setInterval(checkHeartbeat, 30000);
}

async function checkHeartbeat() {
    try {
        var res = await fetch('/api/validation/heartbeat');
        var data = await res.json();

        if (lastHeartbeat.model_version &&
            (data.model_version !== lastHeartbeat.model_version ||
             data.sample_count !== lastHeartbeat.sample_count)) {
            showRefreshBar('Data changed: model=' + data.model_version +
                ', samples=' + data.sample_count);
        }

        // Update YOLO enrichment running status
        if (data.enrichment_running) {
            var el = document.getElementById('yolo-enrich-status');
            if (el) {
                el.textContent = 'ENRICHING... (' + data.yolo_cache_count + ' cached)';
                el.style.color = '#FF9800';
            }
        } else if (lastHeartbeat.enrichment_running && !data.enrichment_running) {
            // Enrichment just finished
            loadYoloStatus();
            showToast('YOLO enrichment complete');
        }

        lastHeartbeat = data;
    } catch(e) { /* ignore */ }
}

function showRefreshBar(msg) {
    var bar = document.getElementById('refresh-bar');
    document.getElementById('refresh-msg').textContent = msg;
    bar.style.display = 'flex';
    // Auto-refresh after 5s
    var countdown = 5;
    var cdEl = document.getElementById('refresh-countdown');
    cdEl.textContent = '(' + countdown + 's)';
    refreshTimer = setInterval(function() {
        countdown--;
        cdEl.textContent = '(' + countdown + 's)';
        if (countdown <= 0) {
            doRefresh();
        }
    }, 1000);
}

function dismissRefresh() {
    document.getElementById('refresh-bar').style.display = 'none';
    if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }
}

function doRefresh() {
    dismissRefresh();
    loadModels();
    loadSamples();
    loadStats();
    loadYoloStatus();
    showToast('Refreshed');
}

// ── Toast ─────────────────────────────────────────────────────────

function showToast(msg) {
    var t = document.getElementById('toast');
    t.textContent = msg;
    t.style.display = 'block';
    setTimeout(function() { t.style.display = 'none'; }, 2000);
}

// ── Keyboard shortcuts ────────────────────────────────────────────

document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    var rows = document.querySelectorAll('#samples-body tr');

    // Batch shortcuts with Shift
    if (e.shiftKey) {
        if (e.key === 'A' || e.key === 'a') {
            e.preventDefault();
            var allSelected = selectedFiles.size === currentSamples.length;
            toggleSelectAll(!allSelected);
            document.getElementById('select-all').checked = !allSelected;
            return;
        }
        if (e.key === '!') { // Shift+1
            e.preventDefault();
            batchReview('correct');
            return;
        }
        if (e.key === '@') { // Shift+2
            e.preventDefault();
            batchReview('wrong');
            return;
        }
    }

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
    } else if (e.key === 'x' && selectedRow >= 0 && selectedRow < currentSamples.length) {
        // Toggle selection of current row
        var fn = currentSamples[selectedRow].filename;
        var isSelected = selectedFiles.has(fn);
        toggleSelection(fn, !isSelected);
        var cb = rows[selectedRow].querySelector('.batch-cb');
        if (cb) cb.checked = !isSelected;
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

// ── Init ──────────────────────────────────────────────────────────
loadModels();
loadYoloStatus();
loadTagSchema().then(function() { loadSamples(); });
loadStats();
startHeartbeat();
