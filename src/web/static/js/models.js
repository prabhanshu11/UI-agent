/* models.js — Model tracking page logic */

async function loadModels() {
    var data = await fetchJSON('/api/models');
    if (!data) {
        document.getElementById('stats-row').textContent = 'Failed to load model data';
        return;
    }

    renderStats(data);
    renderFamilies(data);
}

function renderStats(data) {
    var row = document.getElementById('stats-row');
    while (row.firstChild) row.removeChild(row.firstChild);

    var items = [
        ['CNN Models', data.cnn_count || 0],
        ['YOLO Models', data.yolo_count || 0],
        ['Training Samples', data.total_samples || 0],
        ['Loss Events', data.loss_event_count || 0],
        ['Velocity Runs', data.velocity_run_count || 0],
    ];

    items.forEach(function(item) {
        var stat = document.createElement('div');
        stat.className = 'stat-item';
        var lbl = document.createElement('span');
        lbl.className = 'stat-label';
        lbl.textContent = item[0] + ':';
        var val = document.createElement('span');
        val.className = 'stat-value';
        val.textContent = item[1];
        stat.appendChild(lbl);
        stat.appendChild(val);
        row.appendChild(stat);
    });
}

function renderFamilies(data) {
    var grid = document.getElementById('models-grid');
    while (grid.firstChild) grid.removeChild(grid.firstChild);

    // CNN Family
    if (data.cnn_models && data.cnn_models.length > 0) {
        grid.appendChild(buildCNNFamily(data.cnn_models, data.active_cnn));
    }

    // YOLO Family
    if (data.yolo_models && data.yolo_models.length > 0) {
        grid.appendChild(buildYOLOFamily(data.yolo_models));
    }

    // Training Pipeline
    if (data.sample_stats) {
        grid.appendChild(buildSampleStats(data.sample_stats));
    }

    // Velocity Tests
    if (data.velocity_runs && data.velocity_runs.length > 0) {
        grid.appendChild(buildVelocityRuns(data.velocity_runs));
    }

    // Loss Events Summary
    if (data.loss_event_count > 0) {
        grid.appendChild(buildLossEventSummary(data));
    }
}

function buildCNNFamily(models, activeName) {
    var card = document.createElement('div');
    card.className = 'model-family';

    var title = document.createElement('div');
    title.className = 'family-title';
    title.style.color = '#9b59b6';
    title.textContent = 'TinyCursorNet (14K params)';

    var badge = document.createElement('span');
    badge.className = 'family-badge badge-count';
    badge.textContent = models.length + ' versions';
    title.appendChild(badge);
    card.appendChild(title);

    models.forEach(function(m) {
        var isActive = (m.version === activeName);
        var row = document.createElement('div');
        row.className = 'version-row' + (isActive ? ' active' : '');

        var name = document.createElement('div');
        name.className = 'version-name' + (isActive ? ' active' : '');
        name.textContent = m.version + (isActive ? ' \u2713' : '');
        row.appendChild(name);

        // Val accuracy
        var accMetric = makeMetric('Val Acc', (m.val_accuracy * 100).toFixed(1) + '%',
            m.val_accuracy > 0.8 ? 'good' : m.val_accuracy > 0.7 ? 'ok' : 'bad');
        row.appendChild(accMetric);

        // Samples
        var sampMetric = makeMetric('Samples', m.train_samples + '/' + m.val_samples, '');
        row.appendChild(sampMetric);

        // Epochs
        var epochMetric = makeMetric('Epochs', String(m.epochs), '');
        row.appendChild(epochMetric);

        // Best loss
        var lossMetric = makeMetric('Best Loss', m.best_loss.toFixed(4),
            m.best_loss < 0.4 ? 'good' : m.best_loss < 0.5 ? 'ok' : 'bad');
        row.appendChild(lossMetric);

        card.appendChild(row);
    });

    return card;
}

function buildYOLOFamily(models) {
    var card = document.createElement('div');
    card.className = 'model-family';

    var title = document.createElement('div');
    title.className = 'family-title';
    title.style.color = '#00e5ff';
    title.textContent = 'YOLOv8n-cursor';
    var badge = document.createElement('span');
    badge.className = 'family-badge badge-count';
    badge.textContent = models.length + ' versions';
    title.appendChild(badge);
    card.appendChild(title);

    models.forEach(function(m) {
        var row = document.createElement('div');
        row.className = 'version-row';

        var name = document.createElement('div');
        name.className = 'version-name';
        name.style.color = '#00e5ff';
        name.textContent = m.name;
        row.appendChild(name);

        if (m.metrics) {
            var mapMetric = makeMetric('mAP50', m.metrics.map50 !== undefined ?
                (m.metrics.map50 * 100).toFixed(1) + '%' : '--', '');
            row.appendChild(mapMetric);

            var precMetric = makeMetric('Precision', m.metrics.precision !== undefined ?
                (m.metrics.precision * 100).toFixed(1) + '%' : '--', '');
            row.appendChild(precMetric);

            var recMetric = makeMetric('Recall', m.metrics.recall !== undefined ?
                (m.metrics.recall * 100).toFixed(1) + '%' : '--', '');
            row.appendChild(recMetric);
        }

        var sizeMetric = makeMetric('Size', m.size_mb.toFixed(1) + ' MB', '');
        row.appendChild(sizeMetric);

        card.appendChild(row);
    });

    return card;
}

function buildSampleStats(stats) {
    var card = document.createElement('div');
    card.className = 'model-family';

    var title = document.createElement('div');
    title.className = 'family-title';
    title.style.color = '#2ecc71';
    title.textContent = 'Human Training Pipeline';
    card.appendChild(title);

    var items = [
        ['Total Samples', stats.total],
        ['Positive', stats.positive],
        ['Negative', stats.negative],
        ['Reviewed', stats.reviewed],
    ];

    if (stats.sources) {
        Object.keys(stats.sources).forEach(function(src) {
            items.push([src, stats.sources[src]]);
        });
    }

    items.forEach(function(item) {
        var row = document.createElement('div');
        row.className = 'hw-row';
        row.style.padding = '4px 8px';
        row.style.fontSize = '0.75rem';
        var lbl = document.createElement('span');
        lbl.className = 'hw-label';
        lbl.textContent = item[0];
        var val = document.createElement('span');
        val.className = 'hw-value';
        val.style.color = '#ccc';
        val.textContent = item[1];
        row.appendChild(lbl);
        row.appendChild(val);
        card.appendChild(row);
    });

    return card;
}

function buildVelocityRuns(runs) {
    var card = document.createElement('div');
    card.className = 'model-family';

    var title = document.createElement('div');
    title.className = 'family-title';
    title.style.color = '#f39c12';
    title.textContent = 'Velocity Tests';
    var badge = document.createElement('span');
    badge.className = 'family-badge badge-count';
    badge.textContent = runs.length + ' runs';
    title.appendChild(badge);
    card.appendChild(title);

    runs.forEach(function(run) {
        var row = document.createElement('div');
        row.className = 'version-row';

        var name = document.createElement('div');
        name.className = 'version-name';
        name.style.color = '#f39c12';
        name.style.width = '140px';
        name.textContent = run.run_id;
        row.appendChild(name);

        if (run.summary) {
            var profiles = Object.keys(run.summary);
            var totalAcc = 0, accCount = 0;
            profiles.forEach(function(p) {
                if (run.summary[p].accuracy_pct !== undefined) {
                    totalAcc += run.summary[p].accuracy_pct;
                    accCount++;
                }
            });
            var avgAcc = accCount > 0 ? totalAcc / accCount : 0;

            var accMetric = makeMetric('Avg Accuracy',
                avgAcc.toFixed(1) + '%',
                avgAcc > 90 ? 'good' : avgAcc > 70 ? 'ok' : 'bad');
            row.appendChild(accMetric);

            var profMetric = makeMetric('Profiles', profiles.length + ' tested', '');
            row.appendChild(profMetric);
        }

        card.appendChild(row);
    });

    return card;
}

function buildLossEventSummary(data) {
    var card = document.createElement('div');
    card.className = 'model-family';

    var title = document.createElement('div');
    title.className = 'family-title';
    title.style.color = '#e74c3c';
    title.textContent = 'Loss Events';
    var badge = document.createElement('span');
    badge.className = 'family-badge badge-count';
    badge.textContent = data.loss_event_count + ' events';
    title.appendChild(badge);
    card.appendChild(title);

    var link = document.createElement('a');
    link.href = '/loss_events';
    link.style.cssText = 'color:#e74c3c;font-size:0.75rem;text-decoration:underline';
    link.textContent = 'View all loss events \u2192';
    card.appendChild(link);

    return card;
}

function makeMetric(label, value, cls) {
    var div = document.createElement('div');
    div.className = 'version-metric';
    var lbl = document.createElement('div');
    lbl.className = 'metric-label';
    lbl.textContent = label;
    var val = document.createElement('div');
    val.className = 'metric-value' + (cls ? ' ' + cls : '');
    val.textContent = value;
    div.appendChild(lbl);
    div.appendChild(val);
    return div;
}

loadModels();
