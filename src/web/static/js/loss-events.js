/* loss-events.js — Cursor Loss Events page logic
 * Note: innerHTML used with server-originated data only (internal dashboard, localhost). */
var events = [];
var currentTimeline = [];
var currentEventId = null;
var selectedIdx = 0;

async function loadEvents() {
  var res = await fetch('/api/loss_events');
  var data = await res.json();
  events = data.events;
  document.getElementById('ring-status').textContent =
    'Ring buffer: ' + data.ring_buffer_size + ' frames | ' +
    (data.active ? 'CAPTURING...' : 'watching');
  renderEventsList();
}

function renderEventsList() {
  var el = document.getElementById('events-list');
  // Clear previous content
  while (el.firstChild) el.removeChild(el.firstChild);

  if (events.length === 0) {
    var empty = document.createElement('div');
    empty.className = 'empty-state';
    var h2 = document.createElement('h2');
    h2.textContent = 'No loss events captured yet';
    var p = document.createElement('p');
    p.textContent = 'Events are recorded automatically when cursor confidence drops. ' +
      'The ring buffer holds the last 60 seconds \u2014 when a loss happens, it saves the timeline.';
    empty.appendChild(h2);
    empty.appendChild(p);
    el.appendChild(empty);
    return;
  }

  for (var i = 0; i < events.length; i++) {
    var ev = events[i];
    var card = document.createElement('div');
    card.className = 'event-card';
    card.setAttribute('data-event-id', ev.event_id);
    card.addEventListener('click', (function(eid) {
      return function() { openEvent(eid); };
    })(ev.event_id));

    var trigger = document.createElement('span');
    trigger.className = 'trigger';
    trigger.textContent = ev.trigger;
    card.appendChild(trigger);

    var meta = document.createElement('div');
    meta.className = 'meta';

    var timeSpan = document.createElement('span');
    timeSpan.textContent = ev.trigger_utc;
    meta.appendChild(timeSpan);

    var framesSpan = document.createElement('span');
    framesSpan.textContent = ev.total_frames + ' frames';
    meta.appendChild(framesSpan);

    var durSpan = document.createElement('span');
    durSpan.textContent = ev.duration_s + 's duration';
    meta.appendChild(durSpan);

    var recSpan = document.createElement('span');
    recSpan.className = ev.recovered ? 'recovered' : 'not-recovered';
    recSpan.textContent = ev.recovered ? 'Recovered' : 'Not recovered';
    meta.appendChild(recSpan);

    if (ev.phases) {
      var hadBadge = document.createElement('span');
      hadBadge.className = 'badge badge-had';
      hadBadge.textContent = ev.phases.had_it + ' had';
      meta.appendChild(hadBadge);

      var lostBadge = document.createElement('span');
      lostBadge.className = 'badge badge-lost';
      lostBadge.textContent = ev.phases.lost + ' lost';
      meta.appendChild(lostBadge);

      var recBadge = document.createElement('span');
      recBadge.className = 'badge badge-recovered';
      recBadge.textContent = (ev.phases.recovering + ev.phases.recovered) + ' recovery';
      meta.appendChild(recBadge);
    }

    card.appendChild(meta);
    el.appendChild(card);
  }
}

async function openEvent(eventId) {
  currentEventId = eventId;
  var res = await fetch('/api/loss_events/' + eventId + '/timeline');
  var data = await res.json();
  currentTimeline = data.timeline;

  document.getElementById('events-list').style.display = 'none';
  document.getElementById('timeline-view').classList.add('active');

  var m = data.manifest;
  var header = document.getElementById('tl-header');
  while (header.firstChild) header.removeChild(header.firstChild);
  var h2 = document.createElement('h2');
  h2.textContent = m.trigger;
  header.appendChild(h2);
  var metaDiv = document.createElement('div');
  metaDiv.className = 'tl-meta';
  metaDiv.textContent = m.trigger_utc + ' | ' + m.total_frames + ' frames | ' +
    m.duration_s + 's | ' + (m.recovered ? 'Recovered' : 'Not recovered');
  header.appendChild(metaDiv);

  renderStrip();
  selectFrame(0);
}

function renderStrip() {
  var strip = document.getElementById('tl-strip');
  while (strip.firstChild) strip.removeChild(strip.firstChild);

  for (var i = 0; i < currentTimeline.length; i++) {
    var t = currentTimeline[i];
    var confClass = t.live_cnn.confidence > 0.7 ? 'conf-high' : (t.live_cnn.confidence > 0.4 ? 'conf-mid' : 'conf-low');
    var src = t.frame ? '/api/loss_events/' + currentEventId + '/frame/' + t.frame : '';

    var frame = document.createElement('div');
    frame.className = 'tl-frame phase-' + t.phase + (i === selectedIdx ? ' selected' : '');
    frame.setAttribute('data-idx', i);
    frame.addEventListener('click', (function(idx) {
      return function() { selectFrame(idx); };
    })(i));

    if (src) {
      var img = document.createElement('img');
      img.src = src;
      img.loading = 'lazy';
      frame.appendChild(img);
    } else {
      var placeholder = document.createElement('div');
      placeholder.style.cssText = 'width:116px;height:65px;background:#111';
      frame.appendChild(placeholder);
    }

    var label = document.createElement('div');
    label.className = 'tl-label';
    label.textContent = t.t_offset_s + 's \u00B7 ' + t.phase;
    frame.appendChild(label);

    var conf = document.createElement('div');
    conf.className = 'tl-conf';
    var bar = document.createElement('span');
    bar.className = 'conf-bar ' + confClass;
    bar.style.width = Math.round(t.live_cnn.confidence * 60) + 'px';
    conf.appendChild(bar);
    conf.appendChild(document.createTextNode(' ' + (t.live_cnn.confidence * 100).toFixed(0) + '%'));
    frame.appendChild(conf);

    strip.appendChild(frame);
  }
}

function selectFrame(idx) {
  selectedIdx = idx;
  var t = currentTimeline[idx];

  // Update strip selection
  var frames = document.querySelectorAll('.tl-frame');
  frames.forEach(function(f, i) { f.classList.toggle('selected', i === idx); });
  if (frames[idx]) frames[idx].scrollIntoView({behavior: 'smooth', block: 'nearest', inline: 'center'});

  // Build detail panel using DOM methods
  var panel = document.getElementById('detail-panel');
  while (panel.firstChild) panel.removeChild(panel.firstChild);

  var frameSrc = t.frame ? '/api/loss_events/' + currentEventId + '/frame/' + t.frame : '';
  var patchSrc = t.patch ? '/api/loss_events/' + currentEventId + '/frame/' + t.patch : '';

  var liveConf = t.live_cnn.confidence;
  var retroConf = t.retrospective_cnn.confidence;
  function confColor(c) { return c > 0.7 ? '#2ecc71' : (c > 0.4 ? '#f39c12' : '#e74c3c'); }

  // Frame image
  var frameDiv = document.createElement('div');
  frameDiv.className = 'detail-frame';
  if (frameSrc) {
    var fImg = document.createElement('img');
    fImg.src = frameSrc;
    frameDiv.appendChild(fImg);
  } else {
    var fPlaceholder = document.createElement('div');
    fPlaceholder.style.cssText = 'height:300px;background:#111;border-radius:4px';
    frameDiv.appendChild(fPlaceholder);
  }
  panel.appendChild(frameDiv);

  // Patch image
  var patchDiv = document.createElement('div');
  patchDiv.className = 'detail-patch';
  var patchLabel = document.createElement('div');
  patchLabel.style.cssText = 'font-size:0.65rem;color:#888;margin-bottom:4px';
  patchLabel.textContent = '64x64 patch at (' + t.cursor.x + ',' + t.cursor.y + ')';
  patchDiv.appendChild(patchLabel);
  if (patchSrc) {
    var pImg = document.createElement('img');
    pImg.src = patchSrc;
    patchDiv.appendChild(pImg);
  } else {
    var pPlaceholder = document.createElement('div');
    pPlaceholder.style.cssText = 'width:192px;height:192px;background:#111;border-radius:4px';
    patchDiv.appendChild(pPlaceholder);
  }
  panel.appendChild(patchDiv);

  // Info table
  var infoDiv = document.createElement('div');
  infoDiv.className = 'detail-info';
  var table = document.createElement('table');

  function addRow(label, value, style) {
    var tr = document.createElement('tr');
    var td1 = document.createElement('td');
    td1.textContent = label;
    tr.appendChild(td1);
    var td2 = document.createElement('td');
    if (style) td2.style.cssText = style;
    if (typeof value === 'string') {
      td2.textContent = value;
    } else {
      td2.appendChild(value);
    }
    tr.appendChild(td2);
    table.appendChild(tr);
  }

  addRow('Time offset', t.t_offset_s + 's');
  var phaseColor = t.phase === 'had_it' ? '#2ecc71' : t.phase === 'lost' ? '#e74c3c' : '#f39c12';
  addRow('Phase', t.phase, 'color:' + phaseColor);
  addRow('Position', '(' + t.cursor.x + ', ' + t.cursor.y + ')');
  addRow('Method', t.cursor.method);

  var valSpan = document.createElement('span');
  valSpan.style.color = t.cursor.validated ? '#2ecc71' : '#e74c3c';
  valSpan.textContent = t.cursor.validated ? 'YES' : 'NO';
  addRow('Validated', valSpan);

  var liveSpan = document.createElement('span');
  liveSpan.style.color = confColor(liveConf);
  liveSpan.textContent = (liveConf * 100).toFixed(1) + '% (' + t.live_cnn.model + ')';
  addRow('Live CNN', liveSpan);

  var retroSpan = document.createElement('span');
  retroSpan.style.color = confColor(retroConf);
  retroSpan.textContent = (retroConf * 100).toFixed(1) + '% (' + t.retrospective_cnn.inference_ms + 'ms)';
  addRow('Retro CNN', retroSpan);

  addRow('Silhouette', (t.silhouette.confidence * 100).toFixed(0) + '% \u00B7 ' +
    t.silhouette.method + ' \u00B7 ROI ' + t.silhouette.roi_size + ' \u00B7 misses ' + t.silhouette.misses);
  addRow('Blobs', String(t.blob_count));
  addRow('Velocity', (t.velocity_px_s || 0) + ' px/s');
  addRow('Acceleration', (t.acceleration_px_s2 || 0) + ' px/s\u00B2');

  if (t.edge_cases && t.edge_cases.length > 0) {
    var ecContainer = document.createElement('span');
    t.edge_cases.forEach(function(c) {
      var ecSpan = document.createElement('span');
      ecSpan.style.cssText = 'background:#3a1a2a;color:#e74c3c;padding:1px 6px;border-radius:8px;font-size:0.6rem;margin-right:4px';
      ecSpan.textContent = c;
      ecContainer.appendChild(ecSpan);
    });
    addRow('Edge cases', ecContainer);
  }

  infoDiv.appendChild(table);

  // Blob candidates with CNN scores
  if (t.blob_candidates && t.blob_candidates.length > 0) {
    var bcHeader = document.createElement('div');
    bcHeader.style.cssText = 'margin-top:8px;font-size:0.65rem;color:#888';
    bcHeader.textContent = 'Blob candidates (CNN scored):';
    infoDiv.appendChild(bcHeader);

    var bcRow = document.createElement('div');
    bcRow.style.cssText = 'display:flex;gap:4px;flex-wrap:wrap;margin-top:4px';
    for (var bi = 0; bi < t.blob_candidates.length; bi++) {
      var bc = t.blob_candidates[bi];
      var bcColor = bc.cnn_score > 0.7 ? '#2ecc71' : (bc.cnn_score > 0.4 ? '#f39c12' : '#e74c3c');
      var bcSrc = '/api/loss_events/' + currentEventId + '/frame/' + bc.patch;

      var bcDiv = document.createElement('div');
      bcDiv.style.textAlign = 'center';
      var bcImg = document.createElement('img');
      bcImg.src = bcSrc;
      bcImg.style.cssText = 'width:64px;height:64px;image-rendering:pixelated;border:1px solid #333;border-radius:2px';
      bcDiv.appendChild(bcImg);
      var bcLabel = document.createElement('div');
      bcLabel.style.cssText = 'font-size:0.55rem;color:' + bcColor;
      bcLabel.textContent = (bc.cnn_score * 100).toFixed(0) + '% (' + bc.centroid[0] + ',' + bc.centroid[1] + ')';
      bcDiv.appendChild(bcLabel);
      bcRow.appendChild(bcDiv);
    }
    infoDiv.appendChild(bcRow);
  }

  panel.appendChild(infoDiv);
}

function showList() {
  document.getElementById('events-list').style.display = '';
  document.getElementById('timeline-view').classList.remove('active');
}

// Keyboard nav: left/right to scrub timeline
document.addEventListener('keydown', function(e) {
  if (!document.getElementById('timeline-view').classList.contains('active')) return;
  if (e.key === 'ArrowLeft' || e.key === 'j') { e.preventDefault(); if (selectedIdx > 0) selectFrame(selectedIdx - 1); }
  if (e.key === 'ArrowRight' || e.key === 'k') { e.preventDefault(); if (selectedIdx < currentTimeline.length - 1) selectFrame(selectedIdx + 1); }
});

loadEvents();
setInterval(loadEvents, 10000);
