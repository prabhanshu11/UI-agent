/* yolo.js — YOLO Gallery page logic */
var allEvents = [];
var currentEvent = null;
var currentFrames = [];
var currentIdx = -1;

function loadEvents() {
  fetch('/api/yolo/frames')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      allEvents = data.events || [];
      renderSidebar();
      if (allEvents.length > 0) selectEvent(allEvents[0].event_id);
    });
}

function renderSidebar() {
  var sb = document.getElementById('sidebar');
  while (sb.firstChild) sb.removeChild(sb.firstChild);

  if (allEvents.length === 0) {
    var empty = document.createElement('div');
    empty.style.cssText = 'padding:2rem 1rem;color:#555;font-size:0.75rem;text-align:center';
    empty.textContent = 'No loss events found';
    sb.appendChild(empty);
    return;
  }

  allEvents.forEach(function(ev) {
    var group = document.createElement('div');
    group.className = 'event-group';

    var title = document.createElement('div');
    title.className = 'event-title';
    if (currentEvent === ev.event_id) title.className += ' active';
    title.textContent = ev.event_id + ' (' + ev.count + ')';
    title.setAttribute('data-eid', ev.event_id);
    title.addEventListener('click', function() { selectEvent(ev.event_id); });
    group.appendChild(title);

    var flist = document.createElement('div');
    flist.className = 'frame-list' + (currentEvent === ev.event_id ? ' open' : '');
    ev.frames.forEach(function(fname, i) {
      var item = document.createElement('div');
      item.className = 'frame-item';
      if (currentEvent === ev.event_id && currentIdx === i) item.className += ' selected';
      item.textContent = fname;
      item.addEventListener('click', function() { selectFrame(ev.event_id, i); });
      flist.appendChild(item);
    });
    group.appendChild(flist);
    sb.appendChild(group);
  });
}

function selectEvent(eid) {
  var ev = allEvents.find(function(e) { return e.event_id === eid; });
  if (!ev) return;
  currentEvent = eid;
  currentFrames = ev.frames;
  renderSidebar();
  renderFilmstrip();
  selectFrame(eid, 0);
}

function renderFilmstrip() {
  var strip = document.getElementById('filmstrip');
  while (strip.firstChild) strip.removeChild(strip.firstChild);
  currentFrames.forEach(function(fname, i) {
    var thumb = document.createElement('div');
    thumb.className = 'thumb' + (i === currentIdx ? ' selected' : '');
    var img = document.createElement('img');
    img.src = '/api/loss_events/' + currentEvent + '/frame/frames/' + fname;
    img.alt = fname;
    thumb.appendChild(img);
    thumb.addEventListener('click', function() { selectFrame(currentEvent, i); });
    strip.appendChild(thumb);
  });
}

function selectFrame(eid, idx) {
  currentEvent = eid;
  currentIdx = idx;
  var fname = currentFrames[idx];
  if (!fname) return;

  // Update viewer with YOLO-annotated image
  var viewer = document.getElementById('viewer');
  while (viewer.firstChild) viewer.removeChild(viewer.firstChild);
  var img = document.createElement('img');
  img.src = '/api/yolo/detect/' + eid + '/' + fname;
  img.alt = 'YOLO detection: ' + fname;
  viewer.appendChild(img);

  // Update filmstrip selection
  var thumbs = document.querySelectorAll('.filmstrip .thumb');
  thumbs.forEach(function(t, i) {
    t.className = 'thumb' + (i === idx ? ' selected' : '');
  });

  // Update sidebar selection
  var items = document.querySelectorAll('.frame-item');
  items.forEach(function(item) { item.classList.remove('selected'); });
  var activeItems = document.querySelectorAll('.frame-list.open .frame-item');
  if (activeItems[idx]) activeItems[idx].classList.add('selected');

  // Fetch JSON results for detail bar
  var bar = document.getElementById('detail-bar');
  while (bar.firstChild) bar.removeChild(bar.firstChild);
  var loadSpan = document.createElement('span');
  loadSpan.className = 'loading';
  loadSpan.textContent = 'Running YOLO...';
  bar.appendChild(loadSpan);

  fetch('/api/yolo/detect_json/' + eid + '/' + fname)
    .then(function(r) { return r.json(); })
    .then(function(data) {
      renderDetailBar(data, fname);
    });
}

function renderDetailBar(data, fname) {
  var bar = document.getElementById('detail-bar');
  while (bar.firstChild) bar.removeChild(bar.firstChild);

  addDetail(bar, 'Frame', fname);

  var dets = data.detections || [];
  if (dets.length === 0) {
    var nodet = document.createElement('span');
    nodet.className = 'det-none';
    nodet.textContent = 'No detection';
    bar.appendChild(nodet);
  } else {
    var d = dets[0];
    addDetail(bar, 'Position', d.cx + ', ' + d.cy);

    var confSpan = document.createElement('span');
    var confLabel = document.createElement('span');
    confLabel.className = 'label';
    confLabel.textContent = 'Conf:';
    confSpan.appendChild(confLabel);
    var confVal = document.createElement('span');
    confVal.className = 'det-conf';
    confVal.textContent = ' ' + (d.confidence * 100).toFixed(1) + '%';
    confSpan.appendChild(confVal);
    bar.appendChild(confSpan);

    addDetail(bar, 'Inference', d.inference_ms.toFixed(1) + 'ms');

    if (data.tracked_position) {
      var tp = data.tracked_position;
      var dist = Math.sqrt(Math.pow(d.cx - tp.x, 2) + Math.pow(d.cy - tp.y, 2));
      var distColor = dist < 30 ? '#2ecc71' : dist < 80 ? '#f39c12' : '#e74c3c';
      var distSpan = document.createElement('span');
      var distLabel = document.createElement('span');
      distLabel.className = 'label';
      distLabel.textContent = 'vs Tracked:';
      distSpan.appendChild(distLabel);
      var distVal = document.createElement('span');
      distVal.style.color = distColor;
      distVal.style.marginLeft = '4px';
      distVal.textContent = dist.toFixed(0) + 'px';
      distSpan.appendChild(distVal);
      bar.appendChild(distSpan);
    }
  }

  if (dets.length > 1) {
    addDetail(bar, 'Total', dets.length + ' detections');
  }
}

function addDetail(parent, label, value) {
  var span = document.createElement('span');
  var lbl = document.createElement('span');
  lbl.className = 'label';
  lbl.textContent = label + ':';
  span.appendChild(lbl);
  var val = document.createElement('span');
  val.className = 'value';
  val.textContent = ' ' + value;
  span.appendChild(val);
  parent.appendChild(span);
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
    e.preventDefault();
    if (currentIdx < currentFrames.length - 1) selectFrame(currentEvent, currentIdx + 1);
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (currentIdx > 0) selectFrame(currentEvent, currentIdx - 1);
  }
});

loadEvents();
