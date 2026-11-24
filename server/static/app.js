const $ = (s) => document.querySelector(s);
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};
let offset = 0, current = null, defaultConf = 0.5;
const PAGE = 48;

function headers() { const k = store.get('sc_key', ''); return k ? { 'X-API-Key': k } : {}; }
function toast(msg) { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 3000); }
function fmtTime(iso) { return new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }); }
function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(1) + ' s' : Math.round(ms) + ' ms'; }

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { ...headers(), ...(opts.headers || {}) } });
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch {} throw new Error(m); }
  return r;
}

async function loadHealth() {
  try {
    const h = await (await api('api/health')).json();
    defaultConf = h.default_conf;
    $('#modelInfo').textContent = `Model: ${h.model} · ${h.imgsz}px · ${h.providers[0].replace('ExecutionProvider', '')}`;
    if (!store.get('sc_conf')) setConf(defaultConf);
  } catch (e) { toast('Server not reachable: ' + e.message); }
}
function setConf(v) { $('#conf').value = v; $('#confOut').textContent = Number(v).toFixed(2); }

function itemEl(r) {
  const li = document.createElement('li');
  li.className = 'item';
  li.tabIndex = 0;
  li.innerHTML = `<img loading="lazy" alt="">
    <div class="meta"><div class="n"></div><div class="fn"></div><div class="t"></div></div>`;
  li.querySelector('img').src = r.thumbnail_url.replace(/^\//, '');
  li.querySelector('.n').textContent = r.count;
  li.querySelector('.fn').textContent = r.filename || r.id;
  li.querySelector('.t').textContent = `${fmtTime(r.created_at)} · ${fmtMs(r.ms)}`;
  if (r.source && r.source !== 'web') { const b = document.createElement('span'); b.className = 'badge'; b.textContent = r.source; li.querySelector('.t').append(b); }
  const open = () => openViewer(r);
  li.addEventListener('click', open);
  li.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  return li;
}

async function loadHistory(reset = true) {
  if (reset) offset = 0;
  const data = await (await api(`api/results?limit=${PAGE}&offset=${offset}`)).json();
  if (reset) $('#history').replaceChildren();
  data.items.forEach((r) => $('#history').append(itemEl(r)));
  offset += data.items.length;
  $('#empty').hidden = data.stats.images > 0;
  $('#more').hidden = offset >= data.stats.images;
  $('#sImages').textContent = data.stats.images.toLocaleString();
  $('#sProducts').textContent = data.stats.products.toLocaleString();
  $('#sMs').textContent = data.stats.images ? fmtMs(data.stats.avg_ms) : '–';
}

function showLast(r) {
  $('#last').hidden = false;
  $('#lastImg').src = r.overlay_url.replace(/^\//, '') + '?t=' + Date.now();
  $('#lastImg').onclick = () => openViewer(r);
  $('#lastCount').textContent = r.count;
  $('#lastInfo').textContent = `${r.filename} · ${fmtMs(r.ms)}`;
}

async function upload(files) {
  files = [...files].filter((f) => f.type.startsWith('image/') || /\.(jpe?g|png|webp|bmp)$/i.test(f.name));
  if (!files.length) return toast('Please choose image files');
  const conf = store.get('sc_conf', '');
  $('#progress').hidden = false;
  let done = 0, failed = 0;
  for (const f of files) {
    $('#progressText').textContent = `Counting ${done + 1} of ${files.length}: ${f.name}`;
    $('#bar').style.width = (done / files.length * 100) + '%';
    const fd = new FormData();
    fd.append('file', f, f.name);
    if (conf) fd.append('conf', conf);
    try {
      const r = await (await api('api/count', { method: 'POST', body: fd })).json();
      showLast(r);
      $('#history').prepend(itemEl(r));
      $('#empty').hidden = true;
    } catch (e) { failed++; toast(`${f.name}: ${e.message}`); }
    done++;
  }
  $('#bar').style.width = '100%';
  $('#progressText').textContent = `Done: ${done - failed} counted${failed ? `, ${failed} failed` : ''}`;
  setTimeout(() => { $('#progress').hidden = true; }, 2500);
  loadHistory();
}

function openViewer(r) {
  current = r;
  $('#vCount').textContent = `${r.count} products`;
  $('#vMeta').textContent = `· ${r.filename} · ${fmtTime(r.created_at)} · ${fmtMs(r.ms)}`;
  renderViewer();
  $('#viewer').showModal();
}
function renderViewer() {
  if (!current) return;
  const u = $('#vOverlay').checked ? current.overlay_url : current.image_url;
  $('#vImg').src = u.replace(/^\//, '');
}

// events
const drop = $('#drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => upload(e.dataTransfer.files));
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file').click(); } });
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => { e.preventDefault(); if (e.target.closest('#drop') === null && e.dataTransfer.files.length) upload(e.dataTransfer.files); });
$('#file').addEventListener('change', (e) => { upload(e.target.files); e.target.value = ''; });
$('#more').addEventListener('click', () => loadHistory(false));
$('#vOverlay').addEventListener('change', renderViewer);
$('#vDelete').addEventListener('click', async () => {
  if (!current || !confirm('Delete this result?')) return;
  await api(`api/results/${current.id}`, { method: 'DELETE' });
  $('#viewer').close(); toast('Deleted'); loadHistory();
});
$('#settingsBtn').addEventListener('click', () => { setConf(store.get('sc_conf', defaultConf)); $('#apiKey').value = store.get('sc_key', ''); $('#settings').showModal(); });
$('#conf').addEventListener('input', (e) => { $('#confOut').textContent = Number(e.target.value).toFixed(2); store.set('sc_conf', e.target.value); });
$('#apiKey').addEventListener('change', (e) => { store.set('sc_key', e.target.value.trim()); loadHistory(); });
$('#exportBtn').addEventListener('click', async (e) => {
  if (!store.get('sc_key', '')) return; // plain link works without a key
  e.preventDefault();
  const blob = await (await api('api/export.csv')).blob();
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'shelfcount.csv' });
  a.click();
});

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
loadHealth().then(() => loadHistory()).catch((e) => toast(e.message));
