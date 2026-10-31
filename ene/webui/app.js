/* .ene GUI - vanilla JS, no build step. API: ene/serve.py */
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtT = ms => { if (ms == null) return ''; const s = Math.floor(ms / 1000), h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), ss = s % 60; return (h ? h + ':' + String(m).padStart(2, '0') : m) + ':' + String(ss).padStart(2, '0'); };
const fmtB = n => n == null ? '' : n < 1024 ? n + ' B' : n < 1048576 ? (n / 1024).toFixed(1) + ' KB' : n < 1073741824 ? (n / 1048576).toFixed(1) + ' MB' : (n / 1073741824).toFixed(2) + ' GB';
const fmtD = iso => iso ? new Date(iso).toLocaleString('en-US', { dateStyle: 'short', timeStyle: 'short' }) : '';
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const isDark = () => document.documentElement.dataset.theme === 'dark' || (document.documentElement.dataset.theme !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches);

async function api(path, opt) {
  const r = await fetch(path, opt);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch { } throw new Error(m); }
  return (r.headers.get('content-type') || '').includes('json') ? r.json() : r.text();
}
let toastT; function toast(msg) { const t = $('#toast'); t.textContent = msg; t.style.display = 'block'; clearTimeout(toastT); toastT = setTimeout(() => t.style.display = 'none', 3500); }

const S = { titles: [], detail: null, search: null, pending: null, status: null };

/* ------------------------------------------------------------------ theme */
const THEMES = ['', 'light', 'dark'];
function applyTheme(t) { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; try { localStorage.setItem('ene-theme', t); } catch { } TL.draw(); }
$('#theme-btn').onclick = () => { const cur = document.documentElement.dataset.theme || ''; applyTheme(THEMES[(THEMES.indexOf(cur) + 1) % 3]); };
try { applyTheme(localStorage.getItem('ene-theme') || ''); } catch { }
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => TL.draw());

/* ------------------------------------------------------------------ sequential palette (blue 100->700; in dark mode the anchor flips) */
const SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'].map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)));
function seq(v) {
  const ramp = isDark() ? [...SEQ].reverse() : SEQ;
  const x = Math.max(0, Math.min(1, v)) * (ramp.length - 1), i = Math.floor(x), f = x - i, a = ramp[i], b = ramp[Math.min(i + 1, ramp.length - 1)];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}

/* ------------------------------------------------------------------ router */
function route() {
  const h = location.hash || '#/library';
  const [path, qs] = h.split('?'), [, view, id] = path.split('/');
  const params = new URLSearchParams(qs || '');   // #/title/<id>?q=text&dim=256 -> search straight from the link; &theme=light|dark
  if (params.has('theme')) applyTheme(params.get('theme'));
  $$('.view').forEach(v => v.classList.toggle('on', v.id === 'view-' + view));
  $$('.nav a').forEach(a => a.classList.toggle('on', a.dataset.view === view));
  if (view === 'library') { loadLibrary(); if (params.get('q')) { QB_LIB.set({ text: params.get('q'), dim: params.get('dim'), k: params.get('k') }); QB_LIB.run(); } }
  if (view === 'build') loadBuild();
  if (view === 'title' && id && (!S.detail || S.detail.summary.id !== id)) { if (params.get('q')) S.pending = { text: params.get('q'), dim: params.get('dim'), k: params.get('k') }; openTitle(id); }
  else if (view === 'title' && !id && S.detail) location.hash = '#/title/' + S.detail.summary.id;
}
addEventListener('hashchange', route);

/* ------------------------------------------------------------------ server status */
async function pollStatus() {
  try {
    const st = await api('/api/status'); S.status = st;
    const m = st.model, dot = $('#model-pill .dot'), txt = $('#model-pill span:last-child');
    dot.className = 'dot ' + (m.state === 'loaded' ? 'ok' : m.state === 'loading' ? 'busy' : m.state === 'error' ? 'err' : '');
    txt.textContent = m.state === 'loaded' ? `${m.name} · ${m.device} ${m.dtype} · VRAM ${m.vram_gib} GiB` : m.state === 'error' ? 'model: ' + m.error : 'model: ' + m.state;
    $('#space-note').textContent = m.state === 'loaded' ? `text, image or audio → ${m.name}, ${m.dim} d` : 'text, image or audio → one vector space';
    setWidths(st.widths && st.widths.length ? st.widths : (m.widths || []));
    $('#lib-pill span').textContent = `${st.titles} titles · v${st.version}`;
    $('#roots-line').textContent = 'Scanned roots: ' + st.roots.join(' ; ');
    const busy = m.state === 'loading' || st.jobs.some(j => j.status === 'running');
    setTimeout(pollStatus, busy ? 2000 : 15000);
  } catch (e) { $('#model-pill span:last-child').textContent = 'server unavailable'; setTimeout(pollStatus, 5000); }
}

/* the read-time widths the files offer (their spaces' Matryoshka ladders, from /api/status); '' = every section's native width */
function widthOptions() { return '<option value="">native</option>' + (S.widths || []).map(w => `<option>${w}</option>`).join(''); }
function setDim(sel, dim) { const v = dim ? String(dim) : ''; if (v && ![...sel.options].some(o => o.value === v)) sel.add(new Option(v, v)); sel.value = v; }
function setWidths(ws) {
  if (JSON.stringify(ws) === JSON.stringify(S.widths || [])) return;
  S.widths = ws;
  $$('select[name=dim]').forEach(sel => { const cur = sel.value; sel.innerHTML = widthOptions(); setDim(sel, cur); });
}

/* ------------------------------------------------------------------ query box (text + image + audio) */
function queryBox(root, { scope, onResult, getTitle }) {
  root.innerHTML = `<div class="qbox">
    <input type="text" placeholder="${scope === 'library' ? 'e.g. a giant rabbit takes revenge on three rodents' : 'e.g. flying squirrel, end credits, quiet music...'}">
    <label class="file-chip" data-kind="image"><input type="file" accept="image/*"><span>+ image</span><span class="x" hidden>×</span></label>
    <label class="file-chip" data-kind="audio"><input type="file" accept="audio/*,video/*"><span>+ audio</span><span class="x" hidden>×</span></label>
    <select name="dim" title="Matryoshka width at read time (native = each section's own)">${widthOptions()}</select>
    <select name="k" title="top-k per section"><option>5</option><option selected>8</option><option>12</option><option>20</option></select>
    <button class="btn">Search</button></div><div class="qinfo"></div>`;
  const q = $('input[type=text]', root), btn = $('.btn', root), info = $('.qinfo', root), files = { image: null, audio: null };
  $$('.file-chip', root).forEach(ch => {
    const inp = $('input', ch), lab = $('span', ch), x = $('.x', ch), kind = ch.dataset.kind;
    inp.onchange = () => { files[kind] = inp.files[0] || null; lab.textContent = files[kind] ? files[kind].name.slice(0, 24) : '+ ' + kind; ch.classList.toggle('has', !!files[kind]); x.hidden = !files[kind]; };
    x.onclick = e => { e.preventDefault(); inp.value = ''; inp.onchange(); };
  });
  async function run() {
    const fd = new FormData();
    fd.append('q', q.value); fd.append('scope', scope); fd.append('k', $('select[name=k]', root).value);
    const dim = $('select[name=dim]', root).value; if (dim) fd.append('dim', dim);
    if (scope === 'title') fd.append('title', getTitle());
    if (files.image) fd.append('image', files.image);
    if (files.audio) fd.append('audio', files.audio);
    if (!q.value.trim() && !files.image && !files.audio) { info.textContent = 'type a text or pick an image / an audio file'; return; }
    btn.disabled = true; info.textContent = S.status?.model?.state === 'loaded' ? 'searching...' : 'loading the model (the first time takes ~10 s)...';
    try {
      const res = await api('/api/search', { method: 'POST', body: fd });
      const qi = res.query; info.textContent = `query ${qi.kind}${qi.audio_s ? ` (${qi.audio_s} s)` : ''} · embedding ${qi.embed_ms} ms · dim ${qi.dim} · top-${qi.k}`;
      onResult(res, { text: q.value, image: files.image, audio: files.audio, dim: qi.dim, k: qi.k });
    } catch (e) { info.textContent = 'error: ' + e.message; }
    finally { btn.disabled = false; }
  }
  btn.onclick = run; q.onkeydown = e => { if (e.key === 'Enter') run(); };
  return {
    run, set(st) { q.value = st.text || ''; files.image = st.image || null; files.audio = st.audio || null; setDim($('select[name=dim]', root), st.dim); $('select[name=k]', root).value = st.k || 8;
      $$('.file-chip', root).forEach(ch => { const k = ch.dataset.kind; ch.classList.toggle('has', !!files[k]); $('span', ch).textContent = files[k] ? files[k].name.slice(0, 24) : '+ ' + k; $('.x', ch).hidden = !files[k]; }); }
  };
}

/* ------------------------------------------------------------------ library */
const QB_LIB = queryBox($('#qbox-library'), { scope: 'library', onResult: renderLibraryResults });
async function loadLibrary() {
  try { S.titles = await api('/api/library'); } catch (e) { toast('library: ' + e.message); return; }
  const tb = $('#library-table tbody'); tb.innerHTML = '';
  $('#library-empty').style.display = S.titles.length ? 'none' : 'block';
  for (const t of S.titles) {
    const tr = document.createElement('tr'); tr.className = 'click';
    const bpm = t.duration_ms ? t.bytes / (t.duration_ms / 60000) : 0;
    tr.innerHTML = `<td><b>${esc(t.name)}</b></td><td>${t.year ?? ''}</td><td class="r">${fmtT(t.duration_ms)}</td><td>${esc(t.video || '')}${t.has_video ? '' : ' <span class="muted small">(no source)</span>'}</td><td class="r">${t.packets}</td><td class="r">${t.versions}</td><td class="r">${t.n_sections}</td><td class="r">${t.n_vectors}</td><td class="r">${fmtB(t.bytes)}</td><td class="r">${fmtB(bpm)}</td><td>${fmtD(t.created)}</td><td class="mono small muted">${esc(t.file)}</td>`;
    tr.onclick = () => location.hash = '#/title/' + t.id;
    tb.appendChild(tr);
  }
}
function renderLibraryResults(res, qstate) {
  const box = $('#library-results'); box.innerHTML = '';
  if (!res.titles.length) { box.innerHTML = '<div class="empty">no titles</div>'; return; }
  const hi = Math.max(...res.titles.map(t => t.score)), lo = Math.min(...res.titles.map(t => Math.min(...Object.values(t.roles))));
  for (const t of res.titles) {
    const d = document.createElement('div'); d.className = 'rank';
    const b = t.best, thumb = t.has_video && b && b.t0 != null ? `<img loading="lazy" src="/api/title/${t.id}/frame?t=${thumbT(b.t0, b.t1)}&w=256" alt="">` : '<div class="noview small">no video</div>';
    const roles = Object.entries(t.roles).sort((a, b) => b[1] - a[1]).map(([r, v]) => `<span class="lv"><i style="background:${seq(hi > lo ? (v - lo) / (hi - lo) : 1)}"></i>${esc(r)} ${v.toFixed(3)}</span>`).join('');
    d.innerHTML = `${thumb}<div><div class="row" style="justify-content:space-between"><span class="ttl">${esc(t.name)} <span class="muted" style="font-weight:400">${t.year ?? ''}</span></span><span class="num"><b>${t.score.toFixed(3)}</b></span></div>
      <div class="bar" style="margin:4px 0 6px"><i style="width:${Math.round(100 * t.score / hi)}%"></i></div>
      <div>${roles}</div>
      <div class="small ink2" style="margin-top:4px">best moment: <b>${esc(b.section)}</b> ${b.t0 != null ? `${fmtT(b.t0)}–${fmtT(b.t1)}` : ''} ${b.text ? '· "' + esc(b.text.slice(0, 120)) + '"' : ''}</div></div>`;
    d.onclick = () => { S.pending = { ...qstate, seek: b.t0, sec: b.section, i: b.i }; location.hash = '#/title/' + t.id; };
    box.appendChild(d);
  }
}

/* ------------------------------------------------------------------ title */
const QB_TITLE = queryBox($('#qbox-title'), { scope: 'title', getTitle: () => S.detail.summary.id, onResult: renderTitleResults });
async function openTitle(id) {
  let d; try { d = await api('/api/title/' + id); } catch (e) { toast('title: ' + e.message); location.hash = '#/library'; return; }
  S.detail = d; S.search = null;
  const s = d.summary, desc = d.description || {};
  const src = (Object.values(desc.versions || {})[0] || {}).source || {};
  $('#nav-title').style.display = ''; $('#nav-title').href = '#/title/' + id; $('#nav-title').textContent = s.name;
  $('#t-title').textContent = s.name + (s.year ? ` (${s.year})` : '');
  $('#t-sub').innerHTML = `title ${esc(s.title_id)} · ${esc(s.kind || '')}${s.ids?.imdb ? ` · <a target="_blank" href="https://www.imdb.com/title/${esc(s.ids.imdb)}/">IMDb ${esc(s.ids.imdb)}</a>` : ''} · axis ${s.axis_fps[0]}/${s.axis_fps[1]} fps · ${esc(s.path)}`;
  const vid = src.video || {};
  const stats = [[fmtT(s.duration_ms), 'duration'], [vid.width ? `${vid.width}×${vid.height}` : '—', `video${vid.fps ? ' @ ' + esc(vid.fps) : ''}`], [fmtB(s.bytes), '.ene'], [fmtB(s.duration_ms ? s.bytes / (s.duration_ms / 60000) : 0), 'per minute'], [s.packets, 'packets'], [s.versions, 'versions'], [s.n_sections, 'sections'], [s.n_vectors, 'rows'], [fmtB(src.size_bytes), 'source']];
  $('#t-stats').innerHTML = stats.map(([v, l]) => `<span class="stat"><span class="v num">${esc(v)}</span><span class="l">${esc(l)}</span></span>`).join('');
  const tagSec = d.sections.find(x => x.role === 'tag' && x.part === 'title'), synSec = d.sections.find(x => x.role === 'synopsis' && x.part === 'title');
  $('#t-tags').innerHTML = (tagSec?.texts || []).map(t => `<span class="chip">${esc(t)}</span>`).join('') + (synSec?.texts?.[0] ? `<span class="small ink2" style="flex-basis:100%">${esc(synSec.texts[0])}</span>` : '');
  // player
  const p = $('#player');
  if (d.video) { p.innerHTML = `<video controls preload="metadata" src="/api/title/${id}/video"></video>`; const v = $('video', p); v.addEventListener('timeupdate', () => { TL.needle = v.currentTime * 1000; TL.draw(); }); v.addEventListener('seeking', () => { TL.needle = v.currentTime * 1000; TL.draw(); }); }
  else p.innerHTML = '<div class="noview">the video source is not next to the .ene file</div>';
  $('#player-line').textContent = d.video ? d.video : (src.filename ? 'source named in the description: ' + src.filename : '');
  // sections
  const prepStr = pr => pr ? Object.entries(pr).filter(([k]) => k !== 'profile').map(([k, v]) => `${k} ${Array.isArray(v) ? v.join('×') : v ?? '∅'}`).join(', ') : '';
  $('#t-sections tbody').innerHTML = d.sections.map(x => `<tr><td class="mono">${esc(x.id)}</td><td>${esc(x.part)}</td><td>${esc(x.role)}</td><td>${esc(x.modality || '')}</td><td>${esc(x.lang || '')}</td><td class="mono small">${esc(x.space)}</td><td class="mono small muted">${esc(x.version.slice(0, 8))}</td><td class="r">${x.count}</td><td class="r">${x.dim}${x.dim_real && x.dim_real !== x.dim ? ' / ' + x.dim_real : ''}</td><td>${esc(x.quant)}</td><td>${esc(x.codec)}${x.shared ? ' <span class="muted small">(shared)</span>' : ''}</td><td class="r">${x.bytes.toLocaleString('en-US')}</td><td class="r">${(x.bytes / Math.max(x.count, 1)).toFixed(0)}</td><td class="r">${x.rev}</td><td class="mono small">${esc(x.ref || '')}</td><td class="small muted" title="${esc(JSON.stringify(x.prep || null))}">${esc(prepStr(x.prep))}</td><td>${x.has_text ? '✓' : ''}</td></tr>`).join('');
  $('#t-packets tbody').innerHTML = d.packets.map(g => `<tr><td>${g.index}</td><td class="r">${g.offset.toLocaleString('en-US')}</td><td class="r">${g.length.toLocaleString('en-US')}</td><td class="r">${g.versions}</td><td class="r">${g.sections}</td><td style="color:${g.hash_ok === false ? 'var(--critical)' : 'var(--good)'}">${g.hash_ok === false ? '✗ ERROR' : g.hash_ok ? '✓ OK' : 'not verified'}</td><td>${g.one_block ? 'yes' : ''}</td><td>${g.description ? 'yes' : ''}</td></tr>`).join('');
  $('#t-description').textContent = JSON.stringify(desc, null, 2);
  $('#t-results').innerHTML = ''; $('#t-qsummary').innerHTML = '';
  // benchmark
  $('#t-bench-card').style.display = d.bench ? '' : 'none';
  if (d.bench) api(`/api/title/${id}/bench`).then(t => $('#t-bench').innerHTML = md(t)).catch(() => { });
  TL.init(d);
  if (S.pending) { const pq = S.pending; S.pending = null; QB_TITLE.set(pq); await QB_TITLE.run(); if (pq.seek != null) seekTo(pq.seek, pq.sec, pq.i); }
}
// a single frame (one frame long) -> +300 ms, so the thumbnail does not land on the last frame of the previous scene;
// long rows (chapter, title) -> the middle frame
const thumbT = (t0, t1) => (t1 == null || t1 - t0 <= 100 ? t0 + 300 : t1 - t0 <= 30000 ? t0 : (t0 + t1) / 2) / 1000;
function seekTo(ms, sid, i) {
  const v = $('#player video');
  if (v) { v.currentTime = ms / 1000; v.play().catch(() => { }); }
  TL.needle = ms; TL.sel = sid != null ? { sid, i } : null; TL.draw();
  $$('#t-results .hit').forEach(h => h.classList.toggle('sel', h.dataset.sec === sid && +h.dataset.i === i));
}
function renderTitleResults(res) {
  S.search = res; TL.setScores(res);
  const id = S.detail.summary.id, hasVid = !!S.detail.video;
  const box = $('#t-results'); box.innerHTML = '';
  const best = res.sections.reduce((a, s) => s.max > (a?.max ?? -1) ? s : a, null);
  $('#t-qsummary').innerHTML = `Highest score: <b>${best.max.toFixed(3)}</b> in <b>${esc(best.id)}</b> (${esc(best.role)} / ${esc(best.modality)}). Scoring took ${res.sections.reduce((a, s) => a + s.ms, 0).toFixed(1)} ms for ${res.sections.reduce((a, s) => a + s.count, 0)} rows, dim ${res.query.dim}.`;
  for (const s of res.sections) {
    const c = document.createElement('div'); c.className = 'sec';
    c.innerHTML = `<h3><span>${esc(s.id)}</span><span class="mod">${esc(s.part)} · ${esc(s.role)} / ${esc(s.modality)} · ${s.count} rows · ${s.ms} ms · max ${s.max.toFixed(3)}</span></h3>` + s.top.map((h, j) => {
      const th = hasVid && h.t0 != null ? `<img ${j < 4 ? '' : 'loading="lazy"'} src="/api/title/${id}/frame?t=${thumbT(h.t0, h.t1)}&w=168" alt="">` : '';
      return `<div class="hit ${th ? '' : 'nothumb'}" data-sec="${esc(s.id)}" data-i="${h.i}">${th}<div><span class="s num">${h.score.toFixed(3)}</span><span class="t">${h.t0 != null ? fmtT(h.t0) + (h.t1 - h.t0 > 100 ? '–' + fmtT(h.t1) : '') : '#' + h.i}</span>${h.text ? `<div class="x">${esc(h.text)}</div>` : ''}</div></div>`;
    }).join('');
    $$('.hit', c).forEach(h => h.onclick = () => { const t = s.top.find(x => x.i === +h.dataset.i); seekTo(t.t0 ?? 0, s.id, t.i); });
    box.appendChild(c);
  }
}

/* ------------------------------------------------------------------ timeline (canvas) */
const TL = {
  canvas: $('#tl'), tip: $('#tl-tip'), rows: [], dur: 1, scores: null, needle: 0, hover: null, sel: null, L: 128, R: 12, T: 6, H: 22, G: 3, A: 24,
  init(d) {
    this.dur = Math.max(1, d.summary.duration_ms); this.scores = null; this.needle = 0; this.hover = null; this.sel = null;
    this.rows = d.sections.filter(s => s.times && s.times.length).map(s => {
      const n = s.times.length, cells = new Array(n);
      for (let i = 0; i < n; i++) { const [t0, t1] = s.times[i]; const next = i + 1 < n ? s.times[i + 1][0] : this.dur; cells[i] = [t0, t1 > t0 ? Math.min(t1, Math.max(next, t0 + 1)) : Math.max(next, t0 + 500)]; }
      return { id: s.id, part: s.part, role: s.role, modality: s.modality, times: s.times, texts: s.texts, cells };
    });
    this.draw();
  },
  setScores(res) { this.scores = {}; for (const s of res.sections) this.scores[s.id] = { v: s.scores, min: s.min, max: s.max, top: new Set(s.top.map(h => h.i)) }; this.draw(); },
  x(ms, W) { return this.L + (ms / this.dur) * (W - this.L - this.R); },
  draw() {
    const c = this.canvas; if (!c || !c.offsetParent) return;
    const W = c.clientWidth || 800, Hc = this.T + this.rows.length * (this.H + this.G) + this.A, dpr = devicePixelRatio || 1;
    if (c.width !== W * dpr || c.height !== Hc * dpr) { c.width = W * dpr; c.height = Hc * dpr; c.style.height = Hc + 'px'; }
    const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const col = { surface: cssVar('--surface'), grid: cssVar('--grid'), axis: cssVar('--axis'), muted: cssVar('--muted'), ink: cssVar('--ink'), ink2: cssVar('--ink-2'), needle: cssVar('--needle'), accent: cssVar('--accent') };
    g.fillStyle = col.surface; g.fillRect(0, 0, W, Hc);
    g.font = '12px ' + cssVar('--font'); g.textBaseline = 'middle';
    this.rows.forEach((r, ri) => {
      const y = this.T + ri * (this.H + this.G), sc = this.scores?.[r.id];
      g.fillStyle = col.ink2; g.textAlign = 'right'; g.fillText(r.id, this.L - 8, y + this.H / 2);
      const rng = sc ? Math.max(sc.max - sc.min, 1e-6) : 1;
      for (let i = 0; i < r.cells.length; i++) {
        const [a, b] = r.cells[i], x0 = this.x(a, W), x1 = this.x(b, W), w = x1 - x0;
        g.fillStyle = sc ? seq((sc.v[i] - sc.min) / rng) : col.grid;
        g.fillRect(w > 3 ? x0 + 1 : x0, y, w > 3 ? w - 2 : Math.max(w, 1), this.H);   // 2 px of surface between cells
      }
      if (sc) for (const i of sc.top) { const [a, b] = r.cells[i]; g.strokeStyle = col.ink; g.lineWidth = 1.5; g.strokeRect(this.x(a, W) + 1, y + 1, Math.max(this.x(b, W) - this.x(a, W) - 2, 1), this.H - 2); }
      for (const h of [this.hover, this.sel]) if (h && h.sid === r.id && r.cells[h.i]) { const [a, b] = r.cells[h.i]; g.strokeStyle = h === this.sel ? col.accent : col.ink; g.lineWidth = 2; g.strokeRect(this.x(a, W), y, Math.max(this.x(b, W) - this.x(a, W), 2), this.H); }
    });
    // time axis
    const y0 = this.T + this.rows.length * (this.H + this.G), span = this.dur / 1000, pxPerS = (W - this.L - this.R) / span;
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200].find(s => s * pxPerS >= 64) || 7200;
    g.strokeStyle = col.axis; g.lineWidth = 1; g.beginPath(); g.moveTo(this.L, y0 + 0.5); g.lineTo(W - this.R, y0 + 0.5); g.stroke();
    g.fillStyle = col.muted; g.textAlign = 'center';
    for (let s = 0; s <= span; s += step) { const x = this.x(s * 1000, W); g.beginPath(); g.moveTo(x + 0.5, y0); g.lineTo(x + 0.5, y0 + 4); g.stroke(); g.fillText(fmtT(s * 1000), x, y0 + 14); }
    // player needle
    if (this.needle > 0) { const x = this.x(this.needle, W); g.strokeStyle = col.needle; g.lineWidth = 2; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, y0); g.stroke(); g.fillStyle = col.needle; g.textAlign = 'left'; g.fillText(fmtT(this.needle), Math.min(x + 4, W - 40), y0 + 14); }
  },
  hit(ev) {
    const r = this.canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top, W = this.canvas.clientWidth;
    const ri = Math.floor((y - this.T) / (this.H + this.G)); if (ri < 0 || ri >= this.rows.length || x < this.L) return null;
    const ms = (x - this.L) / (W - this.L - this.R) * this.dur, row = this.rows[ri];
    let lo = 0, hi = row.cells.length - 1;   // binary search by t0
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (row.cells[m][0] <= ms) lo = m; else hi = m - 1; }
    return row.cells[lo] && ms >= row.cells[lo][0] && ms < row.cells[lo][1] ? { sid: row.id, i: lo, row, x, y } : null;
  },
};
TL.canvas.addEventListener('mousemove', ev => {
  const h = TL.hit(ev), tip = TL.tip;
  if (!h) { tip.style.display = 'none'; if (TL.hover) { TL.hover = null; TL.draw(); } return; }
  if (!TL.hover || TL.hover.sid !== h.sid || TL.hover.i !== h.i) {
    TL.hover = h; TL.draw();
    const [t0, t1] = h.row.times[h.i], sc = TL.scores?.[h.sid], id = S.detail.summary.id;
    tip.innerHTML = `<b>${esc(h.sid)}</b> #${h.i} · ${fmtT(t0)}${t1 - t0 > 100 ? '–' + fmtT(t1) : ''}${sc ? ` · score <b>${sc.v[h.i].toFixed(3)}</b>${sc.top.has(h.i) ? ' · top-k' : ''}` : ''}${h.row.texts ? `<div>${esc(h.row.texts[h.i])}</div>` : ''}${S.detail.video ? `<img src="/api/title/${id}/frame?t=${thumbT(t0, t1)}&w=200" alt="">` : ''}`;
    tip.style.display = 'block';
  }
  const wrap = TL.canvas.parentElement.getBoundingClientRect();
  tip.style.left = Math.min(h.x + 14, wrap.width - 330) + 'px'; tip.style.top = (h.y + 16) + 'px';
});
TL.canvas.addEventListener('mouseleave', () => { TL.tip.style.display = 'none'; TL.hover = null; TL.draw(); });
TL.canvas.addEventListener('click', ev => { const h = TL.hit(ev); if (h) seekTo(h.row.times[h.i][0], h.sid, h.i); });
new ResizeObserver(() => TL.draw()).observe(TL.canvas.parentElement);

/* ------------------------------------------------------------------ build */
let jobsTimer;
async function loadBuild() {
  try {
    const media = await api('/api/media'); const tb = $('#media-table tbody'); tb.innerHTML = '';
    for (const m of media) {
      const tr = document.createElement('tr'); tr.className = 'click';
      tr.innerHTML = `<td>${esc(m.file)}<div class="small muted">${esc(m.dir)}</div></td><td class="r">${fmtB(m.bytes)}</td><td>${m.ene ? `<a href="#/title/${m.ene}">✓ open</a>` : '—'}</td>`;
      tr.onclick = e => { if (e.target.tagName === 'A') return; const f = $('#build-form'); f.file.value = m.path; f.title.value = m.file.replace(/\.[^.]+$/, '').replace(/[._]+/g, ' ').replace(/\s*\{.*\}\s*$/, '').trim(); $$('#media-table tr').forEach(r => r.classList.remove('sel')); tr.classList.add('sel'); };
      tb.appendChild(tr);
    }
  } catch (e) { toast('media: ' + e.message); }
  pollJobs();
}
$('#build-btn').onclick = async () => {
  const f = $('#build-form'); if (!f.reportValidity()) return;
  const fd = new FormData(f); if (!f.no_sha256.checked) fd.delete('no_sha256'); else fd.set('no_sha256', 'true');
  for (const k of ['year', 'imdb', 'synopsis', 'tags', 'srt', 'out']) if (!fd.get(k)) fd.delete(k);
  $('#build-btn').disabled = true; $('#build-msg').textContent = 'starting...';
  try { const j = await api('/api/build', { method: 'POST', body: fd }); $('#build-msg').textContent = 'job ' + j.id + ' started'; pollJobs(); }
  catch (e) { $('#build-msg').textContent = 'error: ' + e.message; }
  finally { $('#build-btn').disabled = false; }
};
async function pollJobs() {
  clearTimeout(jobsTimer);
  let jobs; try { jobs = await api('/api/jobs'); } catch { return; }
  const box = $('#jobs'); box.innerHTML = jobs.length ? '' : '<div class="empty">no builds in this session</div>';
  for (const j of [...jobs].reverse()) {
    const d = document.createElement('div'); d.className = 'card'; d.style.marginBottom = '10px';
    const st = j.status === 'done' ? 'var(--good)' : j.status === 'error' ? 'var(--critical)' : 'var(--warning)';
    d.innerHTML = `<div class="row" style="justify-content:space-between"><div><b>${esc(j.params.title)}</b> <span class="small muted mono">${esc(j.params.file)}</span></div><div><span class="pill"><span class="dot" style="background:${st}"></span>${esc(j.status)} · ${j.elapsed_s} s</span> ${j.title_id ? `<a class="btn ghost" href="#/title/${j.title_id}">open title</a>` : ''}</div></div><pre class="log">${esc(j.log.join('\n'))}</pre>`;
    box.appendChild(d); const pre = $('pre', d); pre.scrollTop = pre.scrollHeight;
  }
  if (jobs.some(j => j.status === 'running' || j.status === 'pending') && $('#view-build').classList.contains('on')) jobsTimer = setTimeout(pollJobs, 1000);
}

/* ------------------------------------------------------------------ minimal markdown (benchmark): headings, tables, paragraphs, code, bold */
function md(src) {
  const inl = s => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  const out = []; const lines = src.split(/\r?\n/); let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^#{1,3} /.test(l)) { const n = l.match(/^#+/)[0].length; out.push(`<h${n}>${inl(l.slice(n + 1))}</h${n}>`); i++; continue; }
    if (l.startsWith('|')) {
      const rows = []; while (i < lines.length && lines[i].startsWith('|')) rows.push(lines[i++]);
      const cells = r => r.replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      out.push('<table><thead><tr>' + cells(rows[0]).map(c => `<th>${inl(c)}</th>`).join('') + '</tr></thead><tbody>' + rows.slice(1).filter(r => !/^\|[\s:-|]+\|$/.test(r)).map(r => '<tr>' + cells(r).map(c => `<td>${inl(c)}</td>`).join('') + '</tr>').join('') + '</tbody></table>');
      continue;
    }
    if (l.trim()) { const p = []; while (i < lines.length && lines[i].trim() && !lines[i].startsWith('|') && !/^#{1,3} /.test(lines[i])) p.push(lines[i++]); out.push(`<p>${inl(p.join(' '))}</p>`); continue; }
    i++;
  }
  return out.join('\n');
}

/* ------------------------------------------------------------------ start */
pollStatus(); route();
