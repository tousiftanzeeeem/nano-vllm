"use strict";

const $ = id => document.getElementById(id);
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

let trace = null, cur = 0, playing = false, speed = 480, timer = null, selBlock = null;
let SEQ = 0;  // the traced sequence id, taken from the trace itself

/* ─────────────────────────── models ─────────────────────────── */

async function loadModels(){
  const sel = $("model"), note = $("formNote");
  try {
    const r = await fetch("/api/models");
    const data = await r.json();
    sel.innerHTML = "";
    data.models.forEach(m => {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = m.available ? m.label : `${m.label} — ${m.reason}`;
      o.disabled = !m.available;
      sel.appendChild(o);
    });
    const first = data.models.find(m => m.available);
    if (first) sel.value = first.id;
    $("modelHint").textContent = data.models.length > 1
      ? `${data.models.length - 1} local model${data.models.length > 2 ? "s" : ""} found`
      : "no local models found";

    if (data.runtimeProblem){
      note.className = "formnote";
      note.innerHTML = `<span>Real models are unavailable here — <b>${data.runtimeProblem}</b>. `
        + `The mock engine runs the genuine <code>Scheduler</code> and <code>BlockManager</code> with a stand-in forward pass, `
        + `so every block, page-table entry and admission decision you see is real.</span>`;
    } else {
      note.className = "formnote";
      note.innerHTML = `<span>Real models run with <code>block_size = 256</code> — the paged attention kernel requires it. `
        + `Use the mock engine for a block size small enough to watch.</span>`;
    }
    syncBlockSize();
  } catch (e) {
    note.className = "formnote err";
    note.textContent = "Could not reach the server. Is `python -m nanovllm.viz` running?";
  }
}

function syncBlockSize(){
  const isMock = $("model").value === "__mock__";
  const bs = $("blockSize"), note = $("realNote");
  bs.disabled = !isMock;
  if (!isMock){
    if (bs.value !== "256"){ bs.dataset.prev = bs.value; }
    bs.value = 256;
    // 256-token pages: a short prompt occupies one block and never grows,
    // so nudge the numbers toward something that actually pages.
    if (+$("maxTokens").value < 300) $("maxTokens").value = 600;
    if (+$("numBlocks").value < 8) $("numBlocks").value = 12;
    if (+$("budget").value < 256) $("budget").value = 512;
    note.hidden = false;
  } else {
    if (bs.dataset.prev) bs.value = bs.dataset.prev;
    note.hidden = true;
  }
}
$("model").addEventListener("change", syncBlockSize);

/* ─────────────────────────── run ─────────────────────────── */

let running = false;
async function run(){
  if (running) return;
  running = true;
  const btn = $("run"), note = $("formNote");
  btn.disabled = true; btn.textContent = "Tracing…";
  setPlaying(false);
  try {
    const r = await fetch("/api/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        model: $("model").value,
        prompt: $("prompt").value,
        maxTokens: +$("maxTokens").value,
        blockSize: +$("blockSize").value,
        numBlocks: +$("numBlocks").value,
        maxNumBatchedTokens: +$("budget").value,
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "trace failed");
    trace = data;
    SEQ = trace.seqs[0].id;
    $("transport").hidden = false;
    $("scope").hidden = false;
    $("scrub").max = trace.frames.length - 1;
    $("engineTag").textContent =
      `${trace.model.label} · ${trace.model.tokenizer} tokenizer · ${trace.cfg.numBlocks}×${trace.cfg.blockSize}-token blocks`;
    const capped = trace.cfg.poolAllocated > trace.cfg.numBlocks;
    $("poolSrc").textContent = capped
      ? `showing ${trace.cfg.numBlocks} of ${trace.cfg.poolAllocated} blocks the GPU allocated · ${trace.cfg.blockSize} tokens each`
      : `${trace.cfg.numBlocks} blocks × ${trace.cfg.blockSize} tokens`;
    $("footNote").innerHTML =
      `Traced <code>${trace.frames.length}</code> engine steps · prompt <code>${trace.frames[0].seqMeta[SEQ].numPrompt}</code> tokens `
      + `· generated <code>${trace.tokenTexts.length - trace.frames[0].seqMeta[SEQ].numPrompt}</code>`;
    note.className = "formnote";
    selBlock = null;
    buildPool(trace.cfg.numBlocks);
    seek(0);
  } catch (e) {
    note.className = "formnote err";
    note.textContent = e.message;
  } finally {
    running = false;
    btn.disabled = false; btn.textContent = "Trace request";
  }
}
$("run").addEventListener("click", run);

/* ─────────────────────────── pool grid ─────────────────────────── */

const CELL = 30, PAD = 5;
let poolCols = 11;

function buildPool(n){
  poolCols = n <= 24 ? 8 : n <= 48 ? 11 : 16;
  const svg = $("pool");
  const rows = Math.ceil(n / poolCols);
  const w = poolCols * (CELL + PAD) - PAD, h = rows * (CELL + PAD) - PAD;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);
  svg.innerHTML =
    `<defs><pattern id="sh" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
       <line x1="0" y1="0" x2="0" y2="6" stroke="#fff" stroke-opacity="0.55" stroke-width="2.5"></line>
     </pattern></defs>`;
  for (let i = 0; i < n; i++){
    const x = (i % poolCols) * (CELL + PAD), y = Math.floor(i / poolCols) * (CELL + PAD);
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("transform", `translate(${x},${y})`);
    g.style.cursor = "pointer";
    g.dataset.bid = i;
    g.innerHTML =
      `<rect class="cell-body" width="${CELL}" height="${CELL}" rx="3"></rect>
       <rect class="cell-hatch" width="${CELL}" height="${CELL}" rx="3" fill="url(#sh)" opacity="0"></rect>
       <rect class="cell-ring" x="1" y="1" width="${CELL-2}" height="${CELL-2}" rx="2.5" fill="none" stroke="none" stroke-width="2"></rect>
       <text class="cell-id" x="${CELL/2}" y="${CELL/2+3.5}" text-anchor="middle"
             font-family="IBM Plex Mono, monospace" font-size="9.5">${i}</text>
       <text class="cell-ref" x="${CELL-3}" y="9" text-anchor="end"
             font-family="IBM Plex Mono, monospace" font-size="8" font-weight="600" fill="#fff" opacity="0"></text>`;
    g.addEventListener("click", () => { selBlock = i; render(); });
    svg.appendChild(g);
  }
}

function paintPool(f){
  const svg = $("pool");
  const fresh = new Set(f.events.filter(e => e.kind === "block_alloc" || e.kind === "block_append")
                                .map(e => e.block_id));
  f.blocks.forEach(b => {
    const g = svg.querySelector(`g[data-bid="${b.id}"]`);
    if (!g) return;
    const body = g.querySelector(".cell-body"), hatch = g.querySelector(".cell-hatch");
    const idT = g.querySelector(".cell-id"), refT = g.querySelector(".cell-ref");
    const ring = g.querySelector(".cell-ring");
    if (b.free){
      body.setAttribute("fill", cssVar("--free"));
      body.setAttribute("fill-opacity", "0.45");
      idT.setAttribute("fill", cssVar("--ink-3"));
      hatch.setAttribute("opacity", "0");
      refT.setAttribute("opacity", "0");
    } else {
      body.setAttribute("fill", cssVar("--seq0"));
      body.setAttribute("fill-opacity", b.hashed ? "1" : "0.42");
      idT.setAttribute("fill", b.hashed ? "rgba(255,255,255,.9)" : cssVar("--ink-2"));
      hatch.setAttribute("opacity", b.ref > 1 ? "1" : "0");
      refT.textContent = b.ref > 1 ? "×" + b.ref : "";
      refT.setAttribute("opacity", b.ref > 1 ? "1" : "0");
    }
    const sel = selBlock === b.id;
    ring.setAttribute("stroke", sel ? cssVar("--ink") : fresh.has(b.id) ? cssVar("--warm") : "none");
    ring.setAttribute("stroke-opacity", sel || fresh.has(b.id) ? "1" : "0");
  });

  const b = selBlock !== null ? f.blocks[selBlock] : null;
  $("iId").textContent = b ? "#" + b.id : "—";
  $("iRef").textContent = b ? b.ref : "—";
  $("iTok").textContent = b
    ? (b.tokens.length ? b.tokens.join(" ") : b.free ? "free" : "allocated, not yet filled")
    : "click a block to inspect";
}

/* ─────────────────────────── token tape ─────────────────────────── */

function renderTape(f){
  const m = f.seqMeta[SEQ];
  const bs = trace.cfg.blockSize;
  const table = f.blockTables[SEQ] || [];
  const activeStart = m.numCached - m.numScheduled;
  const tape = $("tape");
  tape.innerHTML = "";

  const nBlocks = Math.max(1, Math.ceil(m.numTokens / bs));
  for (let li = 0; li < nBlocks; li++){
    const from = li * bs, to = Math.min(m.numTokens, from + bs);
    const phys = table[li];
    const group = document.createElement("div");
    const filling = from <= m.numCached - 1 && m.numCached <= to;
    group.className = "blockgroup" + (filling ? " filling" : "");
    const hd = document.createElement("div");
    hd.className = "bghd";
    hd.innerHTML = phys === undefined
      ? `<span>L${li}</span><span class="unmapped">unmapped</span>`
      : `<span>L${li}</span><span>→</span><span class="phys">#${phys}</span>`;
    group.appendChild(hd);

    const chips = document.createElement("div");
    chips.className = "chips";
    for (let i = from; i < to; i++){
      const c = document.createElement("span");
      let cls = "chip ";
      if (i >= m.numCached) cls += (i >= activeStart && i < m.numCached + m.numScheduled) ? "active" : "pending";
      else if (i >= activeStart && m.numScheduled > 0) cls += "active";
      else cls += i >= m.numPrompt ? "gen" : "cached";
      if (i === m.numTokens - 1 && i >= m.numPrompt) cls += " newest";
      c.className = cls;
      c.textContent = trace.tokenTexts[i] ?? "·";
      c.title = `token ${i}`;
      chips.appendChild(c);
    }
    group.appendChild(chips);
    tape.appendChild(group);
  }
}

/* ─────────────────────────── render ─────────────────────────── */

function render(){
  const f = trace.frames[cur];
  if (!f) return;
  const m = f.seqMeta[SEQ];

  $("stepNow").textContent = f.step;
  $("stepTot").textContent = "/" + (trace.frames.length - 1);
  $("scrub").value = cur;
  const pill = $("phasePill");
  pill.textContent = f.phase; pill.className = "phase " + f.phase;

  const util = f.used / f.total;
  $("mKv").innerHTML = `${Math.round(util*100)}<small>%</small>`;
  const bar = $("mKvBar");
  bar.style.width = (util*100) + "%";
  bar.className = util > 0.85 ? "warn" : "";
  $("mBlocks").innerHTML = `${(f.blockTables[SEQ]||[]).length}<small>/${f.total}</small>`;
  $("mTokens").innerHTML = `${m.numTokens - m.numPrompt}<small>/${m.maxTokens} out</small>`;
  $("mPhase").textContent = f.phase;
  $("mPre").textContent = f.preemptions;
  $("mPreWrap").classList.toggle("alert", f.preemptions > 0);
  $("mLat").innerHTML = `${f.latencyMs.toFixed(1)}<small>ms</small>`;

  // why banner
  const why = $("why"), tag = why.querySelector(".tag"), txt = $("whyText");
  const ev = f.notes.find(n => n.k === "evict"), bl = f.notes.find(n => n.k === "blocks"),
        ck = f.notes.find(n => n.k === "chunk"), ht = f.notes.find(n => n.k === "hit"),
        bg = f.notes.find(n => n.k === "budget"), dn = f.notes.find(n => n.k === "done");
  const grew = f.events.some(e => e.kind === "block_append");
  why.className = "why";
  if (ev){
    why.classList.add("evict"); tag.textContent = "preempt";
    txt.innerHTML = `Pool exhausted — sequence evicted, <span class="mono">${ev.blocks}</span> blocks freed, `
      + `<span class="mono">${ev.lost}</span> tokens of KV thrown away and requeued for recompute.`;
  } else if (bl){
    why.classList.add("block"); tag.textContent = "held back";
    txt.innerHTML = `Cannot admit: needs <span class="mono">${bl.need}</span> blocks, only <span class="mono">${bl.free}</span> free.`;
  } else if (ck){
    why.classList.add("block"); tag.textContent = "chunked prefill";
    txt.innerHTML = `Prompt enters in slices — <span class="mono">${ck.took}</span> of <span class="mono">${ck.of}</span> `
      + `remaining tokens fit this step's budget of <span class="mono">${f.budget}</span>.`;
  } else if (ht){
    why.classList.add("hit"); tag.textContent = "prefix cache hit";
    txt.innerHTML = `Reused <span class="mono">${ht.blocks}</span> cached prefix block(s) — that KV is never recomputed.`;
  } else if (bg){
    why.classList.add("block"); tag.textContent = "budget";
    txt.innerHTML = `Waiting a step: <span class="mono">${bg.want ?? "?"}</span> tokens wanted, `
      + `<span class="mono">${bg.left ?? 0}</span> left in the batch.`;
  } else if (dn){
    tag.textContent = "finished";
    txt.innerHTML = `Sequence finished (<span class="mono">${dn.reason}</span>) — all blocks returned to the pool.`;
  } else if (grew){
    why.classList.add("grow"); tag.textContent = "new block";
    txt.innerHTML = `The last block filled up, so <span class="mono">may_append()</span> grabbed a fresh one. `
      + `KV grows one page at a time — never a resized contiguous buffer.`;
  } else if (f.phase === "prefill"){
    tag.textContent = "prefill";
    txt.innerHTML = `Computing KV for <span class="mono">${f.batch[0]?.tokens ?? 0}</span> prompt tokens in one pass.`;
  } else {
    tag.textContent = "decode";
    txt.innerHTML = `One forward pass, one new token appended into logical block `
      + `<span class="mono">L${Math.floor((m.numTokens - 1) / trace.cfg.blockSize)}</span>.`;
  }

  renderTape(f);
  paintPool(f);

  // block table
  const pt = $("ptable"); pt.innerHTML = "";
  const table = f.blockTables[SEQ] || [];
  if (!table.length){
    pt.innerHTML = `<div class="empty">No blocks held — ${m.status === "finished" ? "finished, everything returned to the pool" : "nothing allocated yet"}.</div>`;
  } else {
    const fillingIdx = Math.floor(Math.max(0, m.numCached - 1) / trace.cfg.blockSize);
    table.forEach((pid, li) => {
      const blk = f.blocks[pid];
      const here = Math.min(trace.cfg.blockSize, Math.max(0, m.numTokens - li * trace.cfg.blockSize));
      const r = document.createElement("div");
      r.className = "prow" + (li === fillingIdx ? " filling" : "");
      r.innerHTML = `<span class="lg">L${li}</span><span class="ar">→</span><span class="ph">#${pid}</span>`
        + `<span class="meta">${here}/${trace.cfg.blockSize} tok${blk.ref > 1 ? ` · shared ×${blk.ref}` : ""}${blk.hashed ? "" : " · unhashed"}</span>`;
      r.addEventListener("click", () => { selBlock = pid; render(); });
      pt.appendChild(r);
    });
  }

  // batch bar
  const bb = $("budgetBar"); bb.innerHTML = "";
  const used = f.batch.reduce((a, b) => a + b.tokens, 0);
  f.batch.forEach(b => {
    const d = document.createElement("div");
    d.className = "slab" + (b.chunked ? " chunked" : "");
    d.style.flex = `${b.tokens} 0 0`;
    d.textContent = b.tokens >= 2 ? b.tokens : "";
    bb.appendChild(d);
  });
  if (f.budget - used > 0){
    const s = document.createElement("div");
    s.style.flex = `${f.budget - used} 0 0`;
    bb.appendChild(s);
  }
  $("budUsed").textContent = `${used} tok scheduled`;
  $("budCap").textContent = `max_num_batched_tokens ${f.budget}`;

  // output so far
  const np = m.numPrompt;
  const gen = trace.tokenTexts.slice(np, m.numTokens).join("").replace(/·/g, " ").replace(/↵/g, "\n");
  const out = $("outText");
  out.textContent = gen || "—";
  if (f.emitted && gen){
    const tail = (f.emitted.text || "").replace(/·/g, " ").replace(/↵/g, "\n");
    if (tail && gen.endsWith(tail)){
      out.innerHTML = "";
      out.append(document.createTextNode(gen.slice(0, gen.length - tail.length)));
      const b = document.createElement("b"); b.textContent = tail;
      out.append(b);
    }
  }

  // event log
  const log = $("evlog");
  if (!f.events.length){ log.innerHTML = `<div class="none">no events</div>`; }
  else {
    log.innerHTML = f.events.map(e => {
      const rest = Object.entries(e).filter(([k]) => k !== "kind" && k !== "step")
        .map(([k, v]) => `${k}=${v}`).join("  ");
      return `<div><span class="k">${e.kind}</span>${rest}</div>`;
    }).join("");
  }
}

/* ─────────────────────────── transport ─────────────────────────── */

function seek(i){
  cur = Math.max(0, Math.min(trace.frames.length - 1, i));
  render();
}
function setPlaying(p){
  playing = p;
  const ic = $("playIcon");
  if (ic) ic.innerHTML = p ? '<path d="M4 3h3v10H4zM9 3h3v10H9z"></path>' : '<path d="M4 3l9 5-9 5z"></path>';
  clearInterval(timer);
  if (p && trace) timer = setInterval(() => {
    if (cur >= trace.frames.length - 1){ setPlaying(false); return; }
    seek(cur + 1);
  }, speed);
}
$("bPlay").addEventListener("click", () => {
  if (!trace) return;
  if (!playing && cur >= trace.frames.length - 1) cur = 0;
  setPlaying(!playing);
});
$("bNext").addEventListener("click", () => { setPlaying(false); seek(cur + 1); });
$("bPrev").addEventListener("click", () => { setPlaying(false); seek(cur - 1); });
$("bFirst").addEventListener("click", () => { setPlaying(false); seek(0); });
$("bLast").addEventListener("click", () => { setPlaying(false); seek(trace.frames.length - 1); });
$("scrub").addEventListener("input", e => { setPlaying(false); seek(+e.target.value); });
document.querySelectorAll(".speeds button").forEach(b => {
  b.addEventListener("click", () => {
    speed = +b.dataset.sp;
    document.querySelectorAll(".speeds button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    if (playing) setPlaying(true);
  });
});
document.addEventListener("keydown", e => {
  if (["TEXTAREA", "INPUT", "SELECT"].includes(e.target.tagName)) return;
  if (!trace) return;
  if (e.key === " "){ e.preventDefault(); $("bPlay").click(); }
  if (e.key === "ArrowRight"){ setPlaying(false); seek(cur + 1); }
  if (e.key === "ArrowLeft"){ setPlaying(false); seek(cur - 1); }
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (trace){ buildPool(trace.cfg.numBlocks); render(); }
});

loadModels();
