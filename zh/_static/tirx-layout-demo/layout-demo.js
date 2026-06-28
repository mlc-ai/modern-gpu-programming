/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * TIRx tensor-layout visualizer.
 *
 * The page shell, CSS vocabulary, and the draw()/hover/arrow interaction
 * pattern are adapted from the team's own course material
 * (mlsyscourse/slides-modern-gpu-programming, data-layout/site/demo/
 * tile_distributed.html). The TIRx S/R/O parser and the logical->physical
 * mapper below are original and mirror tvm/python/tvm/tirx/layout.py
 * (_flatten_coord / _split_coord and the TileLayout forward mapping).
 * This file is NOT derived from any third-party layout demo.
 */

'use strict';

// ── TIRx named axes (from tvm/python/tvm/tirx/layout.py _AXIS_NAMES, plus the
// device axis `pid` used by distributed layouts) ──────────────────────────────
const AXIS_ORDER = [
  'pid', 'bx', 'by', 'bz', 'cbx', 'cby', 'cbz', 'tx',
  'warpid', 'laneid', 'wgid', 'tid_in_wg', 'wid_in_wg', 'tid',
  'm', 'P', 'F', 'Bank', 'TCol', 'TLane',
];
// "Owner" axes name a physical unit that owns data (threads, devices); the rest
// (m, P, F, Bank, TCol, TLane) are storage/memory coordinates within an owner.
const OWNER_AXES = new Set([
  'pid', 'bx', 'by', 'bz', 'cbx', 'cby', 'cbz', 'tx',
  'warpid', 'laneid', 'wgid', 'tid_in_wg', 'wid_in_wg', 'tid',
]);
const KNOWN_AXES = new Set(AXIS_ORDER);
const MAX_ELEMENTS = 1024;  // render cap

function isOwnerAxis(a) { return OWNER_AXES.has(a); }
function product(arr) { return arr.reduce((a, b) => a * b, 1); }

// ── Parser ───────────────────────────────────────────────────────────────────
// Grammar mirrors layout.py: S[shape:stride] + R[shape:stride] + offset,
// stride/offset terms are "n@axis" (a bare int defaults to axis "m").

function splitTopLevel(s, sep) {
  const out = [];
  let depth = 0, cur = '';
  for (const ch of s) {
    if (ch === '(' || ch === '[') depth++;
    else if (ch === ')' || ch === ']') {
      depth--;
      if (depth < 0) throw new Error('unmatched closing bracket or parenthesis');
    }
    if (ch === sep && depth === 0) { out.push(cur); cur = ''; }
    else cur += ch;
  }
  if (depth !== 0) throw new Error('unmatched opening bracket or parenthesis');
  out.push(cur);
  return out;
}

function stripParens(s) {
  s = s.trim();
  if (s.startsWith('(') && s.endsWith(')')) return s.slice(1, -1);
  return s;
}

function parseIntStrict(s) {
  const t = s.trim();
  if (!/^-?\d+$/.test(t)) throw new Error(`expected integer, got "${t}"`);
  return parseInt(t, 10);
}

function parseTerm(tok) {
  const t = tok.trim();
  if (t.includes('@')) {
    const parts = t.split('@');
    if (parts.length !== 2) throw new Error(`bad term "${t}"`);
    const numPart = parts[0].trim();
    const axis = parts[1].trim();
    if (!KNOWN_AXES.has(axis)) throw new Error(`unknown axis "${axis}"`);
    const stride = numPart === '' ? 1 : parseIntStrict(numPart);
    return { stride, axis };
  }
  if (/^-?\d+$/.test(t)) return { stride: parseInt(t, 10), axis: 'm' };
  if (KNOWN_AXES.has(t)) return { stride: 1, axis: t };
  throw new Error(`bad term "${t}"`);
}

function defaultStrides(extents) {
  const n = extents.length;
  const strides = new Array(n).fill(1);
  for (let i = n - 2; i >= 0; i--) strides[i] = strides[i + 1] * extents[i + 1];
  return strides;
}

function parseExtents(s) {
  // Parse a comma-separated extent list and reject non-positive / oversized
  // extents, so bad input (e.g. R[1000000:...] or a 0/negative extent) can't
  // NaN-propagate or freeze the tab with huge loops in physOwners().
  const extents = splitTopLevel(stripParens(s), ',').map(parseIntStrict);
  for (const e of extents) {
    if (e <= 0) throw new Error(`extent must be positive, got ${e}`);
    if (e > MAX_ELEMENTS) throw new Error(`extent ${e} exceeds maximum ${MAX_ELEMENTS}`);
  }
  return extents;
}

function parseBracket(inner) {
  const parts = splitTopLevel(inner, ':');
  if (parts.length === 1) {
    const extents = parseExtents(parts[0]);
    const strides = defaultStrides(extents);
    return extents.map((e, i) => ({ extent: e, stride: strides[i], axis: 'm' }));
  }
  if (parts.length !== 2) throw new Error('layout bracket must be "shape : stride"');
  const extents = parseExtents(parts[0]);
  const terms = splitTopLevel(stripParens(parts[1]), ',').map(parseTerm);
  if (extents.length !== terms.length) {
    throw new Error(`shape has ${extents.length} dims but stride has ${terms.length}`);
  }
  return extents.map((e, i) => ({ extent: e, stride: terms[i].stride, axis: terms[i].axis }));
}

function bracketBody(piece, prefix) {
  const open = piece.indexOf('[');
  const close = piece.lastIndexOf(']');
  if (open < 0 || close < 0 || close < open) throw new Error(`malformed ${prefix}[...]`);
  return piece.slice(open + 1, close);
}

function parseSwizzlePrefix(src) {
  // Optional "Swizzle(per_element, swizzle_len, atom_len[, inner]) [∘|o|*] <layout>".
  const m = src.match(/^Swizzle\s*\(([^)]*)\)\s*(?:∘|o|\.|\*)?\s*([\s\S]*)$/i);
  if (!m) return { swizzle: null, rest: src };
  const a = m[1].split(',').map((s) => s.trim()).filter((s) => s.length);
  if (a.length < 3) throw new Error('Swizzle needs (per_element, swizzle_len, atom_len)');
  const per_element = parseIntStrict(a[0]);
  const swizzle_len = parseIntStrict(a[1]);
  const atom_len = parseIntStrict(a[2]);
  if (per_element < 0 || swizzle_len < 0 || atom_len < swizzle_len
      || per_element >= 31 || atom_len >= 31) {
    // atom_len/per_element feed 32-bit bitwise shifts in swizzleAddr; cap < 31.
    throw new Error('swizzle requires 0≤per_element<31, swizzle_len≥0, swizzle_len≤atom_len<31');
  }
  const inner = a[3] === undefined ? true : (a[3] === 'true' || a[3] === '1');
  return { swizzle: { per_element, swizzle_len, atom_len, inner }, rest: m[2].trim() };
}

function parseLayout(srcRaw) {
  const { swizzle, rest } = parseSwizzlePrefix(srcRaw.trim());
  const src = rest;
  const layout = { shard: [], replica: [], offset: {}, swizzle };
  let sawShard = false;
  for (let piece of splitTopLevel(src, '+')) {
    piece = piece.trim();
    if (piece === '') continue;
    if (piece.startsWith('S[')) {
      layout.shard = parseBracket(bracketBody(piece, 'S'));
      sawShard = true;
    } else if (piece.startsWith('R[')) {
      layout.replica = parseBracket(bracketBody(piece, 'R'));
    } else {
      const t = parseTerm(piece);
      layout.offset[t.axis] = (layout.offset[t.axis] || 0) + t.stride;
    }
  }
  if (!sawShard) throw new Error('layout needs a shard term, e.g. S[...]');
  return layout;
}

// ── Mapper (mirrors layout.py _flatten_coord / _split_coord + forward map) ─────

function flattenCoord(coord, shape) {
  let flat = 0;
  for (let i = 0; i < shape.length; i++) flat = flat * shape[i] + coord[i];
  return flat;
}

function splitCoord(flat, extents) {
  const n = extents.length;
  const res = new Array(n);
  let remaining = flat;
  for (let i = n - 1; i >= 0; i--) {
    if (i === 0) res[0] = remaining;
    else { res[i] = remaining % extents[i]; remaining = Math.floor(remaining / extents[i]); }
  }
  return res;
}

function coordFromFlat(flat, shape) { return splitCoord(flat, shape); }

function forwardBase(coord, shape, layout) {
  const flat = flattenCoord(coord, shape);
  const comps = splitCoord(flat, layout.shard.map((it) => it.extent));
  const phys = {};
  for (let k = 0; k < layout.shard.length; k++) {
    const it = layout.shard[k];
    phys[it.axis] = (phys[it.axis] || 0) + comps[k] * it.stride;
  }
  for (const axis of Object.keys(layout.offset)) {
    phys[axis] = (phys[axis] || 0) + layout.offset[axis];
  }
  return phys;
}

// Replica broadcasts the same logical element onto multiple physical owners:
// L(x) = { D(x) + r + O | r in R }.
function physOwners(coord, shape, layout) {
  let owners = [forwardBase(coord, shape, layout)];
  for (const rep of layout.replica) {
    const next = [];
    for (const o of owners) {
      for (let k = 0; k < rep.extent; k++) {
        const o2 = Object.assign({}, o);
        o2[rep.axis] = (o2[rep.axis] || 0) + k * rep.stride;
        next.push(o2);
      }
    }
    owners = next;
  }
  return owners;
}

function axesUsed(layout) {
  const s = new Set();
  for (const it of layout.shard) s.add(it.axis);
  for (const it of layout.replica) s.add(it.axis);
  for (const a of Object.keys(layout.offset)) s.add(a);
  return AXIS_ORDER.filter((a) => s.has(a));
}

function coordStr(phys, axes) {
  return axes.map((a) => `${a}=${phys[a] || 0}`).join(' ');
}

// Swizzle a linear memory address (mirrors src/tirx/ir/layout/swizzle_layout.cc
// SwizzleLayoutNode::Apply): low `per_element` bits are kept; above them, the
// swizzle bits are XOR'd to scatter bank conflicts.
function swizzleAddr(m, sw) {
  const base = 1 << sw.per_element;
  const innerMask = (1 << sw.swizzle_len) - 1;
  const outerMask = innerMask << sw.atom_len;
  const x = Math.floor(m / base);
  const fx = sw.inner ? (x ^ ((x & outerMask) >> sw.atom_len))
                      : (x ^ ((x & innerMask) << sw.atom_len));
  return fx * base + (m % base);
}

// Resolve the swizzle from the dtype + mode dropdowns (mirrors
// tma_utils.mma_atom_layout): per_element = bit_length(128//bits) - 1,
// swizzle_len = mode, atom_len = 3. Falls back to a typed Swizzle(...) prefix.
const SWIZZLE_LEN = { none: 0, '32': 1, '64': 2, '128': 3 };
function computeSwizzle() {
  const mode = swmodeSel ? swmodeSel.value : 'off';
  if (mode === 'off') return (ST.layout && ST.layout.swizzle) ? ST.layout.swizzle : null;
  const bits = +(dtypeSel ? dtypeSel.value : 16) || 16;
  const per_element = Math.floor(128 / bits).toString(2).length - 1;
  return {
    per_element, swizzle_len: SWIZZLE_LEN[mode] || 0, atom_len: 3, inner: true,
    bits, mode,
  };
}

// ── State + recompute ──────────────────────────────────────────────────────--
function getComputedStyleVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
const PALETTE = Array.from({ length: 8 }, (_, i) =>
  getComputedStyleVar(`--color-group-${i}`) || '#5b9bd5');
function paletteColor(v) { const n = PALETTE.length; return PALETTE[((v % n) + n) % n]; }

const ST = {
  shape: [4, 8],
  layout: null,
  error: null,
  tooBig: false,
  banks: 32,
  swizzle: null,
  gridAxes: [], yAxis: null, xAxis: null, cellAxes: [],
  yVals: [], xVals: [],
  byFlat: [],     // flat -> { owners:[phys], keys:[gridKey], color }
  byCell: new Map(),   // "y#x" -> [{flat, slot}]
};
let hovFlat = null;
let drawing = false;

function mk(t, c) { const d = document.createElement(t); if (c) d.className = c; return d; }
function gridKey(phys, axes) { return axes.map((a) => phys[a] || 0).join(','); }

function recompute() {
  ST.error = null; ST.tooBig = false;
  try {
    ST.shape = splitTopLevel(stripParens(shapeInput.value), ',').map(parseIntStrict);
    if (ST.shape.length === 0 || ST.shape.some((x) => x <= 0)) throw new Error('shape must be positive ints');
    ST.layout = parseLayout(exprInput.value);
  } catch (e) { ST.error = e.message; return; }

  const total = product(ST.shape);
  if (total > MAX_ELEMENTS) { ST.tooBig = true; return; }

  ST.shapeTotal = total;
  ST.shardTotal = product(ST.layout.shard.map((it) => it.extent));
  ST.mismatch = ST.shardTotal !== total;
  ST.swizzle = computeSwizzle();
  // elements that share one 4-byte bank word (e.g. 2 fp16, 4 fp8, 1 fp32)
  ST.elemsPerBank = ST.swizzle ? Math.max(1, Math.round(4 / ((ST.swizzle.bits || 32) / 8))) : 1;

  // 1) owners per element; in swizzle mode, also map the memory address through
  //    the swizzle and derive synthetic line/bank coordinates.
  ST.byFlat = new Array(total);
  for (let flat = 0; flat < total; flat++) {
    const owners = physOwners(coordFromFlat(flat, ST.shape), ST.shape, ST.layout);
    if (ST.swizzle) {
      // A shared-memory bank is 4 bytes; an element occupies dtype_bytes, so the
      // bank word index is floor(element_addr * dtype_bytes / 4). 32 banks per line.
      const bytes = (ST.swizzle.bits || 32) / 8;
      for (const o of owners) {
        const sm = swizzleAddr(o.m || 0, ST.swizzle);
        const word = Math.floor((sm * bytes) / 4);
        o.__sm = sm; o.__word = word; o.bank = word % ST.banks; o.line = Math.floor(word / ST.banks);
      }
    }
    ST.byFlat[flat] = { owners };
  }

  // 2) choose grid + color axes
  if (ST.swizzle) {
    ST.gridAxes = ['line', 'bank']; ST.yAxis = 'line'; ST.xAxis = 'bank';
    ST.cellAxes = []; ST.colorAxis = 'bank';
  } else {
    const used = axesUsed(ST.layout);
    const owners = used.filter(isOwnerAxis);
    ST.gridAxes = owners.length ? owners : used.filter((a) => !isOwnerAxis(a));
    ST.yAxis = ST.gridAxes[0] || null;
    ST.xAxis = ST.gridAxes[1] || null;
    ST.cellAxes = used.filter((a) => !ST.gridAxes.includes(a));
    // color axis = first grid axis from shard/offset, so a replica-only row axis
    // doesn't collapse every element to one color.
    const shardOffsetAxes = new Set(ST.layout.shard.map((it) => it.axis));
    for (const a of Object.keys(ST.layout.offset)) shardOffsetAxes.add(a);
    ST.colorAxis = ST.gridAxes.find((a) => shardOffsetAxes.has(a)) || ST.gridAxes[0] || null;
  }

  // 3) build cells, hover keys, colors
  ST.byCell = new Map();
  const yset = new Set(), xset = new Set(), cset = new Set();
  for (let flat = 0; flat < total; flat++) {
    const rec = ST.byFlat[flat];
    const keys = [];
    for (const o of rec.owners) {
      const y = ST.yAxis ? (o[ST.yAxis] || 0) : 0;
      const x = ST.xAxis ? (o[ST.xAxis] || 0) : 0;
      yset.add(y); xset.add(x);
      keys.push(gridKey(o, ST.gridAxes));
      const ck = y + '#' + x;
      if (!ST.byCell.has(ck)) ST.byCell.set(ck, []);
      ST.byCell.get(ck).push({ flat, slot: ST.swizzle ? ('addr ' + o.__sm) : coordStr(o, ST.cellAxes) });
    }
    rec.keys = keys;
    const cv = ST.colorAxis ? (rec.owners[0][ST.colorAxis] || 0) : 0;
    rec.color = paletteColor(cv);
    cset.add(cv);
  }
  ST.yVals = [...yset].sort((a, b) => a - b);
  ST.xVals = [...xset].sort((a, b) => a - b);
  ST.colorVals = [...cset].sort((a, b) => a - b);
}

// ── Display geometry for the logical grid ──────────────────────────────────--
function logicalGridDims() {
  if (ST.shape.length === 2) return { rows: ST.shape[0], cols: ST.shape[1] };
  if (ST.shape.length === 1) return { rows: 1, cols: ST.shape[0] };
  return { rows: 1, cols: product(ST.shape) };  // N-D -> flat strip
}

// ── Draw ────────────────────────────────────────────────────────────────────
function resetFit() {
  const p = document.getElementById('panels');
  if (p) { p.style.transform = 'none'; p.style.marginBottom = ''; }
}
function fitEmbed() {
  if (!document.body.classList.contains('lock')) return;
  const p = document.getElementById('panels');
  if (!p) return;
  const natural = p.offsetWidth;
  const pad = 2 * parseFloat(getComputedStyle(document.body).paddingLeft || '0');
  const avail = document.documentElement.clientWidth - pad;
  if (avail > 0 && natural > avail) {
    const sc = avail / natural;
    p.style.transformOrigin = 'top left';
    p.style.transform = 'scale(' + sc + ')';
    p.style.marginBottom = (-(p.offsetHeight * (1 - sc))) + 'px';
  }
}
function postHeight() {
  if (window.parent === window) return;
  const h = Math.ceil(document.body.scrollHeight);
  window.parent.postMessage({ tirxLayoutDemoHeight: h + 4 }, '*');
}
function draw() {
  drawing = true;
  resetFit();
  const status = document.getElementById('status');
  const g0 = document.getElementById('g0');
  const phys = document.getElementById('phys');
  const fb = document.getElementById('fb');
  const lg = document.getElementById('lg');

  if (ST.error) {
    status.innerHTML = `<span style="color:var(--color-bad)">parse error: ${escapeHtml(ST.error)}</span>`;
    g0.innerHTML = ''; phys.innerHTML = ''; lg.innerHTML = '';
    fb.innerHTML = '<div class="ftitle">Fix the layout expression to continue.</div>';
    setTimeout(() => { drawing = false; }, 0); return;
  }
  if (ST.tooBig) {
    status.innerHTML = `<span style="color:var(--color-bad)">${product(ST.shape)} elements exceeds the ${MAX_ELEMENTS} render cap — use a smaller shape.</span>`;
    g0.innerHTML = ''; phys.innerHTML = ''; lg.innerHTML = '';
    fb.innerHTML = '<div class="ftitle">Shape too large to visualize.</div>';
    setTimeout(() => { drawing = false; }, 0); return;
  }

  status.innerHTML = `<span style="color:var(--color-good)">ok</span> &nbsp; ` +
    `${product(ST.shape)} logical elements &nbsp;|&nbsp; ` +
    `${ST.yVals.length * (ST.xVals.length || 1)} physical cells`;
  if (ST.mismatch) {
    status.innerHTML += ` &nbsp;<span style="color:#b45309;font-weight:600">` +
      `⚠ shard total ${ST.shardTotal} ≠ shape total ${ST.shapeTotal} — mapping may be ill-formed</span>`;
  }
  if (ST.swizzle) {
    const s = ST.swizzle;
    const label = s.mode ? (s.mode === 'none' ? 'no swizzle' : s.mode + 'B swizzle') : 'swizzle';
    status.innerHTML += ` &nbsp;<span style="color:var(--dim)">` +
      `${label}${s.bits ? ', ' + s.bits + '-bit' : ''} → Swizzle(${s.per_element},${s.swizzle_len},${s.atom_len})</span>`;
  }
  document.getElementById('n0').textContent = `logical shape (${ST.shape.join(', ')})`;
  document.getElementById('nphys').textContent =
    (ST.xAxis ? `rows = ${ST.yAxis}, cols = ${ST.xAxis}` : `tiles = ${ST.yAxis || '(none)'}`) +
    (ST.cellAxes.length ? '   ·   in-cell: ' + ST.cellAxes.join(', ') : '');

  const hovKeys = hovFlat !== null ? new Set(ST.byFlat[hovFlat].keys) : null;
  drawLogical(hovKeys);
  drawPhysical(hovKeys);
  drawFormula();
  drawArrow();
  drawLegend();
  fitEmbed();
  postHeight();
  setTimeout(() => { drawing = false; }, 0);
}

function sharesOwner(flat, hovKeys) {
  if (!hovKeys) return false;
  return ST.byFlat[flat].keys.some((k) => hovKeys.has(k));
}

function drawLogical(hovKeys) {
  const g = document.getElementById('g0');
  g.innerHTML = '';
  const { rows, cols } = logicalGridDims();
  g.style.gridTemplateColumns = '30px repeat(' + cols + ', 46px)';
  g.appendChild(mk('div', 'hdr'));
  for (let c = 0; c < cols; c++) { const h = mk('div', 'hdr'); h.textContent = 'c' + c; g.appendChild(h); }
  for (let r = 0; r < rows; r++) {
    const rl = mk('div', 'rl'); rl.textContent = (rows > 1) ? ('r' + r) : ''; g.appendChild(rl);
    for (let c = 0; c < cols; c++) {
      const flat = r * cols + c;
      const d = mk('div', 'cell');
      d.dataset.flat = flat;
      d.textContent = flat;
      d.style.background = ST.byFlat[flat].color;
      d.style.color = '#fff';
      if (hovFlat !== null) {
        if (flat === hovFlat) d.classList.add('hov');
        else if (!sharesOwner(flat, hovKeys)) d.classList.add('dm');
      }
      g.appendChild(d);
    }
  }
}

function cellEntries(y, x) { return ST.byCell.get(y + '#' + x) || []; }

function makeSlot(entry) {
  const s = mk('div', 'gcell');
  s.style.background = ST.byFlat[entry.flat].color;
  s.style.color = '#fff';
  s.dataset.flat = entry.flat;
  s.textContent = entry.flat;
  s.title = entry.slot ? `element ${entry.flat} @ ${entry.slot}` : `element ${entry.flat}`;
  if (hovFlat !== null) {
    if (entry.flat === hovFlat) s.classList.add('hov');
    else s.classList.add('dm');
  }
  return s;
}

function drawPhysical(hovKeys) {
  const wrap = document.getElementById('phys');
  wrap.innerHTML = '';
  if (!ST.yAxis) { wrap.textContent = '(no physical axes)'; return; }

  if (ST.xAxis) {
    // 2D table: rows = yAxis values, cols = xAxis values.
    const table = mk('div', 'phys-table' + (ST.swizzle ? ' bank-mode' : ''));
    // In bank mode a cell is one 4-byte bank word holding elemsPerBank elements
    // laid out horizontally; otherwise a fixed 54px cell.
    const colW = ST.swizzle ? (ST.elemsPerBank * 48 + 8) : 54;
    table.style.gridTemplateColumns = '44px repeat(' + ST.xVals.length + ', ' + colW + 'px)';
    table.appendChild(corner());
    for (const x of ST.xVals) table.appendChild(axHdr(String(x), false, `${ST.xAxis}=${x}`));
    for (const y of ST.yVals) {
      table.appendChild(axHdr(String(y), true, `${ST.yAxis}=${y}`));
      for (const x of ST.xVals) {
        const cell = mk('div', 'pcell');
        const entries = cellEntries(y, x);
        if (hovFlat !== null && entries.some((e) => e.flat === hovFlat)) cell.classList.add('hov-cell');
        else if (hovFlat !== null && entries.length) cell.classList.add('dm-cell');
        for (const e of entries) cell.appendChild(makeSlot(e));
        table.appendChild(cell);
      }
    }
    wrap.appendChild(table);
  } else {
    // 1D: wrapped list of owner tiles, one per yAxis value.
    const list = mk('div', 'phys-1d');
    for (const y of ST.yVals) {
      const tile = mk('div', 'thread-tile');
      const lbl = mk('div', 'thread-lbl'); lbl.textContent = `${ST.yAxis}=${y}`; tile.appendChild(lbl);
      const slots = mk('div', 'thread-slots');
      const entries = cellEntries(y, 0).slice().sort((a, b) => a.flat - b.flat);
      if (hovFlat !== null && entries.some((e) => e.flat === hovFlat)) tile.classList.add('hov-tile');
      else if (hovFlat !== null && entries.length) tile.classList.add('dm-tile');
      for (const e of entries) slots.appendChild(makeSlot(e));
      tile.appendChild(slots);
      list.appendChild(tile);
    }
    wrap.appendChild(list);
  }
}

function corner() {
  const d = mk('div', 'ax-hdr corner');
  d.textContent = '↘';
  if (ST.yAxis) d.title = `rows = ${ST.yAxis}` + (ST.xAxis ? `, cols = ${ST.xAxis}` : '');
  return d;
}
function axHdr(text, isRow, title) {
  const d = mk('div', 'ax-hdr' + (isRow ? ' row' : ''));
  d.textContent = text;
  if (title) d.title = title;
  return d;
}

function drawFormula() {
  const fb = document.getElementById('fb');
  if (hovFlat === null) { fb.innerHTML = '<div class="ftitle">Click a logical element to see its mapping.</div>'; return; }
  const flat = hovFlat;
  const coord = coordFromFlat(flat, ST.shape);
  const comps = splitCoord(flat, ST.layout.shard.map((it) => it.extent));
  const perAxis = {};
  const termStrings = [];
  for (let k = 0; k < ST.layout.shard.length; k++) {
    const it = ST.layout.shard[k];
    perAxis[it.axis] = (perAxis[it.axis] || 0) + comps[k] * it.stride;
    termStrings.push(`${comps[k]}·${it.stride}@${it.axis}`);
  }
  const offStrings = [];
  for (const axis of Object.keys(ST.layout.offset)) {
    perAxis[axis] = (perAxis[axis] || 0) + ST.layout.offset[axis];
    offStrings.push(`${ST.layout.offset[axis]}@${axis}`);
  }
  const owners = ST.byFlat[flat].owners;
  let html = `<div class="ftitle">element ${flat} at logical (${coord.join(', ')})` +
    ` &nbsp;→&nbsp; shard split = (${comps.join(', ')})</div>`;
  html += '<div class="fcontent">';
  html += `terms: ${termStrings.join('  +  ')}` +
    (offStrings.length ? `  +  offset[${offStrings.join(', ')}]` : '') + '<br>';
  const baseParts = AXIS_ORDER.filter((a) => perAxis[a] !== undefined).map((a) => `<b>${perAxis[a]}</b>@${a}`);
  html += `base location: ${baseParts.join(' , ')}`;
  if (ST.swizzle) {
    const o0 = owners[0];
    const sw = ST.swizzle;
    const bytes = (sw.bits || 32) / 8;
    html += `<br>swizzle(${sw.per_element},${sw.swizzle_len},${sw.atom_len}): ` +
      `m=${o0.m || 0} → elem ${o0.__sm} → byte ${o0.__sm * bytes} → ` +
      `<b>bank ${o0.bank}</b>, line ${o0.line} (${bytes}-byte dtype, 4-byte banks ×32)`;
  }
  if (owners.length > 1) {
    html += `<br>owners (×${owners.length} via replica): ` +
      owners.map((o) => '{ ' + coordStr(o, ST.gridAxes) + ' }').join('  ,  ');
  }
  html += '</div>';
  fb.innerHTML = html;
}

// ── Arrow overlay (adapted from the course draw pattern) ───────────────────--
const panels = document.getElementById('panels');
const arrowSvg = document.getElementById('arrow');

function drawArrow() {
  arrowSvg.innerHTML = '';
  if (hovFlat === null) return;
  const leftCell = document.querySelector('#g0 .cell.hov');
  const rightCells = document.querySelectorAll('#phys .gcell.hov');
  if (!leftCell || rightCells.length === 0) return;
  const pr = panels.getBoundingClientRect();
  const ns = 'http://www.w3.org/2000/svg';
  const defs = document.createElementNS(ns, 'defs');
  const marker = document.createElementNS(ns, 'marker');
  marker.setAttribute('id', 'ah'); marker.setAttribute('markerWidth', '8'); marker.setAttribute('markerHeight', '6');
  marker.setAttribute('refX', '7'); marker.setAttribute('refY', '3'); marker.setAttribute('orient', 'auto');
  const poly = document.createElementNS(ns, 'polygon');
  poly.setAttribute('points', '0 0, 8 3, 0 6'); poly.setAttribute('fill', '#222');
  marker.appendChild(poly); defs.appendChild(marker); arrowSvg.appendChild(defs);
  const a = leftCell.getBoundingClientRect();
  const x1 = a.left + a.width / 2 - pr.left, y1 = a.top + a.height / 2 - pr.top;
  rightCells.forEach((rc) => {
    const b = rc.getBoundingClientRect();
    const x2 = b.left + b.width / 2 - pr.left, y2 = b.top + b.height / 2 - pr.top;
    const mx = (x1 + x2) / 2, my = Math.min(y1, y2) - 24;
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', `M${x1},${y1} Q${mx},${my} ${x2},${y2}`);
    path.setAttribute('marker-end', 'url(#ah)');
    arrowSvg.appendChild(path);
  });
}

function clearHov() {
  document.querySelectorAll('.cell.hov, .gcell.hov').forEach((d) => d.classList.remove('hov'));
  arrowSvg.innerHTML = '';
  hovFlat = null;
}

panels.addEventListener('click', (e) => {
  const cell = e.target.closest('.cell') || e.target.closest('.gcell');
  if (!cell || cell.dataset.flat === undefined) return;
  const flat = +cell.dataset.flat;
  if (hovFlat === flat) { clearHov(); draw(); return; }
  clearHov();
  hovFlat = flat;
  draw();
});

// ── Legend ──────────────────────────────────────────────────────────────────
function swatchEl(color) { const w = mk('div', 'swtch'); w.style.background = color; return w; }

function drawLegend() {
  const lg = document.getElementById('lg');
  lg.innerHTML = '';

  // Color key: an actual swatch-per-value table using the same palette as the grid.
  const r0 = mk('div', 'leg-row');
  const lead = mk('div', 'li');
  lead.innerHTML = `<b>color = ${ST.colorAxis || 'physical'} value:</b>`;
  r0.appendChild(lead);
  const vals = ST.colorVals || [];
  if (vals.length <= 8) {
    for (const v of vals) {
      const li = mk('div', 'li');
      li.appendChild(swatchEl(paletteColor(v)));
      li.appendChild(document.createTextNode(String(v)));
      r0.appendChild(li);
    }
  } else {
    for (let k = 0; k < 8; k++) {
      const li = mk('div', 'li');
      li.appendChild(swatchEl(PALETTE[k]));
      li.appendChild(document.createTextNode('≡' + k));
      r0.appendChild(li);
    }
    const note = mk('div', 'li');
    note.textContent = `(${ST.colorAxis} mod 8; ${vals.length} values)`;
    r0.appendChild(note);
  }
  lg.appendChild(r0);

  const r1 = mk('div', 'leg-row');
  const c2 = mk('div', 'li'); c2.textContent = 'number = logical element index'; r1.appendChild(c2);
  if (ST.layout.replica.length) {
    r1.appendChild(mk('div', 'leg-sep'));
    const c3 = mk('div', 'li');
    c3.textContent = 'replica → identical copies: the same color appears in multiple physical cells';
    r1.appendChild(c3);
  }
  lg.appendChild(r1);

  const r2 = mk('div', 'leg-row');
  const m = mk('div', 'li');
  const owners = axesUsed(ST.layout).filter(isOwnerAxis);
  const mem = axesUsed(ST.layout).filter((a) => !isOwnerAxis(a));
  m.textContent = 'owner axes: ' + (owners.join(', ') || '(none — pure memory layout)') +
    '   ·   memory axes: ' + (mem.join(', ') || '(none)');
  r2.appendChild(m);
  lg.appendChild(r2);
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// ── Presets + controls ───────────────────────────────────────────────────────
// Case-study presets use scaled-down shapes so every element renders; the
// mapping semantics match the full-size examples in the docs.
const PRESETS = [
  { label: 'Shard → lanes (intro)', shape: '4, 8', expr: 'S[(4,8):(8@laneid,1@laneid)]' },
  { label: 'Shard + registers', shape: '4, 8', expr: 'S[(4,2,4):(8@laneid,1@m,1@laneid)]' },
  { label: 'Shard + replica', shape: '4, 8', expr: 'S[(4,8):(8@laneid,1@laneid)] + R[2:1@warpid]' },
  {
    label: 'Tensor-core tile (doc example)', shape: '8, 16',
    expr: 'S[(8,2,4,2):(4@laneid,1@warpid,1@laneid,1)] + R[2:4@warpid] + 5@warpid',
  },
  { label: 'Distributed 2×2 GPU mesh (pid)', shape: '4, 4', expr: 'S[(2,2,2,2):(1@pid,2@m,2@pid,1@m)]' },
  { label: 'Mesh + replica (pid)', shape: '4, 4', expr: 'S[(2,2,4):(1@pid,2@m,1@m)] + R[2:2@pid]' },
  { label: 'Accelerator scratchpad (P/F)', shape: '4, 8', expr: 'S[(2,4,4):(4@F,1@P,1@F)]' },
  { label: 'Blackwell tensor memory (TLane/TCol)', shape: '4, 8', expr: 'S[(2,4,4):(4@TCol,1@TLane,1@TCol)]' },
  { label: 'SMEM, no swizzle (bank conflicts)', shape: '8, 64', expr: 'S[(8,64):(64@m,1@m)]', dtype: 16, mode: 'none' },
  { label: 'SMEM swizzle 128B (fp16)', shape: '8, 64', expr: 'S[(8,64):(64@m,1@m)]', dtype: 16, mode: '128' },
  { label: '1-D shard', shape: '8', expr: 'S[8:4@laneid]' },
  { label: 'Extents only (default strides)', shape: '8, 4', expr: 'S[(8,4)]' },
];

const DTYPES = [
  { label: 'float16 (16-bit)', bits: 16 },
  { label: 'bfloat16 (16-bit)', bits: 16 },
  { label: 'float8 (8-bit)', bits: 8 },
  { label: 'float32 (32-bit)', bits: 32 },
  { label: 'tfloat32 (32-bit)', bits: 32 },
  { label: 'float64 (64-bit)', bits: 64 },
];
const SWMODES = [
  { label: 'off (general layout)', value: 'off' },
  { label: 'none — raw banks', value: 'none' },
  { label: '32B swizzle', value: '32' },
  { label: '64B swizzle', value: '64' },
  { label: '128B swizzle', value: '128' },
];

const shapeInput = document.getElementById('shape');
const exprInput = document.getElementById('expr');
const presetSel = document.getElementById('preset');
const dtypeSel = document.getElementById('dtype');
const swmodeSel = document.getElementById('swmode');

function applyPreset(i) {
  const p = PRESETS[i];
  shapeInput.value = p.shape;
  exprInput.value = p.expr;
  if (dtypeSel) dtypeSel.value = String(p.dtype || 16);
  if (swmodeSel) swmodeSel.value = p.mode || 'off';
  refresh();
}
function refresh() { clearHov(); recompute(); draw(); }

function init() {
  PRESETS.forEach((p, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = p.label; presetSel.appendChild(o);
  });
  DTYPES.forEach((d) => {
    const o = document.createElement('option');
    o.value = d.bits; o.textContent = d.label; dtypeSel.appendChild(o);
  });
  SWMODES.forEach((s) => {
    const o = document.createElement('option');
    o.value = s.value; o.textContent = s.label; swmodeSel.appendChild(o);
  });
  presetSel.addEventListener('change', () => applyPreset(+presetSel.value));
  shapeInput.addEventListener('input', refresh);
  exprInput.addEventListener('input', refresh);
  dtypeSel.addEventListener('change', refresh);
  swmodeSel.addEventListener('change', refresh);
  window.addEventListener('resize', () => { resetFit(); fitEmbed(); postHeight(); });
  window.addEventListener('load', () => { resetFit(); fitEmbed(); postHeight(); });
  // Deep-linking for embeds: ?preset=<index|label-slug>&notitle
  const params = new URLSearchParams(location.search);
  let presetIdx = 0;
  const want = params.get('preset');
  if (want !== null) {
    const slug = (x) => x.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    if (/^\d+$/.test(want)) {
      presetIdx = Math.min(PRESETS.length - 1, Math.max(0, parseInt(want, 10)));
    } else {
      const w = slug(want);
      const found = PRESETS.findIndex((p) => slug(p.label).includes(w));
      if (found >= 0) presetIdx = found;
    }
  }
  presetSel.value = presetIdx;
  if (params.has('notitle')) document.body.classList.add('notitle');
  if (params.has('lock')) document.body.classList.add('lock');
  applyPreset(presetIdx);
}

init();
