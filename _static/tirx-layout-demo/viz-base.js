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
 * Derived from mlsyscourse/slides-modern-gpu-programming
 * (data-layout/site/viz-base.js). Shared behavior for the TIRx
 * layout visualization; not derived from any third-party demo.
 */

// Shared behavior for all viz HTMLs
document.addEventListener('DOMContentLoaded', function() {
  var p = new URLSearchParams(location.search);
  if (p.has('notitle')) document.body.classList.add('notitle');

  // Forward arrow keys to parent (reveal.js) when embedded
  if (window.parent !== window) {
    document.addEventListener('keydown', function(e) {
      if ([37, 38, 39, 40, 27, 32].indexOf(e.keyCode) !== -1) {
        // Left, Up, Right, Down, Escape, Space
        window.parent.postMessage({ type: 'revealKey', keyCode: e.keyCode }, '*');
      }
    });
  }
});

// Auto-height: when embedded in the book, measure our own content height and post
// it to the parent so it can size the iframe to fit (no inner scrollbar). This
// demo is responsive (it fills the iframe width), so only the HEIGHT needs to
// follow content. Push-based, so it catches our own DOM changes (editing the
// layout, clicking a cell, switching presets) that an outside observer can miss.
(function () {
  if (window.parent === window) return;
  var lastH = 0;
  function report() {
    var b = document.body, de = document.documentElement;
    var h = (b ? b.scrollHeight : 0) || (de ? de.scrollHeight : 0) || 0;
    if (h && Math.abs(h - lastH) > 1) {
      lastH = h;
      window.parent.postMessage({ type: 'demoHeight', height: h }, '*');
    }
  }
  var scheduled = false;
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(function () { scheduled = false; report(); });
  }
  try { new ResizeObserver(schedule).observe(document.documentElement); } catch (e) {}
  try {
    new MutationObserver(schedule).observe(document.documentElement, {
      subtree: true, childList: true, attributes: true, characterData: true
    });
  } catch (e) {}
  document.addEventListener('DOMContentLoaded', schedule);
  window.addEventListener('load', schedule);
  window.addEventListener('click', function () { setTimeout(schedule, 0); }, true);
  [100, 300, 600, 1200].forEach(function (t) { setTimeout(schedule, t); });
})();
