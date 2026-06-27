// Shared behavior for all viz HTMLs
document.addEventListener('DOMContentLoaded', function() {
  var p = new URLSearchParams(location.search);
  if (p.has('notitle')) document.body.classList.add('notitle');

  // Directly opened demo pages are copied outside Sphinx templates, so they do
  // not receive html_js_files. Load the same language switch used by book pages.
  if (window.parent === window && !p.has('notitle')) {
    var langSwitchPath = window.location.pathname.indexOf('/zh/demo/') >= 0
      ? '../../_static/lang-switch.js'
      : '../_static/lang-switch.js';
    var s = document.createElement('script');
    s.src = new URL(langSwitchPath, window.location.href).href;
    document.head.appendChild(s);
  }

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

// Auto-height: when embedded in the book (demo-embed.js), the demo measures its
// OWN content height and posts it to the parent, which sizes the iframe to fit so
// there is never an inner scrollbar. This is push-based on purpose — the demo
// catches its own DOM changes (a click that appends rows, expands a panel, …),
// which a parent watching the iframe's <body> from outside can miss. Measuring
// body.scrollHeight (not documentElement, which is floored to the viewport) lets
// the reported height grow AND shrink with the content.
(function () {
  if (window.parent === window) return;   // only when embedded
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
  // documentElement exists even while we are still in <head>, so observers can be
  // attached immediately; the first read happens in the rAF after layout.
  try { new ResizeObserver(schedule).observe(document.documentElement); } catch (e) {}
  try {
    new MutationObserver(schedule).observe(document.documentElement, {
      subtree: true, childList: true, attributes: true, characterData: true
    });
  } catch (e) {}
  document.addEventListener('DOMContentLoaded', schedule);
  window.addEventListener('load', schedule);
  // Clicks often trigger async content changes; re-measure right after.
  window.addEventListener('click', function () { setTimeout(schedule, 0); }, true);
  // Catch late settling (fonts, deferred render).
  [100, 300, 600, 1200].forEach(function (t) { setTimeout(schedule, t); });
})();
