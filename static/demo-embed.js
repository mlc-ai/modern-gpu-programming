// Zoomable viewer for embedded slide demos (iframes served from /demo/...).
// The demos are designed ~1200-1300px wide and center their content, so in the
// narrow article column they look small with margins. This wraps each demo in a
// viewport that (a) defaults to fit-the-column-width with no extra whitespace and
// (b) has +/- buttons to zoom in/out repeatedly, panning by scroll when zoomed in.
// Demos are same-origin (copied into the site root), so we can measure them.
(function () {
  function measure(iframe, fallbackW) {
    var nat = { w: fallbackW || 1280, h: 600 };
    try {
      var doc = iframe.contentDocument;
      var bs = doc.body && getComputedStyle(doc.body);
      var mw = bs ? parseFloat(bs.maxWidth) : NaN;
      if (mw && mw > 200) nat.w = Math.round(mw);
      iframe.style.width = nat.w + "px";          // lay the demo out at its design width
      iframe.style.height = "auto";
      // Prefer body.scrollHeight: documentElement.scrollHeight is floored to the
      // iframe's own viewport, so it can only grow, never shrink (the "stuck tall" bug).
      nat.h = (doc.body ? doc.body.scrollHeight : 0)
            || (doc.documentElement ? doc.documentElement.scrollHeight : 0)
            || nat.h;
    } catch (e) { /* not ready / cross-origin: keep fallback */ }
    return nat;
  }

  function setup(iframe) {
    if (iframe.dataset.demoReady) return;
    iframe.dataset.demoReady = "1";

    var embed = document.createElement("div");    embed.className = "demo-embed";
    var bar = document.createElement("div");      bar.className = "demo-toolbar";
    var viewport = document.createElement("div"); viewport.className = "demo-viewport";
    var stage = document.createElement("div");    stage.className = "demo-stage";
    // Insert the viewer exactly where the iframe sits, then move the iframe into
    // it. Do NOT remove the iframe's parent: when a demo is placed directly in the
    // page body (no wrapper div), the parent is the section, and removing it would
    // delete the surrounding prose.
    iframe.parentNode.insertBefore(embed, iframe);
    embed.appendChild(bar);
    embed.appendChild(viewport);
    viewport.appendChild(stage);
    stage.appendChild(iframe);

    // strip the inline sizing from the markdown so we control it
    iframe.style.minWidth = "0";
    iframe.style.border = "0";

    var nat = { w: 1280, h: 600 }, z = 1, fitZ = 1;

    function apply() {
      // Use CSS `zoom` (not `transform: scale`): zoom scales the layout box AND
      // keeps pointer/click events correctly mapped inside the iframe — a scaled
      // (transform) iframe mis-routes clicks, so the demo's buttons stop working.
      iframe.style.width = nat.w + "px";
      iframe.style.height = nat.h + "px";
      iframe.style.zoom = z;
    }
    var settleUntil = 0;
    function recompute(resetZoom) {
      nat = measure(iframe, nat.w);
      var vw = viewport.clientWidth || 800;
      fitZ = vw / nat.w;                          // true fit-width (the ⤢ button uses this)
      if (resetZoom) z = fitZ;                    // default to fit; preserve the user's zoom on content changes
      apply();
      settleUntil = Date.now() + 120;             // ignore ResizeObserver echoes from our own writes
    }
    // Re-measure when the demo's own content changes height (e.g. a click expands a
    // panel). Same-origin, so we observe the inner <body> directly — no postMessage
    // needed. (A cross-origin embed would instead have the demo postMessage its height.)
    function observe() {
      try {
        var doc = iframe.contentDocument;
        if (!doc || !doc.body || typeof ResizeObserver === "undefined" || iframe._demoRO) return;
        var raf = 0;
        iframe._demoRO = new ResizeObserver(function () {
          if (Date.now() < settleUntil || raf) return;
          raf = requestAnimationFrame(function () { raf = 0; recompute(false); });
        });
        iframe._demoRO.observe(doc.body);
      } catch (e) { /* cross-origin / unsupported: skip */ }
    }
    function btn(label, title, fn) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "demo-zoom-btn"; b.textContent = label; b.title = title;
      b.addEventListener("click", fn);
      return b;
    }
    bar.appendChild(btn("−", "Zoom out", function () { z = Math.max(0.2, z / 1.2); apply(); }));
    bar.appendChild(btn("⤢", "Fit width", function () { z = fitZ; apply(); }));
    bar.appendChild(btn("+", "Zoom in", function () { z = Math.min(6, z * 1.2); apply(); }));

    function init() { recompute(true); observe(); setTimeout(function () { recompute(true); observe(); }, 400); }
    iframe.addEventListener("load", init);
    try {
      if (iframe.contentDocument && iframe.contentDocument.readyState === "complete") init();
    } catch (e) {}

    var rt;
    window.addEventListener("resize", function () {
      clearTimeout(rt);
      rt = setTimeout(function () { recompute(true); }, 200);
    });
  }

  function init() {
    document.querySelectorAll('iframe[src*="/demo/"]').forEach(setup);
  }
  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
