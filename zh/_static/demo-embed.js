// Zoomable viewer for embedded slide demos (iframes served from /demo/... or /demo_zh/...).
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
      // Scale with `transform: scale` (not CSS `zoom`): Safari does not apply `zoom`
      // to an <iframe>'s content, so the zoom controls did nothing and demos looked
      // broken there. With transform-origin at the top-left and the stage sized to
      // the *scaled* box, pointer/click events still map correctly into the iframe
      // in both Chrome and Safari.
      iframe.style.width = nat.w + "px";
      iframe.style.height = nat.h + "px";
      iframe.style.transformOrigin = "0 0";
      iframe.style.transform = "scale(" + z + ")";
      stage.style.width = Math.round(nat.w * z) + "px";
      stage.style.height = Math.round(nat.h * z) + "px";
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
    // Push-based height: the demo measures itself and postMessages its content
    // height (see viz-base.js), which the global listener below routes here. This
    // is the reliable path — the demo catches its own DOM changes that an outside
    // observer can miss. We only update the height; width/zoom stay as recompute()
    // set them, so the demo still fits the column width.
    iframe._setNatHeight = function (h) {
      if (h && Math.abs(h - nat.h) > 1) { nat.h = h; apply(); settleUntil = Date.now() + 120; }
    };
    // Fallback: re-measure when the demo's own content changes height (e.g. a click
    // expands a panel). Same-origin, so we observe the inner <body> directly.
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
    // The middle button toggles full screen (the old fit-width button was
    // redundant: fit-width is already the default and reapplied on every resize).
    function goFullscreen() {
      var fsEl = document.fullscreenElement || document.webkitFullscreenElement;
      if (fsEl === embed) {                                  // already full screen: exit
        var exit = document.exitFullscreen || document.webkitExitFullscreen;
        if (exit) { try { exit.call(document); } catch (e) {} }
      } else {
        var req = embed.requestFullscreen || embed.webkitRequestFullscreen;
        if (req) { try { req.call(embed); } catch (e) {} }
      }
    }
    function onFsChange() {
      var fsEl = document.fullscreenElement || document.webkitFullscreenElement;
      var nowFs = (fsEl === embed), wasFs = embed.classList.contains("demo-fs");
      if (nowFs === wasFs) return;                           // not a change for this embed
      embed.classList.toggle("demo-fs", nowFs);
      // Refit to the new viewport size (fills the screen on enter, the column on
      // exit). recompute() drives the transform, so the zoom buttons still work.
      requestAnimationFrame(function () { recompute(true); });
    }
    document.addEventListener("fullscreenchange", onFsChange);
    document.addEventListener("webkitfullscreenchange", onFsChange);

    bar.appendChild(btn("−", "Zoom out", function () { z = Math.max(0.2, z / 1.2); apply(); }));
    bar.appendChild(btn("⛶", "Full screen", goFullscreen));
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

  // A demo posts its content height (viz-base.js); route it to the matching iframe.
  window.addEventListener("message", function (e) {
    var d = e.data;
    if (!d || d.type !== "demoHeight" || !d.height) return;
    var frames = document.querySelectorAll('iframe[src*="/demo/"], iframe[src*="/demo_zh/"]');
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].contentWindow === e.source && frames[i]._setNatHeight) {
        frames[i]._setNatHeight(d.height);
        break;
      }
    }
  });

  function init() {
    document.querySelectorAll('iframe[src*="/demo/"], iframe[src*="/demo_zh/"]').forEach(setup);
  }
  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
