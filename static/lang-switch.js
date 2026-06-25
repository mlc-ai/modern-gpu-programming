// Language switch for the bilingual book build.
// English pages live at the site root; Chinese pages live under /zh/.
(function () {
  var BTN_ID = "mgp-lang-switch";

  function onReady(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  function scriptRoot() {
    var scripts = document.getElementsByTagName("script");
    for (var i = scripts.length - 1; i >= 0; i--) {
      var src = scripts[i].src || "";
      var marker = "/_static/lang-switch.js";
      var pos = src.indexOf(marker);
      if (pos >= 0) return src.slice(0, pos + 1);
    }
    return new URL("./", window.location.href).href;
  }

  function relativePath(rootUrl) {
    var root = new URL(rootUrl, window.location.href);
    var here = new URL(window.location.href);
    var rootPath = root.pathname;
    if (here.pathname.indexOf(rootPath) === 0) {
      return here.pathname.slice(rootPath.length).replace(/^\/+/, "");
    }
    return here.pathname.replace(/^\/+/, "");
  }

  function targetFor(rootUrl) {
    var rel = relativePath(rootUrl);
    var isZh = rel === "zh" || rel.indexOf("zh/") === 0;
    var targetRel = isZh ? rel.replace(/^zh\/?/, "") : "zh/" + rel;
    if (targetRel === "index.html") targetRel = "";
    if (targetRel === "zh/index.html") targetRel = "zh/";
    return {
      isZh: isZh,
      href: new URL(targetRel, rootUrl).href + (window.location.hash || "")
    };
  }

  function addStyle() {
    if (document.getElementById(BTN_ID + "-style")) return;
    var style = document.createElement("style");
    style.id = BTN_ID + "-style";
    style.textContent =
      "#" + BTN_ID + "{" +
      "position:fixed;top:12px;right:14px;z-index:2147483000;" +
      "display:inline-flex;align-items:center;gap:6px;" +
      "padding:6px 11px;border:1px solid rgba(100,116,139,.35);" +
      "border-radius:999px;background:rgba(255,255,255,.94);" +
      "color:#1e293b;text-decoration:none;font:600 13px/1.2 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;" +
      "box-shadow:0 2px 8px rgba(15,23,42,.12);backdrop-filter:blur(8px)" +
      "}" +
      "#" + BTN_ID + ":hover{background:#fff;color:#0f172a;text-decoration:none;border-color:rgba(59,130,246,.55)}" +
      "@media (max-width:640px){#" + BTN_ID + "{top:auto;right:10px;bottom:10px}}" +
      "@media (prefers-color-scheme:dark){#" + BTN_ID + "{background:rgba(15,23,42,.9);color:#e2e8f0;border-color:rgba(148,163,184,.35)}#" + BTN_ID + ":hover{background:#0f172a;color:#fff}}";
    document.head.appendChild(style);
  }

  onReady(function () {
    if (document.getElementById(BTN_ID)) return;
    if (window.parent !== window) return;
    if (new URLSearchParams(window.location.search).has("notitle")) return;

    var root = scriptRoot();
    var target = targetFor(root);
    addStyle();

    var link = document.createElement("a");
    link.id = BTN_ID;
    link.href = target.href;
    link.textContent = target.isZh ? "English" : "中文";
    link.title = target.isZh ? "View English version" : "查看中文版";
    link.setAttribute("aria-label", link.title);
    document.body.appendChild(link);
  });
})();
