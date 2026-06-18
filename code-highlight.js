/* ── Code highlight — shared TIRx/Python tokenizer + code block builder ──
 *
 * Usage:
 *   <link rel="stylesheet" href="code-highlight.css">
 *   <script src="code-highlight.js"></script>
 *
 * API:
 *   highlightCode(text)  — tokenize text, return HTML with <span> tokens
 *   escapeHtml(text)     — escape < > & for safe HTML insertion
 *
 *   createCodeBlock(container, code, options)
 *     Renders highlighted code into container with optional regions + focus lines.
 *     options:
 *       blockDefs:    [{ key, start, end, color }]              — clickable regions
 *       focusLines:   { key: [lineNos] } or { key: { lines, color } } — per-region line highlights
 *       onBlockClick: function(key)                             — callback on region click
 *
 * Auto-init:
 *   Elements with class "auto-hl" get their textContent highlighted on load.
 *   Combine with code-dark / code-light for theme:
 *     <pre class="code-dark auto-hl">Tx.copy_async(Asmem, A[...])</pre>
 *
 * Panel structure (optional wrapper for header + rounded corners):
 *     <div class="cb-panel dark">
 *       <div class="cb-header">Title</div>
 *       <pre class="code-dark" id="my-code"></pre>
 *     </div>
 *
 * Token classes (same as code-highlight.css):
 *   .kw  — keyword      .fn  — callable     .str — string
 *   .num — number        .cmt — comment      .op  — operator
 *   .typ — type          .dec — decorator
 *
 * Namespace convention: only the callable name gets .fn, NOT the prefix.
 *   Tx.copy_async  →  Tx.<span class="fn">copy_async</span>
 *   T.ptx.tcgen05.mma  →  T.ptx.tcgen05.<span class="fn">mma</span>
 */

(function (root) {
  "use strict";

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  var KEYWORDS = /^(def|with|for|in|if|else|elif|and|or|not|return|True|False|None|range|class|import|from|as|pass|break|continue|while|try|except|finally|raise|yield|lambda|assert|del|global|nonlocal)$/;

  function classifyToken(tok) {
    if (tok.startsWith("#")) return "cmt";
    if (tok.startsWith("@")) return "dec";
    if (tok.startsWith('"') || tok.startsWith("'")) return "str";
    if (/^\d[\d.]*$/.test(tok)) return "num";
    if (KEYWORDS.test(tok)) return "kw";
    if (/^(?:Tx|T)(\.[A-Za-z_]\w*)+$/.test(tok)) return "fn";
    if (/^[A-Za-z_]\w*$/.test(tok)) return "fn";
    return "";
  }

  var TOKEN_RE = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|#.*$|@[A-Za-z_][\w.]*|\b(?:def|with|for|in|if|else|elif|and|or|not|return|True|False|None|range|class|import|from|as|pass|break|continue|while|try|except|finally|raise|yield|lambda|assert|del|global|nonlocal)\b|\b\d[\d.]*\b|(?:Tx|T)(?:\.[A-Za-z_]\w*)+|\b[A-Za-z_]\w*(?=\())/gm;

  function highlightCode(text) {
    TOKEN_RE.lastIndex = 0;
    var out = "";
    var last = 0;
    var m;
    while ((m = TOKEN_RE.exec(text)) !== null) {
      out += escapeHtml(text.slice(last, m.index));
      var tok = m[0];
      var cls = classifyToken(tok);
      if (cls === "fn" && tok.indexOf(".") !== -1) {
        // Namespace.callable — only wrap the last component
        var lastDot = tok.lastIndexOf(".");
        out += escapeHtml(tok.substring(0, lastDot + 1));
        out += '<span class="fn">' + escapeHtml(tok.substring(lastDot + 1)) + "</span>";
      } else if (cls) {
        out += '<span class="' + cls + '">' + escapeHtml(tok) + "</span>";
      } else {
        out += escapeHtml(tok);
      }
      last = TOKEN_RE.lastIndex;
    }
    out += escapeHtml(text.slice(last));
    return out;
  }

  // Auto-init: highlight elements with class "auto-hl"
  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", function () {
      var els = document.querySelectorAll(".auto-hl");
      for (var i = 0; i < els.length; i++) {
        els[i].innerHTML = highlightCode(els[i].textContent);
      }
    });
  }

  // ── createCodeBlock: render code with optional regions + focus lines ──

  function hexToRgb(hex) {
    return [
      parseInt(hex.slice(1, 3), 16),
      parseInt(hex.slice(3, 5), 16),
      parseInt(hex.slice(5, 7), 16)
    ];
  }

  function createCodeBlock(container, code, options) {
    options = options || {};
    var blockDefs = options.blockDefs;
    var focusLines = options.focusLines;
    var onBlockClick = options.onBlockClick;
    var lines = code.split("\n");

    if (blockDefs) {
      // Render code split into clickable regions
      container.innerHTML = blockDefs.map(function (b) {
        var lineHtml = lines.slice(b.start - 1, b.end).map(function (line, idx) {
          var lineNo = b.start + idx;
          var content = line.length ? highlightCode(line) : "&nbsp;";
          return '<div class="cb-line" data-ln="' + lineNo + '">' + content + "</div>";
        }).join("");
        var style = b.color ? ' style="border-left-color:' + b.color + '"' : "";
        return '<div class="cb-region" data-region="' + b.key + '"' + style + ">" + lineHtml + "</div>";
      }).join("");

      // Color map for hover/active/focus
      var colorMap = {};
      blockDefs.forEach(function (b) { if (b.color) colorMap[b.key] = b.color; });

      var regions = container.querySelectorAll(".cb-region");
      var allLines = container.querySelectorAll(".cb-line");
      var activeKey = null;

      function clearFocus() {
        for (var i = 0; i < allLines.length; i++) {
          allLines[i].classList.remove("cb-focus");
          allLines[i].style.background = "";
        }
      }

      function applyFocus(key) {
        if (!focusLines || !focusLines[key]) return;
        var fl = focusLines[key];
        var lns = Array.isArray(fl) ? fl : fl.lines;
        var color = Array.isArray(fl) ? null : fl.color;
        if (!color && colorMap[key]) {
          var rgb = hexToRgb(colorMap[key]);
          color = "rgba(" + rgb.join(",") + ",0.18)";
        }
        if (!color) color = "rgba(255,255,255,0.15)";
        for (var i = 0; i < lns.length; i++) {
          var el = container.querySelector('.cb-line[data-ln="' + lns[i] + '"]');
          if (el) {
            el.classList.add("cb-focus");
            el.style.background = color;
          }
        }
      }

      for (var r = 0; r < regions.length; r++) {
        (function (el) {
          var regionKey = el.dataset.region;

          // Hover with region color
          if (colorMap[regionKey]) {
            var rgb = hexToRgb(colorMap[regionKey]);
            var hoverBg = "rgba(" + rgb.join(",") + ",0.15)";
            el.addEventListener("mouseenter", function () {
              if (!el.classList.contains("active")) el.style.background = hoverBg;
            });
            el.addEventListener("mouseleave", function () {
              if (!el.classList.contains("active")) el.style.background = "";
            });
          }

          // Click to activate
          el.addEventListener("click", function () {
            if (regionKey === activeKey) return;
            activeKey = regionKey;
            for (var j = 0; j < regions.length; j++) {
              regions[j].classList.remove("active");
              regions[j].style.background = "";
            }
            el.classList.add("active");
            if (colorMap[regionKey]) {
              var rgb2 = hexToRgb(colorMap[regionKey]);
              el.style.background = "rgba(" + rgb2.join(",") + ",0.15)";
            }
            clearFocus();
            applyFocus(regionKey);
            if (onBlockClick) onBlockClick(regionKey);
          });
        })(regions[r]);
      }
    } else {
      // Simple highlighted code, no regions
      container.innerHTML = lines.map(function (line, idx) {
        var content = line.length ? highlightCode(line) : "&nbsp;";
        return '<div class="cb-line" data-ln="' + (idx + 1) + '">' + content + "</div>";
      }).join("");
    }
  }

  // ── renderAnnotatedCode: highlight with per-line CSS classes ──

  function renderAnnotatedCode(container, code, lineClasses) {
    lineClasses = lineClasses || {};
    var lines = code.split("\n");
    container.innerHTML = lines.map(function (line, idx) {
      var content = line.length ? highlightCode(line) : "&nbsp;";
      var cls = lineClasses[idx + 1];
      if (cls) return '<div class="' + cls + '">' + content + "</div>";
      return "<div>" + content + "</div>";
    }).join("");
  }

  // Export
  root.highlightCode = highlightCode;
  root.escapeHtml = escapeHtml;
  root.createCodeBlock = createCodeBlock;
  root.renderAnnotatedCode = renderAnnotatedCode;
})(typeof window !== "undefined" ? window : this);
