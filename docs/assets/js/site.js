/* PyMETA project page — behaviour: theme toggle, charts from results.js, copy buttons, nav state. */
(function () {
  "use strict";
  var D = window.PYMETA || {};
  function $(s, r) { return (r || document).querySelector(s); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function fmtInt(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }

  /* ---------- theme ---------- */
  var root = document.documentElement;
  function currentTheme() {
    var t = root.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function setTheme(t) {
    root.setAttribute("data-theme", t);
    try { localStorage.setItem("pymeta-theme", t); } catch (e) { /* storage may be unavailable */ }
    var b = $("#theme");
    if (b) b.setAttribute("aria-label", t === "dark" ? "Switch to light mode" : "Switch to dark mode");
  }
  var tb = $("#theme");
  if (tb) tb.addEventListener("click", function () { setTheme(currentTheme() === "dark" ? "light" : "dark"); });

  /* ---------- helpers ---------- */
  function bars(rows, opts) {
    var max = opts.max, fmt = opts.fmt || function (v) { return v; };
    return rows.map(function (r) {
      if (r.group) return '<div class="grp">' + esc(r.group) + "</div>";
      var ref = r.ref != null ? '<i class="ref" style="left:' + (100 * r.ref / max).toFixed(1) + '%"></i>' : "";
      return '<div class="bar"' + (r.title ? ' title="' + esc(r.title) + '"' : "") + '><span class="nm" title="' + esc(r.name) + '">' + esc(r.name) + '</span><span class="tr"><span class="fl ' + (r.cls || "") + '" style="width:' + (100 * r.v / max).toFixed(1) + '%"></span>' + ref + '</span><span class="v">' + fmt(r.v) + "</span></div>";
    }).join("");
  }

  /* ---------- RQ1: macro F1 bars with the best-finetuned reference ---------- */
  function renderRQ1(el) {
    if (!el || !D.rq1) return;
    var gname = { ft: "FINETUNED", old: "EARLIER PROMPTED", "new": "RECENT PROMPTED" };
    var rows = [], prev = null;
    D.rq1.forEach(function (d) {
      if (d.group !== prev) { rows.push({ group: gname[d.group] }); prev = d.group; }
      rows.push({ name: d.model, v: d.f1, cls: d.group === "new" ? "" : d.group, ref: D.bestFinetuned,
                  title: d.model + ": macro F1 " + d.f1 + "%, accuracy " + d.acc + "%" });
    });
    el.innerHTML = '<div class="bars">' + bars(rows, { max: 100, fmt: function (v) { return v.toFixed(1); } }) + "</div>";
  }

  /* ---------- Table 3: Logic Error over-prediction ---------- */
  function renderLerover(el) {
    if (!el || !D.lerover) return;
    var rows = [], prev = null;
    D.lerover.forEach(function (d) {
      if (d.group !== prev) { rows.push({ group: d.group === "old" ? "EARLIER PROMPTED" : "RECENT PROMPTED" }); prev = d.group; }
      rows.push({ name: d.model, v: d.v, cls: d.group === "old" ? "old" : "" });
    });
    el.innerHTML = bars(rows, { max: 100, fmt: function (v) { return v.toFixed(1) + "%"; } });
  }

  /* ---------- Table 1: label distribution ---------- */
  function renderLabelDist(el) {
    if (!el || !D.labelDist) return;
    var max = D.labelDist[0][1];
    el.innerHTML = bars(D.labelDist.map(function (r) { return { name: r[0], v: r[1], cls: "soft" }; }), { max: max, fmt: fmtInt });
  }

  /* ---------- splits table ---------- */
  function renderSplits(el) {
    if (!el || !D.splits) return;
    var rows = 0, qids = 0;
    D.splits.forEach(function (s) { rows += s.rows; qids += s.qids; });
    var tr = D.splits.map(function (s) {
      return "<tr><th>" + esc(s.split) + "</th><td>" + fmtInt(s.rows) + "</td><td>" + s.qids + "</td><td>" + s.errRate.toFixed(1) + "%</td></tr>";
    }).join("");
    el.innerHTML = '<table class="t"><thead><tr><th>Split</th><th>Submissions</th><th>Question IDs</th><th>Error rate</th></tr></thead><tbody>' + tr +
      "<tr><th>total</th><td><b>" + fmtInt(rows) + "</b></td><td><b>" + qids + "</b></td><td>52.3%</td></tr></tbody></table>";
  }

  /* ---------- Figure 1: coverage by predicted label ---------- */
  function renderCoverage(el, key) {
    if (!el || !D.coverage) return;
    var c = D.coverage[key];
    el.innerHTML = "<h4>" + esc(c.title) + ' <span style="font-weight:600;letter-spacing:0;text-transform:none">· n = ' + c.n + " · " + c.overall + "% covered overall</span></h4>" +
      '<div class="bars">' + bars(c.rows.map(function (r) { return { name: r[0], v: r[1] }; }), { max: 1, fmt: function (v) { return v.toFixed(2); } }) + "</div>";
  }

  /* ---------- subset distribution ---------- */
  function renderSubsetDist(el) {
    if (!el || !D.subsetDist) return;
    var max = D.subsetDist[0][1];
    el.innerHTML = bars(D.subsetDist.map(function (r) { return { name: r[0], v: r[1] }; }), { max: max, fmt: String });
  }

  /* ---------- Table 4 ---------- */
  function renderRQ3(el) {
    if (!el || !D.rq3) return;
    var keys = ["exact", "sampleF1", "microF1", "macroF1"];
    var best = keys.map(function (k) { return Math.max.apply(null, D.rq3.map(function (r) { return r[k]; })); });
    var tr = D.rq3.map(function (r) {
      return "<tr><th>" + esc(r.model) + "</th>" + keys.map(function (k, i) { return '<td class="' + (r[k] === best[i] ? "best" : "") + '">' + r[k].toFixed(1) + "</td>"; }).join("") + "</tr>";
    }).join("");
    el.innerHTML = '<table class="t"><thead><tr><th>Model</th><th>Exact set</th><th>Sample F1</th><th>Micro F1</th><th>Macro F1</th></tr></thead><tbody>' + tr + "</tbody></table>";
  }

  /* ---------- Table 5: dumbbell (HTML) ---------- */
  function renderLE45(el) {
    if (!el || !D.le45) return;
    var n = 45, span = 0.65; // the track covers 0–65 %
    function pct(k) { return (100 * (k / n) / span).toFixed(1); }
    var html = '<div class="dumbs" role="img" aria-label="Logic Error predicted on 45 audited programs under single-error and multi-error prompting">' +
      '<div class="dhead"><span></span><span><b class="ok">single-error</b><b class="bad">multi-error</b></span><span></span></div>';
    D.le45.forEach(function (d) {
      var a = pct(d.single), b = pct(d.any);
      html += '<div class="dumb" title="' + esc(d.model) + ": Logic Error in " + d.any + "/45 multi-error predictions (" + (100 * d.any / n).toFixed(1) + "%), first-listed in " + d.first + '">' +
        '<span class="nm">' + esc(d.model) + '</span><span class="tr"><i class="ln" style="left:' + a + "%;width:" + (b - a).toFixed(1) + '%"></i>' +
        '<i class="dot ok" style="left:' + a + '%">' + d.single + '</i><i class="dot bad" style="left:' + b + '%"></i></span><span class="v">' + d.any + "/45</span></div>";
    });
    html += '<div class="dticks"><span></span><span>' + [0, 20, 40, 60].map(function (t) { return '<i style="left:' + (100 * t / 100 / span).toFixed(1) + '%">' + t + "%</i>"; }).join("") + "</span><span></span></div></div>";
    el.innerHTML = html;
  }

  /* ---------- copy buttons ---------- */
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (res, rej) {
      var ta = document.createElement("textarea");
      ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.top = "-1000px";
      document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); res(); } catch (e) { rej(e); } finally { document.body.removeChild(ta); }
    });
  }
  Array.prototype.forEach.call(document.querySelectorAll("[data-copy]"), function (btn) {
    btn.addEventListener("click", function () {
      var src = $(btn.getAttribute("data-copy"));
      if (!src) return;
      copyText(src.textContent).then(function () {
        var old = btn.textContent; btn.textContent = "Copied ✓"; btn.classList.add("ok");
        setTimeout(function () { btn.textContent = old; btn.classList.remove("ok"); }, 1600);
      }).catch(function () { btn.textContent = "Press Ctrl+C"; });
    });
  });

  /* ---------- nav highlight ---------- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".navlinks a[href^='#']"));
  if ("IntersectionObserver" in window && links.length) {
    var map = {};
    links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { links.forEach(function (a) { a.classList.remove("on"); }); var a = map[en.target.id]; if (a) a.classList.add("on"); }
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }

  /* ---------- render ---------- */
  var bib = $("#bibtex"); if (bib && D.bibtex) bib.textContent = D.bibtex;
  renderRQ1($("#chart-rq1"));
  renderLerover($("#chart-lerover"));
  renderLabelDist($("#chart-labels"));
  renderSplits($("#table-splits"));
  renderCoverage($("#chart-cov-a"), "a");
  renderCoverage($("#chart-cov-b"), "b");
  renderSubsetDist($("#chart-subset"));
  renderRQ3($("#table-rq3"));
  renderLE45($("#chart-le45"));
})();
