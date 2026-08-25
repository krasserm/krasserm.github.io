/* Hero figure: Gaussian-process posterior over an ordered axis of focus
   topics, computed exactly in the browser (RBF kernel, zero mean on centered
   observations, seeded samples). */
(function () {
  'use strict';

  var svg = document.getElementById('gp');
  if (!svg) return;

  /* ---- data: focus topics as ordered, unevenly spaced GP observations ---- */
  var obs = [
    { x: 2015.2,  y: 1.00, label: 'System Integration' },
    { x: 2018.3,  y: 1.62, label: 'Event Sourcing' },
    { x: 2020.0,  y: 1.30, label: 'Distributed Systems' },
    { x: 2022.1,  y: 1.72, label: 'Deep Learning' },
    { x: 2024.2,  y: 2.20, label: 'Bayesian ML' },
    { x: 2025.7,  y: 2.90, label: 'AI Search' },
    { x: 2026.65, y: 3.35, label: 'Agentic AI', now: true }
  ];

  /* ---- GP posterior: RBF kernel, exact inference ---- */
  var SF2 = 1.3, ELL = 0.85, SN2 = 0.0025;
  function kern(a, b) { var d = (a - b) / ELL; return SF2 * Math.exp(-0.5 * d * d); }

  var n = obs.length;
  var X = obs.map(function (o) { return o.x; });
  var ybar = obs.reduce(function (s, o) { return s + o.y; }, 0) / n;
  var yc = obs.map(function (o) { return o.y - ybar; });

  var K = [];
  for (var i = 0; i < n; i++) {
    K.push([]);
    for (var j = 0; j < n; j++) K[i].push(kern(X[i], X[j]) + (i === j ? SN2 : 0));
  }

  function inv(M) { /* gauss-jordan */
    var m = M.length, A = M.map(function (r, ri) {
      return r.concat(r.map(function (_, ci) { return ri === ci ? 1 : 0; }));
    });
    for (var c = 0; c < m; c++) {
      var p = c;
      for (var r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      var tmp = A[c]; A[c] = A[p]; A[p] = tmp;
      var pv = A[c][c];
      for (var k = 0; k < 2 * m; k++) A[c][k] /= pv;
      for (r = 0; r < m; r++) if (r !== c) {
        var f = A[r][c];
        for (k = 0; k < 2 * m; k++) A[r][k] -= f * A[c][k];
      }
    }
    return A.map(function (r) { return r.slice(m); });
  }

  var Kinv = inv(K);
  var alpha = Kinv.map(function (row) {
    return row.reduce(function (s, v, j) { return s + v * yc[j]; }, 0);
  });

  var T0 = 2014.3, T1 = 2027.5, NG = 130;
  var ts = [], g;
  for (g = 0; g < NG; g++) ts.push(T0 + (T1 - T0) * g / (NG - 1));

  var Ks = ts.map(function (t) { return X.map(function (x) { return kern(t, x); }); });
  var mu = Ks.map(function (row) {
    return ybar + row.reduce(function (s, v, j) { return s + v * alpha[j]; }, 0);
  });

  var cov = [];
  for (i = 0; i < NG; i++) {
    cov.push([]);
    for (j = 0; j < NG; j++) {
      var s = kern(ts[i], ts[j]);
      for (var a = 0; a < n; a++) for (var b = 0; b < n; b++)
        s -= Ks[i][a] * Kinv[a][b] * Ks[j][b];
      cov[i].push(s + (i === j ? 1e-8 : 0));
    }
  }
  var sd = cov.map(function (row, idx) { return Math.sqrt(Math.max(row[idx], 0)); });

  function chol(A) {
    var m = A.length, L = [];
    for (var r = 0; r < m; r++) { L.push(new Float64Array(m)); }
    for (r = 0; r < m; r++) for (var c = 0; c <= r; c++) {
      var v = A[r][c];
      for (var k = 0; k < c; k++) v -= L[r][k] * L[c][k];
      L[r][c] = (r === c) ? Math.sqrt(Math.max(v, 1e-12)) : v / L[c][c];
    }
    return L;
  }
  var L = chol(cov);

  /* seeded PRNG so the figure is identical on every load */
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var rnd = mulberry32(20260825);
  function gauss() {
    var u = Math.max(rnd(), 1e-9), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  var samples = [];
  for (var sIdx = 0; sIdx < 3; sIdx++) {
    var z = []; for (g = 0; g < NG; g++) z.push(gauss());
    var path = [];
    for (i = 0; i < NG; i++) {
      var acc = mu[i];
      for (j = 0; j <= i; j++) acc += L[i][j] * z[j];
      path.push(acc);
    }
    samples.push(path);
  }

  /* ---- responsive rendering ---- */
  var mq = window.matchMedia('(max-width: 640px)');

  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var mobile = mq.matches;

    var PL = 42, PR = 982, PT = 24, AX = 316;
    var H = mobile ? 660 : 400;
    svg.setAttribute('viewBox', '0 0 1000 ' + H);

    var vmin = Infinity, vmax = -Infinity, i, j, sIdx;
    for (i = 0; i < NG; i++) {
      vmin = Math.min(vmin, mu[i] - 2.05 * sd[i]);
      vmax = Math.max(vmax, mu[i] + 2.05 * sd[i]);
      for (sIdx = 0; sIdx < 3; sIdx++) {
        vmin = Math.min(vmin, samples[sIdx][i]);
        vmax = Math.max(vmax, samples[sIdx][i]);
      }
    }
    var pad = 0.12 * (vmax - vmin); vmin -= pad; vmax += pad;

    function mx(t) { return PL + (t - T0) / (T1 - T0) * (PR - PL); }
    function my(v) { return AX - (v - vmin) / (vmax - vmin) * (AX - PT); }

    function polyline(vals) {
      var d = '';
      for (var i = 0; i < NG; i++) d += (i ? 'L' : 'M') + mx(ts[i]).toFixed(1) + ',' + my(vals[i]).toFixed(1);
      return d;
    }
    function band(mult) {
      var d = '';
      for (var i = 0; i < NG; i++) d += (i ? 'L' : 'M') + mx(ts[i]).toFixed(1) + ',' + my(mu[i] + mult * sd[i]).toFixed(1);
      for (i = NG - 1; i >= 0; i--) d += 'L' + mx(ts[i]).toFixed(1) + ',' + my(mu[i] - mult * sd[i]).toFixed(1);
      return d + 'Z';
    }

    var NS = 'http://www.w3.org/2000/svg';
    function el(name, attrs, parent) {
      var e = document.createElementNS(NS, name);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      (parent || svg).appendChild(e);
      return e;
    }

    /* coordinate grid: one vertical line per focus topic, a few levels */
    var gGrid = el('g', {});
    obs.forEach(function (o) {
      el('line', { x1: mx(o.x), y1: PT, x2: mx(o.x), y2: AX, stroke: '#1a2029', 'stroke-width': 1 }, gGrid);
    });
    for (var lv = 1; lv <= 4; lv++) {
      var yv = my(vmin + lv * (vmax - vmin) / 5);
      el('line', { x1: PL, y1: yv, x2: PR, y2: yv, stroke: '#1a2029', 'stroke-width': 1 }, gGrid);
    }
    obs.forEach(function (o) {
      el('line', { x1: mx(o.x), y1: AX, x2: mx(o.x), y2: AX + 5, stroke: '#39424f', 'stroke-width': 1 }, gGrid);
    });

    /* axis */
    var gAxis = el('g', {});
    el('line', { x1: PL, y1: AX, x2: PR, y2: AX, stroke: '#39424f', 'stroke-width': 1 }, gAxis);
    el('text', {
      x: PL - 14, y: (PT + AX) / 2, fill: '#7d8794', 'font-size': mobile ? 20 : 11,
      'font-family': 'JetBrains Mono, monospace', 'text-anchor': 'middle',
      transform: 'rotate(-90 ' + (PL - 14) + ' ' + (PT + AX) / 2 + ')'
    }, gAxis).textContent = 'attention (a.u.)';

    /* curves, revealed left-to-right on load */
    var gCurves = el('g', { 'class': 'gp-curves' });
    el('path', { d: band(2), fill: '#440154', 'fill-opacity': 0.5 }, gCurves);
    el('path', { d: band(1), fill: '#35b7ab', 'fill-opacity': 0.22 }, gCurves);
    for (sIdx = 0; sIdx < 3; sIdx++)
      el('path', { d: polyline(samples[sIdx]), fill: 'none', stroke: '#35b7ab', 'stroke-opacity': 0.45, 'stroke-width': 1 }, gCurves);
    el('path', { d: polyline(mu), fill: 'none', stroke: '#e8edf4', 'stroke-width': 2.4, 'stroke-linecap': 'round' }, gCurves);

    /* observation points + labels */
    var gPts = el('g', { 'class': 'gp-pts' });
    var gLabels = el('g', { 'class': 'gp-labels' });

    obs.forEach(function (o, idx) {
      var x = mx(o.x), y = my(o.y);
      el('line', { x1: x, y1: y + 8, x2: x, y2: AX, stroke: '#2a3340', 'stroke-width': 1, 'stroke-dasharray': '2 4' }, gPts);
      if (o.now) {
        el('circle', { cx: x, cy: y, r: mobile ? 12 : 9, fill: '#fde725', 'fill-opacity': 0.3 }, gPts);
        el('circle', { cx: x, cy: y, r: mobile ? 7 : 5, fill: '#fde725', stroke: '#0d1117', 'stroke-width': 2 }, gPts);
      } else {
        el('circle', { cx: x, cy: y, r: mobile ? 6 : 4.2, fill: '#0d1117', stroke: '#e8edf4', 'stroke-width': 2 }, gPts);
      }

      var t;
      if (mobile) {
        /* vertical labels hanging from each gridline, attached to the figure */
        var lx = x + 8, ly = AX + 16;
        t = el('text', {
          x: lx, y: ly, fill: o.now ? '#e8edf4' : '#8b95a3', 'font-size': 25,
          'font-family': 'JetBrains Mono, monospace', 'text-anchor': 'end',
          'font-weight': o.now ? 600 : 400,
          transform: 'rotate(-90 ' + lx + ' ' + ly + ')'
        }, gLabels);
      } else {
        var lane = (idx % 2 === 0) ? 349 : 371;
        var anchor = 'middle', tx = x;
        if (idx === 0) { anchor = 'start'; tx = Math.max(x - 40, PL); }
        if (idx === obs.length - 1) { anchor = 'end'; tx = Math.min(x + 30, PR); }
        t = el('text', {
          x: tx, y: lane, fill: o.now ? '#e8edf4' : '#8b95a3', 'font-size': 11.5,
          'font-family': 'JetBrains Mono, monospace', 'text-anchor': anchor,
          'font-weight': o.now ? 600 : 400
        }, gLabels);
      }
      t.textContent = o.label;
    });
  }

  draw();
  if (mq.addEventListener) mq.addEventListener('change', draw);
  else if (mq.addListener) mq.addListener(draw);
})();
