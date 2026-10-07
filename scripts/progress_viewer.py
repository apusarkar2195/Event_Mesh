#!/usr/bin/env python3
"""Web viewer for live ts-slam run progress.

Serves a small dashboard over HTTP so a run can be watched from another
machine on the network. Reads only the files a run already writes
(loss_log.jsonl, plot/stats_*.json, plot/evo_2dplot_*.png, timing.json);
it never touches the run itself.

Usage, from the repo root:

    python scripts/progress_viewer.py                 # serve ./results on port 8800
    python scripts/progress_viewer.py --port 9000
    python scripts/progress_viewer.py --results /path/to/results

Then open http://<this-machine>:8800 from the remote PC, or tunnel with
ssh -L 8800:localhost:8800 <user>@<this-machine> and open http://localhost:8800.

Uses only the Python standard library.
"""

import argparse
import json
import os
import re
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MAX_POINTS = 1500  # per series sent to the browser
LIVE_WINDOW_S = 60  # loss log written within this window counts as "live"

# ---------------------------------------------------------------------------
# Run discovery and parsing
# ---------------------------------------------------------------------------


def safe_join(root, rel):
    """Resolve rel under root; return None if it escapes root."""
    path = os.path.realpath(os.path.join(root, rel))
    root = os.path.realpath(root)
    if path == root or path.startswith(root + os.sep):
        return path
    return None


def list_runs(results_root, limit=40):
    runs = []
    try:
        scenes = sorted(os.listdir(results_root))
    except OSError:
        return runs
    for scene in scenes:
        scene_dir = os.path.join(results_root, scene)
        if not os.path.isdir(scene_dir):
            continue
        for run in os.listdir(scene_dir):
            run_dir = os.path.join(scene_dir, run)
            log = os.path.join(run_dir, "loss_log.jsonl")
            cfg = os.path.join(run_dir, "config.yml")
            if not (os.path.isfile(log) or os.path.isfile(cfg)):
                continue
            try:
                mtime = os.path.getmtime(log if os.path.isfile(log) else cfg)
            except OSError:
                continue
            runs.append(
                {
                    "rel": f"{scene}/{run}",
                    "scene": scene,
                    "run": run,
                    "mtime": mtime,
                    "live": (time.time() - mtime) < LIVE_WINDOW_S,
                }
            )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs[:limit]


def downsample(points):
    if len(points) <= MAX_POINTS:
        return points
    stride = -(-len(points) // MAX_POINTS)  # ceil division
    kept = points[::stride]
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept


_state_cache = {}


def run_state(results_root, rel):
    """Summarise one run directory. Cached on the loss log's (mtime, size)."""
    run_dir = safe_join(results_root, rel)
    if run_dir is None or not os.path.isdir(run_dir):
        return None

    log_path = os.path.join(run_dir, "loss_log.jsonl")
    try:
        stat = os.stat(log_path)
        cache_key = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stat, cache_key = None, None
    cached = _state_cache.get(rel)
    if cached is not None and cached[0] == cache_key and cache_key is not None:
        state = dict(cached[1])
        state["age_s"] = round(time.time() - stat.st_mtime, 1)
        state["live"] = state["age_s"] < LIVE_WINDOW_S
        return state

    tracking = []  # [frame, final tracking loss]
    triangles = []  # [frame, triangle count after densify]
    if stat is not None:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue  # partial line mid-write
                kind = entry.get("type")
                if kind == "tracking":
                    loss = entry.get("final_loss")
                    if loss is not None:
                        tracking.append([entry.get("frame_idx", 0), round(loss, 7)])
                elif kind == "densify":
                    tri = entry.get("tri_after")
                    if tri is not None:
                        triangles.append([entry.get("frame_idx", 0), tri])

    ate = []  # [checkpoint frame, ATE RMSE]
    plot_dir = os.path.join(run_dir, "plot")
    latest_plot = None
    if os.path.isdir(plot_dir):
        # Checkpoint filenames may or may not be zero-padded; keep the real
        # name rather than reconstructing it from the parsed frame number.
        stats, evo = [], []
        for name in os.listdir(plot_dir):
            m = re.fullmatch(r"stats_(\d+)\.json", name)
            if m:
                stats.append((int(m.group(1)), name))
            m = re.fullmatch(r"evo_2dplot_(\d+)\.png", name)
            if m:
                evo.append((int(m.group(1)), name))
        for n, name in sorted(stats):
            try:
                with open(os.path.join(plot_dir, name)) as f:
                    ate.append([n, round(json.load(f)["rmse"], 6)])
            except (OSError, ValueError, KeyError):
                continue
        if evo:
            n, name = max(evo)
            latest_plot = {"checkpoint": n, "url": f"/plot/{rel}/plot/{name}"}

    timing = None
    timing_path = os.path.join(run_dir, "timing.json")
    if os.path.isfile(timing_path):
        try:
            with open(timing_path) as f:
                t = json.load(f)
            timing = {
                "num_frames": t.get("num_frames"),
                "train_fps": t.get("train_fps"),
                "peak_gpu_gb": t.get("peak_gpu_gb"),
            }
        except (OSError, ValueError):
            pass

    dataset_path, sensor_type = None, None
    try:
        with open(os.path.join(run_dir, "config.yml"), errors="replace") as f:
            for line in f:
                m = re.match(r"\s*dataset_path:\s*(\S+)", line)
                if m:
                    dataset_path = m.group(1)
                m = re.match(r"\s*sensor_type:\s*(\S+)", line)
                if m:
                    sensor_type = m.group(1)
    except OSError:
        pass

    started = None
    try:
        started = os.path.getmtime(os.path.join(run_dir, "config.yml"))
    except OSError:
        pass
    last_write = stat.st_mtime if stat is not None else started
    elapsed = (last_write - started) if (started and last_write) else None

    state = {
        "rel": rel,
        "dataset_path": dataset_path,
        "sensor_type": sensor_type,
        "tracking": downsample(tracking),
        "triangles": downsample(triangles),
        "ate": ate,
        "latest_plot": latest_plot,
        "timing": timing,
        "frame": tracking[-1][0] if tracking else 0,
        "elapsed_s": round(elapsed, 1) if elapsed else None,
        "finished": timing is not None,
    }
    if cache_key is not None:
        _state_cache[rel] = (cache_key, dict(state))
    state["age_s"] = round(time.time() - last_write, 1) if last_write else None
    state["live"] = state["age_s"] is not None and state["age_s"] < LIVE_WINDOW_S
    return state


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    results_root = "results"  # overwritten in main()

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                self._json(list_runs(self.results_root))
            elif url.path == "/api/state":
                rel = parse_qs(url.query).get("run", [""])[0]
                state = run_state(self.results_root, rel) if rel else None
                if state is None:
                    self._json({"error": "run not found"}, 404)
                else:
                    self._json(state)
            elif url.path.startswith("/plot/"):
                rel = url.path[len("/plot/"):]
                path = safe_join(self.results_root, rel)
                if path and path.endswith(".png") and os.path.isfile(path):
                    with open(path, "rb") as f:
                        self._send(200, f.read(), "image/png")
                else:
                    self._json({"error": "not found"}, 404)
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return socket.gethostname()


# ---------------------------------------------------------------------------
# The page (all inline: no external requests, works over an SSH tunnel)
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ts-slam progress</title>
<style>
  :root {
    color-scheme: light dark;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
    --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
    --good: #006300; --bad: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface-1: #1a1a19; --page: #0d0d0d;
      --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
      --s1: #3987e5; --s2: #d95926; --s3: #199e70;
      --good: #0ca30c; --bad: #d03b3b;
    }
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--page); color: var(--ink);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 20px;
  }
  .wrap { max-width: 1100px; margin: 0 auto; }
  header { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 16px; }
  header h1 { font-size: 17px; font-weight: 600; margin-right: 4px; }
  select {
    font: inherit; color: var(--ink); background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 8px; padding: 5px 8px; max-width: 60vw;
  }
  .chip { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); font-size: 13px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.live { background: #0ca30c; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 14px; }
  .tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
  .tile .label { font-size: 12px; color: var(--ink-2); }
  .tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
  .tile .sub { font-size: 12px; color: var(--muted); margin-top: 1px; }
  .tile .sub .delta-good { color: var(--good); }
  .tile .sub .delta-bad { color: var(--bad); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }
  @media (max-width: 480px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; min-width: 0; }
  .card h2 { font-size: 13px; font-weight: 600; color: var(--ink-2); }
  .card .head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .card button {
    font: 12px system-ui, sans-serif; color: var(--ink-2); background: none;
    border: 1px solid var(--border); border-radius: 6px; padding: 2px 8px; cursor: pointer;
  }
  .card button:hover { color: var(--ink); }
  .chartbox { position: relative; }
  .chartbox svg { display: block; width: 100%; }
  .tooltip {
    position: absolute; pointer-events: none; display: none;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
    padding: 6px 9px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); white-space: nowrap; z-index: 2;
  }
  .tooltip .v { font-weight: 600; font-size: 13px; }
  .tooltip .k { color: var(--ink-2); }
  .tablebox { display: none; max-height: 240px; overflow: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; }
  th, td { text-align: right; padding: 3px 10px; font-variant-numeric: tabular-nums; border-bottom: 1px solid var(--grid); }
  th { color: var(--ink-2); font-weight: 600; position: sticky; top: 0; background: var(--surface-1); }
  .imgcard img { max-width: 100%; border-radius: 6px; }
  .empty { color: var(--muted); font-size: 13px; padding: 30px 0; text-align: center; }
  main.stale { opacity: 0.6; }
  footer { color: var(--muted); font-size: 12px; margin-top: 16px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>ts-slam progress</h1>
    <select id="runsel" aria-label="Run"></select>
    <span class="chip"><span class="dot" id="livedot"></span><span id="livetext">connecting…</span></span>
  </header>
  <main id="main">
    <section class="tiles" id="tiles"></section>
    <section class="grid">
      <div class="card" id="card-loss"></div>
      <div class="card" id="card-ate"></div>
      <div class="card" id="card-tri"></div>
      <div class="card imgcard" id="card-plot">
        <div class="head"><h2 id="plot-title">Trajectory</h2></div>
        <div class="empty" id="plot-empty">No trajectory plot yet</div>
        <img id="plot-img" alt="Estimated vs ground-truth trajectory" style="display:none">
      </div>
    </section>
  </main>
  <footer>Read-only view of <code>results/</code>. Refreshes every 2.5 s.</footer>
</div>
<script>
"use strict";
const $ = id => document.getElementById(id);
const fmtInt = v => v.toLocaleString("en-GB");
const fmtLoss = v => v.toPrecision(3);
const fmtAte = v => v.toFixed(4);

function niceTicks(lo, hi, n) {
  if (!(hi > lo)) { hi = lo + 1; }
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) ticks.push(v);
  return ticks;
}

// Single-series line chart: hairline grid, 2px line, end dot with surface
// ring, one selective end label, crosshair + tooltip, and a table twin.
function makeChart(cardId, opts) {
  const card = $(cardId);
  card.innerHTML = "";
  const head = document.createElement("div"); head.className = "head";
  const h2 = document.createElement("h2"); h2.textContent = opts.title;
  const btn = document.createElement("button"); btn.textContent = "Table";
  head.append(h2, btn); card.append(head);
  const box = document.createElement("div"); box.className = "chartbox"; card.append(box);
  const tbox = document.createElement("div"); tbox.className = "tablebox"; card.append(tbox);
  const tip = document.createElement("div"); tip.className = "tooltip"; box.append(tip);
  let data = [], showTable = false;

  btn.onclick = () => {
    showTable = !showTable;
    btn.textContent = showTable ? "Chart" : "Table";
    render();
  };

  function render() {
    tbox.style.display = showTable ? "block" : "none";
    let svg = box.querySelector("svg");
    if (svg) svg.remove();
    tip.style.display = "none";
    if (showTable) { renderTable(); return; }
    if (!data.length) {
      if (!box.querySelector(".empty")) {
        const e = document.createElement("div"); e.className = "empty";
        e.textContent = "Waiting for data"; box.append(e);
      }
      return;
    }
    const e = box.querySelector(".empty"); if (e) e.remove();

    const W = Math.max(box.clientWidth, 320), H = 230;
    const M = { l: 52, r: 14, t: 10, b: 26 };
    const xs = data.map(d => d[0]), ys = data.map(d => d[1]);
    const xlo = Math.min(...xs), xhi = Math.max(...xs);
    let ylo = Math.min(...ys), yhi = Math.max(...ys);
    if (ylo === yhi) {
      const mag = Math.max(Math.abs(ylo) * 0.5, 1e-6);
      ylo -= mag; yhi += mag;
    }
    const pad = (yhi - ylo) * 0.08; ylo -= pad; yhi += pad;
    const X = v => M.l + (v - xlo) / Math.max(xhi - xlo, 1e-9) * (W - M.l - M.r);
    const Y = v => H - M.b - (v - ylo) / (yhi - ylo) * (H - M.t - M.b);

    const ns = "http://www.w3.org/2000/svg";
    svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    const add = (tag, attrs, parent) => {
      const el = document.createElementNS(ns, tag);
      for (const k in attrs) el.setAttribute(k, attrs[k]);
      (parent || svg).append(el); return el;
    };

    for (const t of niceTicks(ylo, yhi, 4)) {
      add("line", { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), stroke: "var(--grid)", "shape-rendering": "crispEdges" });
      const lbl = add("text", { x: M.l - 6, y: Y(t) + 3.5, "text-anchor": "end", fill: "var(--muted)",
        "font-size": 11, style: "font-variant-numeric: tabular-nums" });
      lbl.textContent = opts.fmtTick(t);
    }
    add("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, stroke: "var(--axis)", "shape-rendering": "crispEdges" });
    for (const t of niceTicks(xlo, xhi, 6)) {
      const nearRight = X(t) > W - M.r - 18, nearLeftEdge = X(t) < M.l + 18;
      const lbl = add("text", { x: nearRight ? W - M.r : (nearLeftEdge ? M.l : X(t)),
        y: H - M.b + 15, "text-anchor": nearRight ? "end" : (nearLeftEdge ? "start" : "middle"),
        fill: "var(--muted)", "font-size": 11, style: "font-variant-numeric: tabular-nums" });
      lbl.textContent = fmtInt(Math.round(t));
    }

    const path = data.map((d, i) => `${i ? "L" : "M"}${X(d[0]).toFixed(1)},${Y(d[1]).toFixed(1)}`).join("");
    add("path", { d: path, fill: "none", stroke: opts.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" });
    if (opts.markers && data.length <= 60) {
      for (const d of data)
        add("circle", { cx: X(d[0]), cy: Y(d[1]), r: 4, fill: opts.color, stroke: "var(--surface-1)", "stroke-width": 2 });
    }
    const last = data[data.length - 1];
    add("circle", { cx: X(last[0]), cy: Y(last[1]), r: 4, fill: opts.color, stroke: "var(--surface-1)", "stroke-width": 2 });
    const nearLeft = X(last[0]) < M.l + 70;
    const endLbl = add("text", {
      x: nearLeft ? X(last[0]) + 8 : Math.min(X(last[0]), W - M.r - 4),
      y: Y(last[1]) - 7, "text-anchor": nearLeft ? "start" : "end",
      fill: "var(--ink-2)", "font-size": 11, "font-weight": 600 });
    endLbl.textContent = opts.fmtVal(last[1]);

    const hair = add("line", { y1: M.t, y2: H - M.b, stroke: "var(--axis)", visibility: "hidden", "shape-rendering": "crispEdges" });
    const hov = add("circle", { r: 4.5, fill: opts.color, stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" });
    const overlay = add("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b,
      fill: "transparent", style: "touch-action: none" });
    overlay.addEventListener("pointermove", ev => {
      const r = svg.getBoundingClientRect();
      const px = (ev.clientX - r.left) * (W / r.width);
      const xv = xlo + (px - M.l) / Math.max(W - M.l - M.r, 1) * (xhi - xlo);
      let best = 0;
      for (let i = 1; i < data.length; i++)
        if (Math.abs(data[i][0] - xv) < Math.abs(data[best][0] - xv)) best = i;
      const d = data[best];
      hair.setAttribute("x1", X(d[0])); hair.setAttribute("x2", X(d[0]));
      hair.setAttribute("visibility", "visible");
      hov.setAttribute("cx", X(d[0])); hov.setAttribute("cy", Y(d[1]));
      hov.setAttribute("visibility", "visible");
      tip.replaceChildren();
      const v = document.createElement("div"); v.className = "v"; v.textContent = opts.fmtVal(d[1]);
      const k = document.createElement("div"); k.className = "k"; k.textContent = `frame ${fmtInt(d[0])}`;
      tip.append(v, k); tip.style.display = "block";
      const bx = box.getBoundingClientRect();
      const tx = Math.min((X(d[0]) / W) * bx.width + 12, bx.width - tip.offsetWidth - 4);
      tip.style.left = `${Math.max(tx, 0)}px`;
      tip.style.top = `${(Y(d[1]) / H) * bx.height - tip.offsetHeight - 10}px`;
    });
    overlay.addEventListener("pointerleave", () => {
      hair.setAttribute("visibility", "hidden");
      hov.setAttribute("visibility", "hidden");
      tip.style.display = "none";
    });
    box.append(svg);
  }

  function renderTable() {
    tbox.replaceChildren();
    const table = document.createElement("table");
    const tr = document.createElement("tr");
    for (const h of ["Frame", opts.title]) {
      const th = document.createElement("th"); th.textContent = h; tr.append(th);
    }
    table.append(tr);
    for (let i = data.length - 1; i >= 0; i--) {
      const row = document.createElement("tr");
      const a = document.createElement("td"); a.textContent = fmtInt(data[i][0]);
      const b = document.createElement("td"); b.textContent = opts.fmtVal(data[i][1]);
      row.append(a, b); table.append(row);
    }
    tbox.append(table);
  }

  let renderedOnce = false;
  return {
    update(points) {
      const changed = points.length !== data.length ||
        (points.length && points[points.length - 1][1] !== data[data.length - 1][1]);
      data = points;
      if (changed || !renderedOnce) { renderedOnce = true; render(); }
    },
    redraw: render,
  };
}

const lossChart = makeChart("card-loss", {
  title: "Tracking loss (final, per frame)", color: "var(--s1)",
  fmtVal: fmtLoss, fmtTick: v => v.toPrecision(2),
});
const ateChart = makeChart("card-ate", {
  title: "ATE RMSE (m, per checkpoint)", color: "var(--s2)", markers: true,
  fmtVal: v => `${fmtAte(v)} m`, fmtTick: v => v.toFixed(3),
});
const triChart = makeChart("card-tri", {
  title: "Triangles", color: "var(--s3)",
  fmtVal: fmtInt,
  fmtTick: v => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : `${Math.round(v)}`),
});

function tile(label, value, sub, subClass) {
  const t = document.createElement("div"); t.className = "tile";
  const l = document.createElement("div"); l.className = "label"; l.textContent = label;
  const v = document.createElement("div"); v.className = "value"; v.textContent = value;
  t.append(l, v);
  if (sub) {
    const s = document.createElement("div"); s.className = "sub";
    if (subClass) {
      const span = document.createElement("span"); span.className = subClass; span.textContent = sub;
      s.append(span);
    } else { s.textContent = sub; }
    t.append(s);
  }
  return t;
}

function fmtDur(s) {
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`;
}

let currentRun = null, lastPlotCheckpoint = null, pollN = 0;

function renderState(st) {
  const tiles = $("tiles");
  tiles.replaceChildren();

  const total = st.timing && st.timing.num_frames;
  tiles.append(tile("Frame", fmtInt(st.frame), total ? `of ${fmtInt(total)}` : (st.dataset_path || "")));

  const tr = st.tracking;
  tiles.append(tile("Tracking loss", tr.length ? fmtLoss(tr[tr.length - 1][1]) : "—",
    tr.length ? `frame ${fmtInt(tr[tr.length - 1][0])}` : ""));

  if (st.ate.length) {
    const last = st.ate[st.ate.length - 1], prev = st.ate[st.ate.length - 2];
    let sub = `at frame ${fmtInt(last[0])}`, cls = null;
    if (prev) {
      const d = last[1] - prev[1];
      sub = `${d >= 0 ? "+" : "−"}${fmtAte(Math.abs(d))} vs frame ${fmtInt(prev[0])}`;
      cls = d <= 0 ? "delta-good" : "delta-bad";
    }
    tiles.append(tile("ATE RMSE", `${fmtAte(last[1])} m`, sub, cls));
  } else {
    tiles.append(tile("ATE RMSE", "—", "no checkpoint yet"));
  }

  const tris = st.triangles;
  tiles.append(tile("Triangles", tris.length ? fmtInt(tris[tris.length - 1][1]) : "—"));

  if (st.finished && st.timing) {
    tiles.append(tile("Speed", `${(st.timing.train_fps || 0).toFixed(2)} fr/s`,
      st.timing.peak_gpu_gb ? `peak ${st.timing.peak_gpu_gb.toFixed(1)} GB GPU` : ""));
  } else if (st.elapsed_s && st.frame > 0) {
    tiles.append(tile("Speed", `${(st.frame / st.elapsed_s).toFixed(2)} fr/s`,
      `running ${fmtDur(st.elapsed_s)}`));
  }

  lossChart.update(st.tracking);
  ateChart.update(st.ate);
  triChart.update(st.triangles);

  const img = $("plot-img"), empty = $("plot-empty");
  if (st.latest_plot) {
    if (st.latest_plot.checkpoint !== lastPlotCheckpoint) {
      lastPlotCheckpoint = st.latest_plot.checkpoint;
      img.src = st.latest_plot.url;
      $("plot-title").textContent = `Trajectory (checkpoint ${fmtInt(st.latest_plot.checkpoint)})`;
    }
    img.style.display = "block"; empty.style.display = "none";
  } else {
    img.style.display = "none"; empty.style.display = "block";
  }

  const dot = $("livedot"), txt = $("livetext");
  dot.classList.toggle("live", !!st.live);
  if (st.finished) txt.textContent = "finished";
  else if (st.live) txt.textContent = "live";
  else txt.textContent = st.age_s != null ? `idle, last write ${fmtDur(st.age_s)} ago` : "idle";
}

const urlRun = new URLSearchParams(location.search).get("run");

async function refreshRuns(keepSelection) {
  const runs = await (await fetch("/api/runs")).json();
  const sel = $("runsel");
  const prev = keepSelection ? sel.value : null;
  sel.replaceChildren();
  for (const r of runs) {
    const o = document.createElement("option");
    o.value = r.rel;
    o.textContent = `${r.scene} / ${r.run}${r.live ? "  (live)" : ""}`;
    sel.append(o);
  }
  if (prev && runs.some(r => r.rel === prev)) sel.value = prev;
  else if (!keepSelection && urlRun && runs.some(r => r.rel === urlRun)) sel.value = urlRun;
  else if (runs.length) sel.value = (runs.find(r => r.live) || runs[0]).rel;
  return sel.value || null;
}

async function poll() {
  try {
    if (pollN % 10 === 0) {
      const chosen = await refreshRuns(pollN > 0);
      if (chosen && chosen !== currentRun) { currentRun = chosen; lastPlotCheckpoint = null; }
    }
    pollN++;
    if (currentRun) {
      const res = await fetch(`/api/state?run=${encodeURIComponent(currentRun)}`);
      if (res.ok) {
        renderState(await res.json());
        $("main").classList.remove("stale");
      }
    }
  } catch (err) {
    $("main").classList.add("stale");
    $("livetext").textContent = "reconnecting…";
    $("livedot").classList.remove("live");
  }
  setTimeout(poll, 2500);
}

$("runsel").addEventListener("change", ev => {
  currentRun = ev.target.value;
  lastPlotCheckpoint = null;
  history.replaceState(null, "", `?run=${encodeURIComponent(currentRun)}`);
  fetch(`/api/state?run=${encodeURIComponent(currentRun)}`)
    .then(r => r.json()).then(renderState).catch(() => {});
});
let resizeT = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeT);
  resizeT = setTimeout(() => { lossChart.redraw(); ateChart.redraw(); triChart.redraw(); }, 150);
});
poll();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=None, help="results directory (default: <repo>/results)")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.abspath(args.results or os.path.join(repo_root, "results"))
    if not os.path.isdir(results_root):
        ap.error(f"results directory not found: {results_root}")

    Handler.results_root = results_root
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {results_root}")
    print(f"  local:   http://localhost:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  network: http://{lan_ip()}:{args.port}")
    print("Stop with Ctrl-C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
