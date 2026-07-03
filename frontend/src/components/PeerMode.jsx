import React, { useEffect, useMemo, useRef, useState } from "react";
import { getSource } from "../api.js";

// "Peer into the pipeline." A slowed-down, step-by-step replay of a completed
// run that walks through every intricacy of both engines in order — showing,
// for each step, what it did, the real intermediate data it produced, and the
// actual source code that ran. Play it hands-free, or step through manually.
export default function PeerMode({ steps, onClose }) {
  // Interleave the two engines' done-steps into one timeline, mirroring how they
  // executed side by side.
  const timeline = useMemo(() => {
    const g = (steps.graph || []).filter((s) => s.phase === "done");
    const v = (steps.vector || []).filter((s) => s.phase === "done");
    const out = [];
    const n = Math.max(g.length, v.length);
    for (let i = 0; i < n; i++) {
      if (g[i]) out.push(g[i]);
      if (v[i]) out.push(v[i]);
    }
    return out;
  }, [steps]);

  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1600);
  const [sources, setSources] = useState({});
  const timer = useRef(null);

  const current = timeline[cursor];

  // Auto-advance while playing.
  useEffect(() => {
    if (!playing) return;
    if (cursor >= timeline.length - 1) { setPlaying(false); return; }
    timer.current = setTimeout(() => setCursor((c) => c + 1), speed);
    return () => clearTimeout(timer.current);
  }, [playing, cursor, speed, timeline.length]);

  // Lazily fetch + cache the source for the current step.
  useEffect(() => {
    const sym = current?.source_symbol;
    if (!sym || sources[sym]) return;
    getSource(sym).then((d) => setSources((s) => ({ ...s, [sym]: d }))).catch(() => {});
  }, [current, sources]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") { setPlaying(false); setCursor((c) => Math.min(timeline.length - 1, c + 1)); }
      if (e.key === "ArrowLeft") { setPlaying(false); setCursor((c) => Math.max(0, c - 1)); }
      if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [timeline.length, onClose]);

  if (!current) {
    return (
      <div className="peer-overlay" onClick={onClose}>
        <div className="peer-panel" onClick={(e) => e.stopPropagation()}>
          <p className="muted">Run a query first, then peer into it.</p>
        </div>
      </div>
    );
  }

  const src = sources[current.source_symbol];

  return (
    <div className="peer-overlay" onClick={onClose}>
      <div className="peer-panel" onClick={(e) => e.stopPropagation()}>
        <div className="peer-head">
          <div className="peer-title">
            🔬 Peer mode — <span className={`peer-engine peer-${current.engine}`}>{current.engine === "graph" ? "GraphRAG" : "Vector RAG"}</span>
          </div>
          <div className="peer-controls">
            <button onClick={() => { setPlaying(false); setCursor((c) => Math.max(0, c - 1)); }}>◀</button>
            <button onClick={() => setPlaying((p) => !p)}>{playing ? "❚❚ pause" : "▶ play"}</button>
            <button onClick={() => { setPlaying(false); setCursor((c) => Math.min(timeline.length - 1, c + 1)); }}>▶</button>
            <label className="peer-speed">
              speed
              <input type="range" min={400} max={3000} step={200} value={3400 - speed}
                     onChange={(e) => setSpeed(3400 - Number(e.target.value))} />
            </label>
            <button className="peer-close" onClick={onClose}>✕</button>
          </div>
        </div>

        <div className="peer-progress">
          {timeline.map((s, i) => (
            <span key={i}
                  className={`peer-tick peer-${s.engine} ${i === cursor ? "peer-tick-on" : ""} ${i < cursor ? "peer-tick-past" : ""}`}
                  onClick={() => { setPlaying(false); setCursor(i); }}
                  title={`${s.engine}: ${s.title}`} />
          ))}
        </div>

        <div className="peer-body">
          <div className="peer-left">
            <div className="peer-step-title">{current.title}</div>
            <div className="peer-step-dur">{current.duration_ms} ms · step {cursor + 1} of {timeline.length}</div>
            <p className="peer-desc">{current.description}</p>
            <h4>Intermediate data</h4>
            <Payload payload={current.payload} />
          </div>
          <div className="peer-right">
            <div className="peer-code-head">
              <span>{current.source_symbol}</span>
              {src && <span className="muted small">{src.file}:{src.start_line}</span>}
            </div>
            <pre className="code peer-code">
              {src
                ? src.source.split("\n").map((line, i) => (
                    <div key={i} className="code-line">
                      <span className="ln">{src.start_line + i}</span>
                      <span className="lc">{line || " "}</span>
                    </div>
                  ))
                : <div className="muted" style={{ padding: 12 }}>loading source…</div>}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}

// Render intermediate data readably: a compact key/value view, with arrays and
// objects pretty-printed. The point is to make the "little intricacies" visible.
function Payload({ payload }) {
  if (!payload || Object.keys(payload).length === 0)
    return <span className="muted">no data for this step</span>;
  return (
    <div className="peer-kv">
      {Object.entries(payload).map(([k, v]) => (
        <div key={k} className="peer-kv-row">
          <div className="peer-kv-k">{k}</div>
          <div className="peer-kv-v">
            {typeof v === "object"
              ? <pre className="peer-json">{JSON.stringify(v, null, 2)}</pre>
              : String(v)}
          </div>
        </div>
      ))}
    </div>
  );
}
