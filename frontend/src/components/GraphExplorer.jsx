import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide, forceX, forceY,
} from "d3-force";
import { getGraphFull, getGraphMeta } from "../api.js";
import { outcomeStyle } from "../theme.js";
import InfoTip from "./InfoTip.jsx";

// Plain-English glossary of the bill-type codes, so a user never has to wonder
// what "H.Res." or "S.J.Res." means.
const BILL_TYPE_GLOSSARY = [
  ["H.R.", "House Bill"], ["S.", "Senate Bill"],
  ["H.Res.", "House Resolution"], ["S.Res.", "Senate Resolution"],
  ["H.J.Res.", "House Joint Resolution"], ["H.Con.Res.", "House Concurrent Resolution"],
];

// A standalone explorer of the whole knowledge-graph space: every bill is a
// node, and two bills are linked when they share sponsors and/or committees.
// This is where you *see the relationships between bills* directly — pan, zoom,
// filter by subject/congress, and click a bill to light up everything connected
// to it. Node colour = outcome, node size = how connected it is, edge thickness
// = how much two bills share.
const W = 900;
const H = 620;

export default function GraphExplorer() {
  const [meta, setMeta] = useState({ subjects: [], congresses: [] });
  const [subject, setSubject] = useState("");
  const [congress, setCongress] = useState("");
  const [minShared, setMinShared] = useState(1);
  const [limit, setLimit] = useState(80);
  const [data, setData] = useState({ nodes: [], edges: [], total_bills: 0, shown: 0 });
  const [loading, setLoading] = useState(true);
  const [positions, setPositions] = useState({});
  const [selected, setSelected] = useState(null);
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });

  const svgRef = useRef(null);
  const drag = useRef(null);

  useEffect(() => { getGraphMeta().then(setMeta).catch(() => {}); }, []);

  // Fetch the graph whenever a filter changes.
  useEffect(() => {
    let alive = true;
    setLoading(true);
    getGraphFull({ subject, congress, minShared, limit })
      .then((d) => { if (alive) { setData(d); setSelected(null); } })
      .catch(() => {})
      .finally(() => alive && setLoading(false));
    return () => { alive = false; };
  }, [subject, congress, minShared, limit]);

  // Lay the graph out once per data change with a static force pass.
  useEffect(() => {
    if (!data.nodes.length) { setPositions({}); return; }
    const nodes = data.nodes.map((n) => ({ ...n }));
    const links = data.edges.map((e) => ({ ...e }));
    const sim = forceSimulation(nodes)
      .force("link", forceLink(links).id((d) => d.id).distance(60).strength(0.4))
      .force("charge", forceManyBody().strength(-160))
      .force("center", forceCenter(W / 2, H / 2))
      .force("collide", forceCollide((d) => radius(d) + 3))
      .force("x", forceX(W / 2).strength(0.04))
      .force("y", forceY(H / 2).strength(0.04))
      .stop();
    for (let i = 0; i < 300; i++) sim.tick();
    const pos = {};
    nodes.forEach((n) => { pos[n.id] = { x: n.x, y: n.y }; });
    setPositions(pos);
    setView({ x: 0, y: 0, k: 1 });
  }, [data]);

  // Neighbor lookup for highlighting on selection.
  const neighbors = useMemo(() => {
    const m = {};
    data.edges.forEach((e) => {
      (m[e.source] = m[e.source] || new Set()).add(e.target);
      (m[e.target] = m[e.target] || new Set()).add(e.source);
    });
    return m;
  }, [data]);

  const isLit = (id) =>
    !selected || id === selected || (neighbors[selected] && neighbors[selected].has(id));
  const edgeLit = (e) => !selected || e.source === selected || e.target === selected;

  // --- pan / zoom / drag ---
  const toWorld = (clientX, clientY) => {
    const rect = svgRef.current.getBoundingClientRect();
    const sx = (clientX - rect.left) * (W / rect.width);
    const sy = (clientY - rect.top) * (H / rect.height);
    return { x: (sx - view.x) / view.k, y: (sy - view.y) / view.k };
  };
  const onWheel = (e) => {
    e.preventDefault();
    const rect = svgRef.current.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (W / rect.width);
    const sy = (e.clientY - rect.top) * (H / rect.height);
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const k = Math.max(0.3, Math.min(4, view.k * factor));
    setView({ k, x: sx - (sx - view.x) * (k / view.k), y: sy - (sy - view.y) * (k / view.k) });
  };
  const onPointerDown = (e, nodeId) => {
    e.target.setPointerCapture?.(e.pointerId);
    drag.current = { nodeId, moved: false, startX: e.clientX, startY: e.clientY };
  };
  const onPointerMove = (e) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.startX;
    const dy = e.clientY - drag.current.startY;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
    if (drag.current.nodeId) {
      const w = toWorld(e.clientX, e.clientY);
      setPositions((p) => ({ ...p, [drag.current.nodeId]: { x: w.x, y: w.y } }));
    } else {
      setView((v) => ({ ...v, x: v.x + dx * (W / svgRef.current.getBoundingClientRect().width),
                              y: v.y + dy * (H / svgRef.current.getBoundingClientRect().height) }));
      drag.current.startX = e.clientX; drag.current.startY = e.clientY;
    }
  };
  const onPointerUp = (e) => {
    if (drag.current && drag.current.nodeId && !drag.current.moved) {
      setSelected((s) => (s === drag.current.nodeId ? null : drag.current.nodeId));
    }
    drag.current = null;
  };

  const selNode = selected ? data.nodes.find((n) => n.id === selected) : null;
  const labelOf = useMemo(() => {
    const m = {};
    data.nodes.forEach((n) => { m[n.id] = n.label || n.id; });
    return m;
  }, [data]);

  return (
    <section className="explorer">
      <div className="explorer-intro">
        <h2>The knowledge graph</h2>
        <p>
          This is the structured world GraphRAG reasons over. Every dot is a <b>bill</b>;
          a line connects two bills that share a <b>sponsor</b> or a <b>committee</b> — the
          thicker the line, the more they share. Bills cluster into communities of related
          legislation, which is exactly how GraphRAG finds precedents: it walks these links
          instead of matching words. <b>Click a bill</b> to light up everything it's connected
          to; scroll to zoom, drag to pan, drag a dot to move it.
        </p>
        <div className="glossary">
          <span className="muted">bill types:</span>
          {BILL_TYPE_GLOSSARY.map(([short, name]) => (
            <span key={short} className="gloss-item"><b>{short}</b> {name}</span>
          ))}
        </div>
      </div>

      <div className="explorer-controls">
        <Filter label="Subject" help="Show only bills tagged with this policy area.">
          <select value={subject} onChange={(e) => setSubject(e.target.value)}>
            <option value="">all subjects</option>
            {meta.subjects.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </Filter>
        <Filter label="Congress" help="Each Congress lasts two years (the 118th ran 2023–2025). Filter to one.">
          <select value={congress} onChange={(e) => setCongress(e.target.value)}>
            <option value="">all</option>
            {meta.congresses.map((c) => <option key={c} value={c}>{c}th</option>)}
          </select>
        </Filter>
        <Filter label={`Min shared: ${minShared}`}
                help="Only draw a link when two bills share at least this many sponsors/committees. Raise it to keep just the strongest relationships.">
          <input type="range" min={1} max={4} step={1} value={minShared}
                 onChange={(e) => setMinShared(Number(e.target.value))} />
        </Filter>
        <Filter label={`Max bills: ${limit}`}
                help="Cap the graph to the most-connected bills so a large corpus stays readable.">
          <input type="range" min={20} max={200} step={20} value={limit}
                 onChange={(e) => setLimit(Number(e.target.value))} />
        </Filter>
        <span className="explorer-count muted">
          {loading ? "loading…" : `showing ${data.shown} of ${data.total_bills} bills · ${data.edges.length} links`}
        </span>
      </div>

      <div className="explorer-stage">
        <svg
          ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="explorer-svg"
          onWheel={onWheel}
          onPointerDown={(e) => { if (e.target === e.currentTarget || e.target.tagName === "rect") onPointerDown(e, null); }}
          onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerLeave={onPointerUp}
        >
          <rect x="0" y="0" width={W} height={H} fill="transparent" />
          <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
            {data.edges.map((e, i) => {
              const s = positions[e.source]; const t = positions[e.target];
              if (!s || !t) return null;
              return (
                <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                      className={`ex-edge ex-edge-${e.rel}`}
                      strokeWidth={Math.min(5, 0.8 + e.weight * 0.7)}
                      opacity={edgeLit(e) ? 0.6 : 0.06} />
              );
            })}
            {data.nodes.map((n) => {
              const p = positions[n.id]; if (!p) return null;
              const st = outcomeStyle(n.outcome);
              const r = radius(n);
              const lit = isLit(n.id);
              return (
                <g key={n.id} transform={`translate(${p.x},${p.y})`}
                   style={{ cursor: "pointer" }}
                   onPointerDown={(e) => onPointerDown(e, n.id)}>
                  <circle r={r} fill={st.color} opacity={lit ? 1 : 0.2}
                          stroke={n.id === selected ? "#131a2b" : "white"}
                          strokeWidth={n.id === selected ? 2.5 : 1.5} className="ex-node" />
                  {(view.k > 1.4 || n.id === selected) && lit && (
                    <text className="ex-label" y={r + 10}>{n.label || n.id}</text>
                  )}
                </g>
              );
            })}
          </g>
        </svg>

        {selNode && <DetailPanel node={selNode} data={data} labelOf={labelOf}
                                 onSelect={setSelected} onClose={() => setSelected(null)} />}
      </div>

      <div className="explorer-legend">
        <span className="muted">outcome:</span>
        {["became_law", "vetoed", "died_in_committee", "passed_one_chamber_pending"].map((o) => {
          const s = outcomeStyle(o);
          return <span key={o} className="legend-item">
            <span className="legend-dot" style={{ background: s.color }} /> {s.label}</span>;
        })}
        <span className="muted">· node size = connectedness · scroll to zoom, drag to pan, drag a node to move it</span>
      </div>
    </section>
  );
}

function radius(n) { return 6 + Math.min(16, (n.degree || 0) * 1.1); }

function Filter({ label, help, children }) {
  return (
    <label className="explorer-filter">
      <span className="muted">{label} <InfoTip text={help} /></span>
      {children}
    </label>
  );
}

function DetailPanel({ node, data, labelOf, onSelect, onClose }) {
  const st = outcomeStyle(node.outcome);
  const connected = data.edges
    .filter((e) => e.source === node.id || e.target === node.id)
    .map((e) => ({ id: e.source === node.id ? e.target : e.source, weight: e.weight, rel: e.rel }))
    .sort((a, b) => b.weight - a.weight);
  return (
    <div className="detail-panel">
      <button className="detail-close" onClick={onClose}>✕</button>
      <div className="detail-id">{node.label || node.id}</div>
      <div className="detail-type muted small">{node.type_name} · {node.id}</div>
      <div className="detail-title">{node.title}</div>
      <span className="outcome-chip" style={{ color: st.color, background: st.bg }}>{st.label}</span>
      <div className="detail-subjects">
        {node.subjects.slice(0, 8).map((s) => <span key={s} className="tag">{s}</span>)}
      </div>
      <h4>Connected bills ({connected.length})</h4>
      <ul className="detail-links">
        {connected.map((c) => (
          <li key={c.id} onClick={() => onSelect(c.id)}>
            <b>{labelOf[c.id] || c.id}</b>{" "}
            <span className="muted small">shares {c.weight} {c.rel === "both" ? "sponsor+committee" : c.rel}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
