import React, { useEffect, useRef, useState } from "react";
import {
  forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide,
} from "d3-force";
import { NODE_COLORS as COLORS, NODE_LABEL as KIND_LABEL } from "../theme.js";

// The GraphRAG subgraph, drawn as an interactive force-directed diagram. This is
// the visual payoff of the graph engine: you can see the seed bills, the shared
// sponsors and committees that connect them, and the precedent bills the
// traversal reached through those connections. Colour encodes node kind; size
// encodes a precedent's graph score.

const WIDTH = 460;
const HEIGHT = 340;

export default function GraphView({ subgraph }) {
  const { nodes, edges } = subgraph || { nodes: [], edges: [] };
  const [positions, setPositions] = useState({});
  const [hover, setHover] = useState(null);
  const simRef = useRef(null);

  useEffect(() => {
    if (!nodes || nodes.length === 0) return;
    // Work on copies: d3-force mutates the objects it's given with x/y/vx/vy.
    const simNodes = nodes.map((n) => ({ ...n }));
    const simLinks = edges.map((e) => ({ ...e }));

    const sim = forceSimulation(simNodes)
      .force("link", forceLink(simLinks).id((d) => d.id).distance(70).strength(0.6))
      .force("charge", forceManyBody().strength(-240))
      .force("center", forceCenter(WIDTH / 2, HEIGHT / 2))
      .force("collide", forceCollide(26))
      .stop();

    // Run the layout synchronously for a fixed number of ticks, then render the
    // settled positions -- cheap for these small subgraphs and avoids animating
    // on every frame.
    for (let i = 0; i < 200; i++) sim.tick();
    const pos = {};
    simNodes.forEach((n) => {
      pos[n.id] = {
        x: Math.max(24, Math.min(WIDTH - 24, n.x)),
        y: Math.max(24, Math.min(HEIGHT - 24, n.y)),
      };
    });
    setPositions(pos);
    simRef.current = sim;
  }, [subgraph]);

  if (!nodes || nodes.length === 0) {
    return <div className="muted">No connected precedents to graph.</div>;
  }

  const nodeSize = (n) =>
    n.kind === "precedent_bill" ? 9 + Math.min(10, (n.score || 0) * 1.5)
      : n.kind === "seed_bill" ? 12 : 7;

  return (
    <div className="graphview">
      <svg width={WIDTH} height={HEIGHT} className="graph-svg">
        {edges.map((e, i) => {
          const s = positions[e.source]; const t = positions[e.target];
          if (!s || !t) return null;
          return (
            <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                  className={`edge edge-${e.rel}`} />
          );
        })}
        {nodes.map((n) => {
          const p = positions[n.id];
          if (!p) return null;
          const r = nodeSize(n);
          return (
            <g key={n.id} transform={`translate(${p.x},${p.y})`}
               onMouseEnter={() => setHover(n)} onMouseLeave={() => setHover(null)}>
              <circle r={r} fill={COLORS[n.kind] || "#888"}
                      className={n.kind.endsWith("bill") ? "node node-bill" : "node"} />
              {n.kind.endsWith("bill") && (
                <text className="node-label" y={r + 11}>{n.label}</text>
              )}
            </g>
          );
        })}
      </svg>

      <div className="legend">
        {Object.entries(KIND_LABEL).map(([kind, label]) => (
          <span key={kind} className="legend-item">
            <span className="legend-dot" style={{ background: COLORS[kind] }} /> {label}
          </span>
        ))}
      </div>

      {hover && (
        <div className="graph-tip">
          <b>{hover.label}</b> <span className="muted">({KIND_LABEL[hover.kind]})</span>
          {hover.title && <div>{hover.title}</div>}
          {hover.outcome && <div className="muted">outcome: {hover.outcome}</div>}
          {hover.score != null && <div className="muted">score: {hover.score}</div>}
          {hover.party && <div className="muted">{hover.party}-{hover.state}</div>}
        </div>
      )}
    </div>
  );
}
