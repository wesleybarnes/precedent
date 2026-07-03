import React, { useState } from "react";
import { heatColor } from "../theme.js";
import OutcomeChip from "./OutcomeChip.jsx";
import ChunkMap from "./ChunkMap.jsx";

// The vector engine's payoff: the ranked passages. Each is tinted by a
// similarity heat scale (hotter = more similar) so relevance is visible at a
// glance, and each can expand into a ChunkMap showing where in its bill's full
// text that passage sits — i.e. exactly how the bill was chunked.
export default function ChunkList({ chunks, vectorParams }) {
  if (!chunks || chunks.length === 0) {
    return <div className="muted">No similar passages found.</div>;
  }
  return (
    <ol className="chunks">
      {chunks.map((c, i) => (
        <ChunkItem key={c.chunk_id} chunk={c} rank={i + 1} vectorParams={vectorParams} />
      ))}
    </ol>
  );
}

function ChunkItem({ chunk, rank, vectorParams }) {
  const [showMap, setShowMap] = useState(false);
  return (
    <li className="chunk" style={{ borderColor: heatColor(chunk.similarity, 0.9) }}>
      <div className="chunk-head">
        <span className="rank">#{rank}</span>
        <b>{chunk.bill_id}</b>
        <OutcomeChip outcome={chunk.outcome} />
        <span className="sim">{chunk.similarity.toFixed(3)}</span>
      </div>
      <div className="sim-bar">
        <div className="sim-fill" style={{ width: `${Math.max(3, chunk.similarity * 100)}%` }} />
      </div>
      <p className="chunk-text" style={{ background: heatColor(chunk.similarity, 0.5) }}>
        {chunk.text}
      </p>
      <div className="chunk-actions">
        <button className="link-btn" onClick={() => setShowMap((v) => !v)}>
          {showMap ? "▾ hide chunk map" : "▸ show how this bill was chunked"}
        </button>
      </div>
      {showMap && (
        <ChunkMap
          billId={chunk.bill_id}
          activeChunkId={chunk.chunk_id}
          vectorParams={vectorParams}
        />
      )}
    </li>
  );
}
