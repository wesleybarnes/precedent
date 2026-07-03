import React, { useEffect, useState } from "react";
import { getBillChunks } from "../api.js";

// "Highlight which part of the text is being chunked." Given a bill, this
// fetches how its text splits under the *current* chunking params and renders
// the whole thing with each chunk as a distinctly coloured segment — the
// retrieved passage outlined so you can see exactly where in the bill it came
// from. Because the split is recomputed from the live chunk-size / overlap, this
// updates as you turn those sliders and re-run: the educational payoff of the
// whole chunking control.
const SEG_COLORS = [
  "#e7f0ff", "#fdeede", "#e8f7ec", "#f3e9fd", "#fde8ef", "#e6f6fb",
];

export default function ChunkMap({ billId, activeChunkId, vectorParams }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const chunkSize = vectorParams?.chunk_size;
  const chunkOverlap = vectorParams?.chunk_overlap;
  const chunkStrategy = vectorParams?.chunk_strategy;

  useEffect(() => {
    let alive = true;
    setData(null);
    setError(null);
    getBillChunks(billId, chunkSize, chunkOverlap, chunkStrategy)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e)));
    return () => { alive = false; };
  }, [billId, chunkSize, chunkOverlap, chunkStrategy]);

  if (error) return <div className="muted small">Could not load chunk map: {error}</div>;
  if (!data) return <div className="muted small">loading chunk map…</div>;

  return (
    <div className="chunkmap">
      <div className="chunkmap-head">
        <span><b>{data.chunks.length}</b> chunks · {chunkStrategy} · size {chunkSize} · overlap {chunkOverlap}</span>
        <span className="muted small">outlined = retrieved</span>
      </div>
      <div className="chunkmap-body">
        {data.chunks.map((c, i) => {
          const matched = c.chunk_id === activeChunkId;
          return (
            <span
              key={c.chunk_id}
              className={`seg ${matched ? "seg-matched" : ""}`}
              style={{ background: SEG_COLORS[i % SEG_COLORS.length] }}
              title={`chunk #${c.index}${matched ? " (retrieved)" : ""}`}
            >
              {c.text}{" "}
            </span>
          );
        })}
      </div>
      <div className="chunkmap-legend">
        {SEG_COLORS.slice(0, 3).map((col, i) => (
          <span key={i}><span className="swatch" style={{ background: col }} />chunk {i}</span>
        ))}
        <span className="muted">… each colour is one chunk the embedder sees. Overlap makes adjacent chunks share a little text.</span>
      </div>
    </div>
  );
}
