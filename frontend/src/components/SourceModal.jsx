import React, { useEffect, useState } from "react";
import { getSource } from "../api.js";

// "Peer into the code." When a step's view-source is clicked, this fetches the
// actual source of the function that produced that step from the backend's
// /source endpoint (via inspect) and shows it with line numbers. It's what turns
// the visualizer from a demo into something a developer can actually learn the
// implementation from.
export default function SourceModal({ symbol, onClose }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setData(null);
    setError(null);
    getSource(symbol).then(setData).catch((e) => setError(String(e)));
  }, [symbol]);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const startLine = data?.start_line || 1;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">{symbol.split(".").pop()}</div>
            {data && <div className="muted small">{data.file}:{data.start_line}</div>}
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          {error && <div className="error-banner">{error}</div>}
          {!data && !error && <div className="muted">loading source…</div>}
          {data && (
            <pre className="code">
              {data.source.split("\n").map((line, i) => (
                <div key={i} className="code-line">
                  <span className="ln">{startLine + i}</span>
                  <span className="lc">{line || " "}</span>
                </div>
              ))}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
