import React, { useState } from "react";
import InfoTip from "./InfoTip.jsx";

// The tunable RAG knobs. Rendered generically from the /params schema (which now
// carries a `help` description per control), so the backend decides what's
// adjustable *and* how it's explained. Every knob has an ⓘ tooltip with its
// trade-offs, and the chunking-strategy dropdown shows the chosen strategy's
// description inline — this is the educational surface.
export default function Controls({
  schema, graphParams, vectorParams, setGraphParams, setVectorParams, disabled,
}) {
  const [open, setOpen] = useState(true);

  return (
    <section className="controls">
      <button className="controls-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "▾" : "▸"} RAG parameters — tune these and re-run to see the effect
      </button>
      {open && (
        <div className="controls-grid">
          <ParamGroup
            title="GraphRAG" accent="graph" controls={schema.graph.controls}
            values={graphParams} onChange={(k, v) => setGraphParams({ ...graphParams, [k]: v })}
            disabled={disabled}
          />
          <ParamGroup
            title="Vector RAG" accent="vector" controls={schema.vector.controls}
            values={vectorParams} onChange={(k, v) => setVectorParams({ ...vectorParams, [k]: v })}
            disabled={disabled}
          />
        </div>
      )}
    </section>
  );
}

function ParamGroup({ title, accent, controls, values, onChange, disabled }) {
  return (
    <div className={`param-group accent-${accent}`}>
      <h3>{title}</h3>
      {controls.map((c) =>
        c.type === "select" ? (
          <SelectParam key={c.key} c={c} value={values[c.key]} disabled={disabled}
                       onChange={(v) => onChange(c.key, v)} />
        ) : (
          <label key={c.key} className="param">
            <span className="param-label">
              <span>{c.label} <InfoTip text={c.help} /></span>
              <b>{values[c.key]}</b>
            </span>
            <input
              type="range" min={c.min} max={c.max} step={c.step}
              value={values[c.key]} disabled={disabled}
              onChange={(e) => onChange(c.key, Number(e.target.value))}
            />
          </label>
        )
      )}
    </div>
  );
}

function SelectParam({ c, value, onChange, disabled }) {
  const chosen = (c.options || []).find((o) => o.value === value);
  return (
    <label className="param">
      <span className="param-label">
        <span>{c.label} <InfoTip text={c.help} /></span>
      </span>
      <select className="param-select" value={value} disabled={disabled}
              onChange={(e) => onChange(e.target.value)}>
        {c.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
      {chosen?.help && <p className="param-help">{chosen.help}</p>}
    </label>
  );
}
