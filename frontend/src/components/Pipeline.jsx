import React, { useState } from "react";
import GraphView from "./GraphView.jsx";
import ChunkList from "./ChunkList.jsx";
import OutcomeChip from "./OutcomeChip.jsx";

// One engine's lane: the ordered steps of its pipeline, each animating from
// "running" to "done" as the stream delivers events, plus the final answer.
export default function Pipeline({ title, tag, subtitle, accent, steps, onPeek, vectorParams }) {
  const answerStep = steps.find((s) => s.step === "answer" && s.phase === "done");
  const pipelineSteps = steps.filter((s) => s.step !== "answer");

  return (
    <section className={`lane accent-${accent}`}>
      <div className="lane-head">
        <h2>{title}</h2>
        {tag && <span className="lane-badge">{tag}</span>}
      </div>
      <p className="lane-sub">{subtitle}</p>

      {steps.length === 0 && (
        <div className="lane-empty">Run a query to watch this pipeline execute.</div>
      )}

      <ol className="steps">
        {pipelineSteps.map((s) => (
          <StepCard key={s.step} step={s} onPeek={onPeek} vectorParams={vectorParams} />
        ))}
      </ol>

      {answerStep && <AnswerCard payload={answerStep.payload} accent={accent} />}
    </section>
  );
}

function StepCard({ step, onPeek, vectorParams }) {
  const [open, setOpen] = useState(false);
  const done = step.phase === "done";
  const hasPayload = done && step.payload && Object.keys(step.payload).length > 0;

  return (
    <li className={`step ${done ? "step-done" : "step-running"}`}>
      <div className="step-head" onClick={() => hasPayload && setOpen((o) => !o)}>
        <span className={`dot ${done ? "dot-done" : "dot-spin"}`} />
        <span className="step-title">{step.title}</span>
        {done && <span className="step-dur">{step.duration_ms} ms</span>}
        {hasPayload && <span className="step-caret">{open ? "▾" : "▸"}</span>}
      </div>
      <p className="step-desc">{step.description}</p>
      <div className="step-actions">
        {step.source_symbol && (
          <button className="link-btn" onClick={() => onPeek(step.source_symbol)}>
            {"</>"} view source
          </button>
        )}
      </div>
      {open && hasPayload && (
        <div className="step-payload">
          <StepPayload engine={step.engine} step={step.step} payload={step.payload}
                       vectorParams={vectorParams} />
        </div>
      )}
    </li>
  );
}

// Render each step's intermediate data in the most legible form for its kind.
function StepPayload({ engine, step, payload, vectorParams }) {
  if (engine === "graph") {
    if (step === "parse_query") return <Chips items={payload.keywords} />;
    if (step === "seed_match") return <BillList bills={payload.seed_bills} />;
    if (step === "expand_graph")
      return (
        <div className="two-col">
          <div><h4>Legislators</h4><Chips items={payload.legislators} /></div>
          <div><h4>Committees</h4><Chips items={payload.committees} /></div>
        </div>
      );
    if (step === "score_precedents") return <PrecedentTable precedents={payload.precedents} />;
    if (step === "retrieve")
      return (
        <div>
          <div className="rate">
            {payload.enactment_rate != null
              ? `${Math.round(payload.enactment_rate * 100)}% of connected precedents became law`
              : "No precedent outcomes available"}
          </div>
          <GraphView subgraph={payload.subgraph} />
        </div>
      );
    if (step === "build_context") return <PromptBlock prompt={payload.prompt} />;
  }
  if (engine === "vector") {
    if (step === "embed_search")
      return (
        <div className="kv">
          <Row k="index" v={payload.index_mode} />
          <Row k="embedder" v={`${payload.embedder} (${payload.vector_dim} dims)`} />
          <Row k="chunks searched" v={payload.chunk_count} />
          <Row k="query vector (first 8)" v={JSON.stringify(payload.query_vector_preview)} />
        </div>
      );
    if (step === "retrieve") return <ChunkList chunks={payload.chunks} vectorParams={vectorParams} />;
    if (step === "build_context") return <PromptBlock prompt={payload.prompt} />;
  }
  return <pre className="raw">{JSON.stringify(payload, null, 2)}</pre>;
}

function AnswerCard({ payload, accent }) {
  const model = payload.generated
    ? (payload.model ? payload.model.replace(/^.*\//, "") : "generated")
    : "extractive fallback";
  return (
    <div className={`answer accent-${accent}`}>
      <div className="answer-head">
        <h3>Answer</h3>
        <span className={`answer-model ${payload.generated ? "badge-ok" : "badge-warn"}`}>{model}</span>
      </div>
      <p className="answer-text">{payload.answer}</p>
      {payload.note && <p className="answer-note muted">{payload.note}</p>}
    </div>
  );
}

function Chips({ items }) {
  if (!items || items.length === 0) return <span className="muted">none</span>;
  return (
    <div className="chips">
      {items.map((i) => <span key={i} className="tag">{i}</span>)}
    </div>
  );
}

function BillList({ bills }) {
  if (!bills || bills.length === 0) return <span className="muted">no seed bills matched</span>;
  return (
    <ul className="bill-list">
      {bills.map((b) => (
        <li key={b.id}>
          <b>{b.id}</b> — {b.title} <OutcomeChip outcome={b.outcome} />
        </li>
      ))}
    </ul>
  );
}

function PrecedentTable({ precedents }) {
  if (!precedents || precedents.length === 0) return <span className="muted">none</span>;
  return (
    <table className="ptable">
      <thead><tr><th>Bill</th><th>Score</th><th>Shared</th><th>Outcome</th></tr></thead>
      <tbody>
        {precedents.map((p) => (
          <tr key={p.id}>
            <td><b>{p.id}</b><div className="muted small">{p.title}</div></td>
            <td><span className="score-pill">{p.score}</span></td>
            <td className="small">
              {p.shared_legislators?.length ? `${p.shared_legislators.length} sponsor(s)` : ""}
              {p.shared_committees?.length ? ` ${p.shared_committees.length} committee(s)` : ""}
              {!p.shared_legislators?.length && !p.shared_committees?.length ? "subject only" : ""}
            </td>
            <td><OutcomeChip outcome={p.outcome} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PromptBlock({ prompt }) {
  return <pre className="prompt">{prompt}</pre>;
}

function Row({ k, v }) {
  return <div className="kv-row"><span className="kv-k">{k}</span><span className="kv-v">{String(v)}</span></div>;
}
