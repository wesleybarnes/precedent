import React, { useEffect, useState, useCallback } from "react";
import { getHealth, getParams, getModels, streamQuery } from "./api.js";
import QueryBar from "./components/QueryBar.jsx";
import Controls from "./components/Controls.jsx";
import Pipeline from "./components/Pipeline.jsx";
import SourceModal from "./components/SourceModal.jsx";
import GraphExplorer from "./components/GraphExplorer.jsx";
import PeerMode from "./components/PeerMode.jsx";

// The app has two views: "Compare" runs one question through both engines and
// shows each pipeline executing live; "Knowledge Graph" is a standalone explorer
// of the whole bill-relationship space. Peer mode is a slowed-down, code-level
// walkthrough of a completed run.

const EXAMPLES = [
  "How likely is a prescription drug pricing bill to become law?",
  "What happened to past clean energy tax credit bills?",
  "Will a consumer data privacy bill pass?",
  "How have skilled worker visa bills fared?",
];

function reduceStep(prev, ev) {
  const lane = prev[ev.engine] ? [...prev[ev.engine]] : [];
  const idx = lane.findIndex((s) => s.step === ev.step);
  const merged = idx >= 0 ? { ...lane[idx], ...ev } : { ...ev };
  if (idx >= 0) lane[idx] = merged;
  else lane.push(merged);
  lane.sort((a, b) => a.index - b.index);
  return { ...prev, [ev.engine]: lane };
}

// Fixed number of pipeline stages each engine emits — used to turn "steps done"
// into a real progress fraction.
const EXPECTED_STEPS = { graph: 7, vector: 4 };
const doneCount = (lane) => (lane || []).filter((s) => s.phase === "done").length;

export default function App() {
  const [health, setHealth] = useState(null);
  const [schema, setSchema] = useState(null);
  const [graphParams, setGraphParams] = useState(null);
  const [vectorParams, setVectorParams] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState(null);
  const [query, setQuery] = useState(EXAMPLES[0]);
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState({ graph: [], vector: [] });
  const [sourceSymbol, setSourceSymbol] = useState(null);
  const [error, setError] = useState(null);
  const [view, setView] = useState("compare");
  const [peerOpen, setPeerOpen] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => {
    let alive = true;
    let loadedSchema = false;
    const tick = async () => {
      try {
        const h = await getHealth();
        if (!alive) return;
        setHealth(h);
        setError(null);
        if (!loadedSchema) {
          loadedSchema = true;
          const [s, m] = await Promise.all([getParams(), getModels()]);
          if (!alive) return;
          setSchema(s);
          setGraphParams(s.graph.defaults);
          setVectorParams(s.vector.defaults);
          setModels(m.models || []);
          setModel(m.default);
        }
      } catch {
        if (!alive) return;
        setHealth(null);
        loadedSchema = false;
        setError("Cannot reach the API on :8080. In a second terminal at the repo root, run:  "
          + "source .venv/bin/activate && uvicorn precedent.api.main:app --port 8080  "
          + "(or just ./dev.sh to start both).");
      }
    };
    tick();
    const id = setInterval(tick, 4000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const run = useCallback(() => {
    if (!query.trim() || running) return;
    setSteps({ graph: [], vector: [] });
    setError(null);
    setRunning(true);
    setDone(false);
    streamQuery(query, graphParams, vectorParams, model, {
      onStep: (ev) => setSteps((prev) => reduceStep(prev, ev)),
      onDone: () => { setRunning(false); setDone(true); },
      onError: () => {
        setRunning(false);
        setError("Stream interrupted. Check that the API is running.");
      },
    });
  }, [query, graphParams, vectorParams, model, running]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">§</span>
          <div>
            <h1>Precedent</h1>
            <p className="tagline">GraphRAG vs Vector RAG — a legislative precedent visualizer</p>
          </div>
        </div>
        <div className="status">
          {health ? (
            <>
              <Badge ok label={`${health.bills_in_graph} bills · ${health.corpus_source || "seed"}`} />
              <Badge ok={health.llm_enabled}
                     label={health.llm_enabled ? "generation: on" : "extractive (no API key)"} />
              <Badge ok label={`graph: ${health.graph_backend}`} />
              <Badge ok label={`vector: ${health.chroma_mode}`} />
            </>
          ) : (
            <span className="muted">connecting…</span>
          )}
        </div>
      </header>

      <nav className="tabs">
        <button className={`tab ${view === "compare" ? "tab-on" : ""}`} onClick={() => setView("compare")}>
          ⚖ Compare engines
        </button>
        <button className={`tab ${view === "graph" ? "tab-on" : ""}`} onClick={() => setView("graph")}>
          🕸 Knowledge graph
        </button>
      </nav>

      {error && <div className="error-banner">{error}</div>}

      {view === "compare" && (
        <>
          <QueryBar
            query={query} setQuery={setQuery} onRun={run} running={running}
            examples={EXAMPLES} models={models} model={model} setModel={setModel}
            llmEnabled={health?.llm_enabled}
          />
          {schema && (
            <Controls
              schema={schema} graphParams={graphParams} vectorParams={vectorParams}
              setGraphParams={setGraphParams} setVectorParams={setVectorParams}
              disabled={running}
            />
          )}

          {(running || done) && (
            <RunProgress steps={steps} running={running} />
          )}

          {done && (
            <div className="peer-bar">
              <button className="peer-btn" onClick={() => setPeerOpen(true)}>
                🔬 Peer into the pipeline — slow-motion, step by step, down to the code
              </button>
            </div>
          )}

          <main className="lanes">
            <Pipeline
              title="GraphRAG" tag="structural"
              subtitle="Reasons over the knowledge graph: shared sponsors, committees, and outcomes."
              accent="graph" steps={steps.graph} onPeek={setSourceSymbol}
            />
            <Pipeline
              title="Vector RAG" tag="lexical"
              subtitle="Ranks bill passages by semantic (lexical) similarity to your question."
              accent="vector" steps={steps.vector} onPeek={setSourceSymbol}
              vectorParams={vectorParams}
            />
          </main>
        </>
      )}

      {view === "graph" && <GraphExplorer onPeek={setSourceSymbol} />}

      {sourceSymbol && <SourceModal symbol={sourceSymbol} onClose={() => setSourceSymbol(null)} />}
      {peerOpen && <PeerMode steps={steps} onClose={() => setPeerOpen(false)} />}

      <footer className="footer">
        <span>Click <b>view source</b> on any step to see the exact code that ran.</span>
        <span className="muted">Turn the knobs, re-run, and watch retrieval change.</span>
      </footer>
    </div>
  );
}

function Badge({ ok, label }) {
  return <span className={`badge ${ok ? "badge-ok" : "badge-warn"}`}>{label}</span>;
}

// Live retrieval progress, driven by how many pipeline stages have completed
// across both engines. Shows an overall bar plus each engine's own progress and
// the stage it's currently on.
function RunProgress({ steps, running }) {
  const g = doneCount(steps.graph);
  const v = doneCount(steps.vector);
  const overall = Math.min(1, (g + v) / (EXPECTED_STEPS.graph + EXPECTED_STEPS.vector));
  const label = (lane, expected) => {
    const active = (lane || []).find((s) => s.phase === "start" && !lane.find((d) => d.step === s.step && d.phase === "done"));
    const current = (lane || []).filter((s) => s.phase === "done").slice(-1)[0];
    return running ? (active?.title || current?.title || "starting…") : "done";
  };
  return (
    <div className="run-progress">
      <div className="rp-head">
        <span className="rp-title">{running ? "Retrieving…" : "Retrieval complete"}</span>
        <span className="rp-pct">{Math.round(overall * 100)}%</span>
      </div>
      <div className={`rp-bar ${running ? "rp-animated" : ""}`}>
        <div className="rp-fill" style={{ width: `${overall * 100}%` }} />
      </div>
      <div className="rp-lanes">
        <LaneProgress accent="graph" name="GraphRAG" done={g} total={EXPECTED_STEPS.graph}
                      label={label(steps.graph)} />
        <LaneProgress accent="vector" name="Vector RAG" done={v} total={EXPECTED_STEPS.vector}
                      label={label(steps.vector)} />
      </div>
    </div>
  );
}

function LaneProgress({ accent, name, done, total, label }) {
  const frac = Math.min(1, done / total);
  return (
    <div className={`rp-lane accent-${accent}`}>
      <div className="rp-lane-head">
        <span className="rp-lane-name">{name}</span>
        <span className="rp-lane-step muted">{label} · {done}/{total}</span>
      </div>
      <div className="rp-lane-bar"><div className="rp-lane-fill" style={{ width: `${frac * 100}%` }} /></div>
    </div>
  );
}
