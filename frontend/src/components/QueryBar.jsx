import React from "react";

// The question input, the model toggle, and a few one-click example queries.
// The model dropdown is populated from GET /models, so the backend decides which
// models are offered; the choice is sent with every query and drives the answer
// step of both engines. Enter runs the query.
export default function QueryBar({
  query, setQuery, onRun, running, examples, models, model, setModel, llmEnabled,
}) {
  return (
    <section className="querybar">
      <div className="query-row">
        <input
          className="query-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onRun()}
          placeholder="Ask about a bill's prospects, e.g. 'Will a data privacy bill pass?'"
        />
        <div className="model-picker" title={llmEnabled
          ? "The model both engines use to write the grounded answer."
          : "Set ANTHROPIC_API_KEY to enable model generation. Without it, answers are extractive."}>
          <span className="model-label">model</span>
          <select
            className="model-select"
            value={model || ""}
            disabled={running || !models.length}
            onChange={(e) => setModel(e.target.value)}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </div>
        <button className="run-btn" onClick={onRun} disabled={running}>
          {running ? "Running…" : "Run both engines"}
        </button>
      </div>
      {!llmEnabled && (
        <div className="model-hint muted">
          No API key set — answers use the extractive fallback. The model toggle takes
          effect once ANTHROPIC_API_KEY is configured.
        </div>
      )}
      <div className="examples">
        <span className="muted">Try:</span>
        {examples.map((ex) => (
          <button key={ex} className="chip" onClick={() => setQuery(ex)} disabled={running}>
            {ex}
          </button>
        ))}
      </div>
    </section>
  );
}
