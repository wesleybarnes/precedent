// Shared visual language: outcome colors and a similarity→heat scale, used
// across the graph, the chunk list, and the chunk map so the same fact reads
// the same colour everywhere.

// Outcome color-coding: green = made it into law, red = actively killed,
// amber = still alive, slate = died quietly. Legislative status at a glance.
export const OUTCOME_STYLE = {
  became_law: { color: "#0a7d3d", bg: "#e6f6ec", label: "Became law" },
  vetoed: { color: "#c02626", bg: "#fdeaea", label: "Vetoed" },
  died_after_passing_both_chambers: { color: "#b5480b", bg: "#fdefe3", label: "Passed both, died" },
  passed_both_chambers_pending: { color: "#0a66c2", bg: "#e7f0fb", label: "Passed both (pending)" },
  died_after_passing_one_chamber: { color: "#b5480b", bg: "#fdefe3", label: "Passed one, died" },
  passed_one_chamber_pending: { color: "#0a66c2", bg: "#e7f0fb", label: "Passed one (pending)" },
  died_in_committee: { color: "#5b6472", bg: "#eef0f3", label: "Died in committee" },
  pending_in_committee: { color: "#8a6d1f", bg: "#fbf3dd", label: "Pending in committee" },
};

export function outcomeStyle(outcome) {
  return OUTCOME_STYLE[outcome] || { color: "#5b6472", bg: "#eef0f3", label: outcome || "Unknown" };
}

// Map a cosine similarity (~0..1 on this data) to a warm heat colour for the
// vector side. Higher similarity = hotter, more saturated.
export function heatColor(similarity, alpha = 1) {
  const t = Math.max(0, Math.min(1, similarity / 0.5)); // stretch: sims here top out ~0.4
  // interpolate hue from amber (45) to deep orange (18)
  const hue = 45 - t * 27;
  const light = 92 - t * 34;
  const sat = 70 + t * 25;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

// Node colours for the graph, tuned for a light background.
export const NODE_COLORS = {
  seed_bill: "#f59e0b",
  precedent_bill: "#2563eb",
  legislator: "#16a34a",
  committee: "#9333ea",
};
export const NODE_LABEL = {
  seed_bill: "Seed bill",
  precedent_bill: "Precedent bill",
  legislator: "Legislator",
  committee: "Committee",
};
