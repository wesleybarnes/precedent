import React from "react";
import { outcomeStyle } from "../theme.js";

// A small colour-coded pill for a bill's outcome, used everywhere a bill is
// shown (seed list, precedent table, chunk list). Green = law, red = killed,
// blue = advanced, amber/slate = pending/died quietly.
export default function OutcomeChip({ outcome }) {
  const s = outcomeStyle(outcome);
  return <span className="outcome-chip" style={{ color: s.color, background: s.bg }}>{s.label}</span>;
}
