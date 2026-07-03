import React from "react";

// A small ⓘ that reveals an explanation on hover/focus. Used throughout to make
// the app self-teaching: every knob and key term can explain what it is and its
// trade-offs without cluttering the layout.
export default function InfoTip({ text, side = "top" }) {
  if (!text) return null;
  return (
    <span className="infotip" tabIndex={0}>
      <span className="infotip-mark">i</span>
      <span className={`infotip-pop infotip-${side}`}>{text}</span>
    </span>
  );
}
